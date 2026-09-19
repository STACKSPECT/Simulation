from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .com_markers import ComMarkers
from .controls import SPEED_SCALE, RunCancelled, ViewerControls
from .models import Package, Placement, StackState
from .pallet_com import PalletCom, com_from_world_points, compare_pallet_com, estimate_pallet_com
from .planner import Candidate, StablePalletPlanner
from .playback import Playback
from .scenario import Scenario, ShakeConfig
from .shake import (
    EURO_PALLET_MASS_KG,
    GRAVITY,
    attach_trial_scores,
    axis_unit,
    package_fell_off,
    pallet_tilt_deg,
    sine_jolt_force,
    summarise_beam,
    summarise_trials,
    transport_reason,
)
from .stability import validate_stack
from .suction import SuctionArray
from .truck import TruckSlot, load_summary, plan_truck_load, unload_order
from .visual_assets import UR10E_MESH_FILES, load_ur10e_mesh_assets


@dataclass(frozen=True, slots=True)
class HeldPackage:
    """A package carried rigidly by the tool, with its pose in the tool frame."""

    index: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ComMeasurement:
    """What the probe found for one package, next to what the scenario declared."""

    package_id: str
    measured_mass: float
    measured_com: tuple[float, float, float]
    declared_com: tuple[float, float, float]
    error_mm: float
    error_xy_mm: float
    normalised: tuple[float, float, float]
    trustworthy: bool
    replanned: bool
    hidden_mm: float = 0.0
    duration_s: float = 0.0
    method: str = "in_situ"


@dataclass(frozen=True, slots=True)
class PlacementMeasurement:
    package_id: str
    position_error_m: float
    yaw_error_deg: float
    tilt_deg: float
    active_cups: int
    grip_capacity_ratio: float


@dataclass(frozen=True, slots=True)
class PickPose:
    """Where the tool has to go to take one carton, measured off the scene.

    `obstacle_top` is the highest thing left standing next to it -- the conveyor deck, or
    whatever is still stacked in the trailer. Nothing the arm carries may come down below
    that until it is clear of the bay.
    """

    index: int
    x: float
    y: float
    top_z: float
    yaw: float
    obstacle_top: float


def _rgba(index: int, alpha: float = 1.0) -> str:
    colors = (
        (0.14, 0.55, 0.86),
        (0.95, 0.55, 0.16),
        (0.28, 0.72, 0.47),
        (0.76, 0.31, 0.42),
        (0.53, 0.38, 0.78),
        (0.88, 0.74, 0.20),
    )
    red, green, blue = colors[index % len(colors)]
    return f"{red} {green} {blue} {alpha}"


def _cardboard_rgba(index: int, alpha: float = 1.0) -> str:
    colors = (
        (0.64, 0.43, 0.23),
        (0.72, 0.51, 0.29),
        (0.58, 0.38, 0.20),
        (0.76, 0.57, 0.34),
    )
    red, green, blue = colors[index % len(colors)]
    return f"{red} {green} {blue} {alpha}"


def _package_xml(
    package: Package,
    index: int,
    held: HeldPackage | None = None,
    *,
    simplified_graphics: bool,
) -> str:
    """Package body, either free on the floor or rigidly carried by the tool.

    A sealed vacuum cup is a rigid joint, and modelling it as one is not just simpler: a
    weld equality biases the torque the wrist sensor reports, badly enough at these masses
    to make a centre-of-mass measurement meaningless. Re-parenting the body removes the
    constraint altogether, so there is nothing left to mis-account for.
    """
    width, depth, height = package.size
    ix = package.mass * (depth**2 + height**2) / 12
    iy = package.mass * (width**2 + height**2) / 12
    iz = package.mass * (width**2 + depth**2) / 12
    if simplified_graphics:
        visuals = ""
        color = _rgba(index, 0.0)
    else:
        color = _cardboard_rgba(index, 0.0)
        label_width = min(width * 0.27, 0.105)
        label_height = min(height * 0.28, 0.055)
        tape_width = min(width * 0.10, 0.045)
        visuals = f"""
      <geom name="package_tape_{index}" type="box"
            pos="0 0 {height / 2 + 0.001}" size="{tape_width / 2} {depth / 2} 0.0015"
            rgba="0.73 0.61 0.40 0" contype="0" conaffinity="0"/>
      <geom name="package_label_{index}" type="box"
            pos="{width * 0.18} {-depth / 2 - 0.001} {height * 0.08}"
            size="{label_width / 2} 0.0015 {label_height / 2}"
            rgba="0.94 0.95 0.91 0" contype="0" conaffinity="0"/>
      <geom name="package_code_{index}" type="box"
            pos="{width * 0.18} {-depth / 2 - 0.0027} {height * 0.08}"
            size="{label_width * 0.32} 0.0007 {label_height * 0.12}"
            rgba="{_rgba(index, 0.0)}" contype="0" conaffinity="0"/>"""
    inertial = (
        f'<inertial pos="{package.com[0]} {package.com[1]} {package.com[2]}" '
        f'mass="{package.mass}" diaginertia="{ix} {iy} {iz}"/>'
    )
    geom = (
        f'<geom name="package_geom_{index}" type="box" size="{width / 2} {depth / 2} {height / 2}" '
        f'density="0" friction="{package.friction} 0.01 0.001" contype="1" conaffinity="1" '
        f'rgba="{color}"/>'
    )
    if held is not None and held.index == index:
        position = " ".join(f"{value}" for value in held.position)
        quaternion = " ".join(f"{value}" for value in held.quaternion)
        return f"""
    <body name="package_{index}" pos="{position}" quat="{quaternion}">
      {inertial}
      {geom}
      {visuals}
    </body>"""
    parked_x = -3.0 - index * 0.7
    return f"""
    <body name="package_{index}" pos="{parked_x} 0 {height / 2 + 0.002}">
      <freejoint name="package_joint_{index}"/>
      {inertial}
      {geom}
      {visuals}
    </body>"""


def _weld_xml(index: int) -> str:
    # relpose is replaced with the measured transform immediately before grasping.
    #
    # torquescale stays at 1: sixteen cups on a 264 x 184 mm footprint hold a carton flat,
    # and a slack value lets a knock spin it on the tool -- 5 Nm flips it right over at
    # 0.15 against 1.3 deg here. The weld only ever carries a package when the centre of
    # mass is not being measured, since `ComProbe.grasp` re-parents the body instead, so
    # there is no wrist reading for a stiff weld to bias.
    return (
        f'<weld name="suction_{index}" body1="tool" body2="package_{index}" '
        'relpose="0 0 0 1 0 0 0" torquescale="1" active="false"/>'
    )


def _robot_visuals(simplified_graphics: bool, *items: tuple[str, str]) -> str:
    if simplified_graphics:
        return ""
    return "\n".join(f'<geom mesh="{mesh}" material="{material}" class="robot_visual"/>' for mesh, material in items)


def _collision_visibility(simplified_graphics: bool, material: str) -> str:
    if simplified_graphics:
        return f'material="{material}"'
    return 'rgba="0 0 0 0" group="3"'


def _ur10e_xml(cups: str, carried: str = "", *, simplified_graphics: bool) -> str:
    """UR10e with Menagerie's official kinematics, rendered with meshes or primitives."""
    base_visual = _robot_visuals(simplified_graphics, ("base_0", "black"), ("base_1", "jointgray"))
    shoulder_visual = _robot_visuals(
        simplified_graphics,
        ("shoulder_0", "urblue"),
        ("shoulder_1", "black"),
        ("shoulder_2", "jointgray"),
    )
    upper_visual = _robot_visuals(
        simplified_graphics,
        ("upperarm_0", "black"),
        ("upperarm_1", "jointgray"),
        ("upperarm_2", "urblue"),
        ("upperarm_3", "linkgray"),
    )
    forearm_visual = _robot_visuals(
        simplified_graphics,
        ("forearm_0", "urblue"),
        ("forearm_1", "black"),
        ("forearm_2", "jointgray"),
        ("forearm_3", "linkgray"),
    )
    wrist1_visual = _robot_visuals(
        simplified_graphics,
        ("wrist1_0", "black"),
        ("wrist1_1", "urblue"),
        ("wrist1_2", "jointgray"),
    )
    wrist2_visual = _robot_visuals(
        simplified_graphics,
        ("wrist2_0", "black"),
        ("wrist2_1", "urblue"),
        ("wrist2_2", "jointgray"),
    )
    wrist3_visual = _robot_visuals(simplified_graphics, ("wrist3", "linkgray"))
    base_collision = _collision_visibility(simplified_graphics, "ur_joint")
    blue_collision = _collision_visibility(simplified_graphics, "ur_blue")
    light_collision = _collision_visibility(simplified_graphics, "ur_light")
    joint_collision = _collision_visibility(simplified_graphics, "ur_joint")
    gripper = SuctionArray()
    body_x, body_y = gripper.body_length / 2, gripper.body_width / 2
    if simplified_graphics:
        tool = f"""
                    <geom type="box" size="{body_x} {body_y} 0.026" pos="0 0 0.040" material="tool"
                          contype="0" conaffinity="0"/>
                    {cups}"""
    else:
        tool = f"""
                    <geom name="eoat_qc" type="cylinder" pos="0 0 0.007"
                          size="0.0315 0.007" material="jointgray" contype="0" conaffinity="0"/>
                    <geom name="vgp20_body" type="box" pos="0 0 0.040"
                          size="{body_x} {body_y} 0.026" material="tool" contype="0" conaffinity="0"/>
                    <geom name="vgp20_lid" type="box" pos="0 0 0.018"
                          size="{body_x - 0.014} {body_y - 0.014} 0.006" material="jointgray"
                          contype="0" conaffinity="0"/>
                    <geom name="vgp20_accent" type="box" pos="0 {body_y + 0.0015} 0.040"
                          size="0.055 0.0015 0.014" material="onrobot_orange"
                          contype="0" conaffinity="0"/>
                    <geom name="vgp20_vent_p" type="box" pos="{body_x + 0.0015} 0 0.040"
                          size="0.003 0.050 0.014" material="aluminium" contype="0" conaffinity="0"/>
                    <geom name="vgp20_vent_n" type="box" pos="{-body_x - 0.0015} 0 0.040"
                          size="0.003 0.050 0.014" material="aluminium" contype="0" conaffinity="0"/>
                    {cups}"""
    return f"""
    <body name="ur10e_base" pos="0.60 -0.72 0.48" quat="0 0 0 -1" childclass="ur10e">
      <inertial mass="4.0" pos="0 0 0" diaginertia="0.006106 0.006106 0.01125"/>
      {base_visual}
      <geom type="cylinder" size="0.095 0.075" pos="0 0 -0.02" {base_collision}/>
      <body name="shoulder_link" pos="0 0 0.181">
        <inertial pos="0 0 0" mass="7.778" diaginertia="0.0314743 0.0314743 0.0218756"/>
        <joint name="shoulder_pan_joint" class="size4" axis="0 0 1"/>
        {shoulder_visual}
        <geom type="cylinder" size="0.088 0.09" {blue_collision}/>
        <body name="upper_arm_link" pos="0 0.176 0" quat="1 0 1 0">
          <inertial pos="0 0 0.3065" mass="12.93" diaginertia="0.423074 0.423074 0.036366"/>
          <joint name="shoulder_lift_joint" class="size4"/>
          {upper_visual}
          <geom type="cylinder" size="0.084 0.09" pos="0 -0.05 0" quat="1 1 0 0" {joint_collision}/>
          <geom type="capsule" size="0.068" fromto="0 0 0.05 0 0 0.59" {blue_collision}/>
          <body name="forearm_link" pos="0 -0.137 0.613">
            <inertial pos="0 0 0.2855" mass="3.87" diaginertia="0.11059 0.11059 0.010884"/>
            <joint name="elbow_joint" class="size3_limited"/>
            {forearm_visual}
            <geom type="cylinder" size="0.074 0.075" pos="0 0.07 0" quat="1 1 0 0" {joint_collision}/>
            <geom type="capsule" size="0.055" fromto="0 0 0.05 0 0 0.55" {light_collision}/>
            <body name="wrist_1_link" pos="0 0 0.571" quat="1 0 1 0">
              <inertial pos="0 0.135 0" quat="0.5 0.5 -0.5 0.5" mass="1.96"
                        diaginertia="0.0055125 0.0051083 0.0051083"/>
              <joint name="wrist_1_joint" class="size2"/>
              {wrist1_visual}
              <geom type="cylinder" size="0.058 0.07" pos="0 0.06 0" quat="1 1 0 0" {joint_collision}/>
              <body name="wrist_2_link" pos="0 0.135 0">
                <inertial pos="0 0 0.12" quat="0.5 0.5 -0.5 0.5" mass="1.96"
                          diaginertia="0.0055125 0.0051083 0.0051083"/>
                <joint name="wrist_2_joint" axis="0 0 1" class="size2"/>
                {wrist2_visual}
                <geom type="cylinder" size="0.052 0.065" pos="0 0 0.05" {blue_collision}/>
                <body name="wrist_3_link" pos="0 0 0.12">
                  <inertial pos="0 0.092 0" quat="0 1 -1 0" mass="0.202"
                            diaginertia="0.0002045 0.0001443 0.0001443"/>
                  <joint name="wrist_3_joint" class="size2"/>
                  {wrist3_visual}
                  <geom type="cylinder" size="0.050 0.075" pos="0 0.055 0" quat="1 1 0 0" {joint_collision}/>
                  <body name="tool" pos="0 0.10 0" quat="-1 1 0 0">
                    <inertial pos="0 0 0.038" mass="{gripper.mass}"
                              diaginertia="0.0081 0.0158 0.0221"/>
                    <!-- Wrist force/torque sensor. Sitting at the origin of `tool`, whose
                         parent is `wrist_3_link`, it reads the wrench crossing the flange:
                         tool plus whatever the vacuum is holding. -->
                    <site name="ft_site" pos="0 0 0" size="0.006" rgba="0 0 0 0"/>
                    {tool}
                    <site name="suction_site" pos="0 0 {gripper.flange_to_cup}"
                          size="0.008" rgba="1 0 0 0"/>
                    {carried}
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _pallet_xml(scenario: Scenario, *, simplified_graphics: bool) -> str:
    """Euro-pallet as a dynamic body that can be jolted or tipped after stacking.

    Slides keep the deck level for transport jolts. Hinges stay locked until the beam
    trial, when the loaded pallet must balance on a narrow fulcrum. A world weld holds
    it still while the robot works. Visual geoms carry no mass so the 25 kg EUR-pallet
    inertia is the one that moves.
    """
    sim = scenario.simulation
    pallet = scenario.pallet
    center_x = sim.pallet_origin[0] + pallet.width / 2
    center_y = sim.pallet_origin[1] + pallet.depth / 2
    height = sim.pallet_height
    mass = EURO_PALLET_MASS_KG
    ixx = mass * (pallet.depth**2 + height**2) / 12
    iyy = mass * (pallet.width**2 + height**2) / 12
    izz = mass * (pallet.width**2 + pallet.depth**2) / 12
    inertial = (
        f'<inertial pos="0 0 {height / 2}" mass="{mass}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>'
    )
    joints = """
      <joint name="pallet_x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="pallet_y" type="slide" axis="0 1 0" damping="2"/>
      <joint name="pallet_z" type="slide" axis="0 0 1" damping="2"/>
      <joint name="pallet_rx" type="hinge" axis="1 0 0" damping="1" limited="true" range="-0.0001 0.0001"/>
      <joint name="pallet_ry" type="hinge" axis="0 1 0" damping="1" limited="true" range="-0.0001 0.0001"/>"""
    if simplified_graphics:
        geoms = f"""<geom name="pallet" type="box" pos="0 0 {height / 2}"
          size="{pallet.width / 2} {pallet.depth / 2} {height / 2}"
          density="0" rgba="0.48 0.28 0.12 1" friction="1.1 0.01 0.001" conaffinity="3"/>"""
    else:
        top_z = height - 0.011
        bottom_z = 0.011
        top_boards = "\n".join(
            f'<geom name="pallet_top_{index}" type="box" pos="0 {offset} {top_z}" '
            f'size="{pallet.width / 2} 0.055 0.011" material="pallet_wood" density="0" '
            'contype="0" conaffinity="0"/>'
            for index, offset in enumerate((-0.335, -0.17, 0.0, 0.17, 0.335))
        )
        bottom_boards = "\n".join(
            f'<geom name="pallet_runner_{index}" type="box" pos="0 {offset} {bottom_z}" '
            f'size="{pallet.width / 2} 0.048 0.011" material="pallet_wood_dark" density="0" '
            'contype="0" conaffinity="0"/>'
            for index, offset in enumerate((-0.33, 0.0, 0.33))
        )
        blocks = "\n".join(
            f'<geom type="box" pos="{x} {y} {height / 2}" '
            'size="0.075 0.060 0.038" material="pallet_wood_dark" density="0" '
            'contype="0" conaffinity="0"/>'
            for x in (-0.49, 0.0, 0.49)
            for y in (-0.33, 0.0, 0.33)
        )
        geoms = f"""
      <geom name="pallet" type="box" pos="0 0 {height / 2}"
            size="{pallet.width / 2} {pallet.depth / 2} {height / 2}"
            density="0" rgba="0 0 0 0" friction="1.1 0.01 0.001" conaffinity="3"/>
      {top_boards}
      {bottom_boards}
      {blocks}"""
    return f"""
    <body name="pallet" pos="{center_x} {center_y} 0">
      {inertial}
      {joints}
      {geoms}
    </body>"""


def _beam_xml(scenario: Scenario) -> str:
    """Narrow fulcrum parked below the floor until a beam trial stages it.

    Collision bits have to be on in the XML: flipping `contype` after compile does not
    create contacts in this MuJoCo build. Orientation for the Y trial is a mocap yaw.
    """
    config = scenario.simulation.shake
    length = max(scenario.pallet.width, scenario.pallet.depth) / 2 + 0.08
    return f"""
    <body name="beam" mocap="true" pos="0 0 -1">
      <geom name="beam" type="box" size="{length} {config.beam_width / 2} {config.beam_height / 2}"
            rgba="0.78 0.28 0.14 0" friction="1.2 0.02 0.001"
            density="0" contype="1" conaffinity="1"/>
    </body>"""


def _infeed_xml(scenario: Scenario, *, simplified_graphics: bool) -> str:
    sim = scenario.simulation
    if sim.truck is not None:
        return ""
    x, y = sim.infeed_position
    if simplified_graphics:
        return f"""<geom name="infeed" type="box" pos="{x} {y} {sim.infeed_height / 2}"
          size="0.36 0.34 {sim.infeed_height / 2}" rgba="0.30 0.33 0.37 1"
          conaffinity="3"/>"""
    rollers = "\n".join(
        f'<geom type="cylinder" pos="{x + offset} {y} {sim.infeed_height - 0.025}" '
        'quat="0.7071 0.7071 0 0" size="0.026 0.31" material="roller" contype="0" conaffinity="0"/>'
        for offset in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
    )
    legs = "\n".join(
        f'<geom type="box" pos="{x + dx} {y + dy} {sim.infeed_height / 2 - 0.04}" '
        f'size="0.035 0.035 {sim.infeed_height / 2 - 0.04}" material="frame" contype="0" conaffinity="0"/>'
        for dx in (-0.31, 0.31)
        for dy in (-0.29, 0.29)
    )
    return f"""
    <geom name="infeed" type="box" pos="{x} {y} {sim.infeed_height / 2}"
          size="0.36 0.34 {sim.infeed_height / 2}" rgba="0 0 0 0" conaffinity="3"/>
    <geom type="box" pos="{x} {y - 0.335} {sim.infeed_height - 0.055}"
          size="0.37 0.035 0.075" material="frame" contype="0" conaffinity="0"/>
    <geom type="box" pos="{x} {y + 0.335} {sim.infeed_height - 0.055}"
          size="0.37 0.035 0.075" material="frame" contype="0" conaffinity="0"/>
    {rollers}
    {legs}"""


# Clear width between the load and the far wall of the body. A carton swings out this
# far while the wrist tilts to weigh it, and the body is solid, so anything closer is
# something the cell would hit.
_AISLE = 0.46

# The trailer is structure, not scenery: a carton that swings into it hits it, and so
# does the arm. `conaffinity=3` picks up both the packages (contact class 1) and the arm
# (class 2). The pallet is excluded from world contacts, so a jolted pallet is still
# free to slide -- the trailer never becomes an invisible end stop for the shake test.
_SOLID = 'conaffinity="3" friction="0.7 0.01 0.001"'
# Trim: sits on a panel that already carries the contacts, or out of everyone's way.
_TRIM = 'contype="0" conaffinity="0"'

_TRUCK_MATERIALS = """<material name="truck_deck" rgba="0.34 0.28 0.22 1" roughness="0.85"/>
    <material name="truck_deck_line" rgba="0.24 0.19 0.15 1" roughness="0.85"/>
    <material name="truck_body" rgba="0.87 0.88 0.90 1" roughness="0.45"/>
    <material name="truck_rib" rgba="0.78 0.79 0.82 1" roughness="0.45"/>
    <material name="truck_door" rgba="0.82 0.84 0.87 1" roughness="0.45"/>
    <material name="truck_frame" rgba="0.36 0.39 0.43 1" metallic="0.5" roughness="0.35"/>"""


def _truck_xml(scenario: Scenario, *, simplified_graphics: bool) -> str:
    """The trailer backed onto the dock, seen from inside through its open doors.

    Only the deck carries collisions. The body is drawn around the bay the arm can
    actually reach, and the kerb side is left open -- curtain drawn back -- because the
    arm crosses that line on every single cycle and a panel there would be a wall the
    cell spends its whole run inside of.

    The far side is a real wall, so it has to stand clear of everything the cell swings.
    That is not the bay: weighing a carton tilts it about a third of a metre out past
    where it was picked from, and a body drawn tight around the load is a body the
    carton passes straight through. `_AISLE` is that swing, measured on the cell, and it
    also happens to be what the far half of a trailer looks like once it is empty.
    """
    bay = scenario.simulation.truck
    if bay is None:
        return ""
    x0, x1, y0, y1 = bay.bounds
    deck = bay.floor_height
    rear = x1 + 0.08
    cab = x0 - 1.30
    far = y0 - _AISLE
    kerb = y1 + 0.035
    wall = 1.30
    deck_geom = (
        f'<geom name="truck_deck" type="box" pos="{(cab + rear) / 2} {(far + kerb) / 2} {deck / 2}" '
        f'size="{(rear - cab) / 2} {(kerb - far) / 2} {deck / 2}" '
    )
    if simplified_graphics:
        return f"""
    {deck_geom}rgba="0.26 0.28 0.31 1" friction="0.9 0.01 0.001" conaffinity="3"/>
    <geom name="truck_far_wall" type="box" pos="{(cab + rear) / 2} {far} {deck + wall / 2}"
          size="{(rear - cab) / 2} 0.02 {wall / 2}" rgba="0.80 0.82 0.85 1" {_SOLID}/>
    <geom name="truck_kerb_rail" type="box" pos="{(x0 - 0.25 + rear) / 2} {kerb} {deck + 0.055}"
          size="{(rear - x0 + 0.25) / 2} 0.02 0.055" rgba="0.80 0.82 0.85 1" {_SOLID}/>"""
    planks = "\n".join(
        f'<geom type="box" pos="{(cab + rear) / 2} {y} {deck + 0.0015}" '
        f'size="{(rear - cab) / 2} 0.004 0.0015" material="truck_deck_line" {_TRIM}/>'
        for y in np.linspace(far + 0.10, kerb - 0.10, 7)
    )
    # Corrugated body panel: uprights every 0.24 m, the way a box trailer is ribbed.
    ribs = "\n".join(
        f'<geom type="box" pos="{x} {far - 0.022} {deck + wall / 2}" '
        f'size="0.012 0.008 {wall / 2 - 0.02}" material="truck_rib" {_TRIM}/>'
        for x in np.arange(cab + 0.12, rear - 0.05, 0.24)
    )
    # Body panels only close up past the bay: the arm crosses the kerb line and the roof
    # line on every cycle, so the working end is an open curtain side and the enclosed
    # trailer starts a carton's width behind where the arm stops reaching.
    deep = x0 - 0.25
    return f"""
    {deck_geom}rgba="0 0 0 0" friction="0.9 0.01 0.001" conaffinity="3"/>
    <geom name="truck_deck_face" type="box" pos="{(cab + rear) / 2} {(far + kerb) / 2} {deck / 2}"
          size="{(rear - cab) / 2} {(kerb - far) / 2} {deck / 2 - 0.001}" material="truck_deck"
          {_TRIM}/>
    {planks}
    <geom name="truck_far_wall" type="box" pos="{(cab + rear) / 2} {far - 0.012} {deck + wall / 2}"
          size="{(rear - cab) / 2} 0.012 {wall / 2}" material="truck_body"
          {_SOLID}/>
    {ribs}
    <geom name="truck_kerb_wall" type="box" pos="{(cab + deep) / 2} {kerb + 0.012} {deck + wall / 2}"
          size="{(deep - cab) / 2} 0.012 {wall / 2}" material="truck_body"
          {_SOLID}/>
    <geom name="truck_roof" type="box" pos="{(cab + deep) / 2} {(far + kerb) / 2} {deck + wall + 0.012}"
          size="{(deep - cab) / 2} {(kerb - far) / 2 + 0.024} 0.012" material="truck_body"
          {_SOLID}/>
    <geom name="truck_bulkhead" type="box" pos="{cab - 0.012} {(far + kerb) / 2} {deck + wall / 2}"
          size="0.012 {(kerb - far) / 2} {wall / 2}" material="truck_rib"
          {_SOLID}/>
    <geom name="truck_kerb_rail" type="box" pos="{(deep + rear) / 2} {kerb + 0.012} {deck + 0.055}"
          size="{(rear - deep) / 2} 0.012 0.055" material="truck_body"
          {_SOLID}/>
    <geom name="truck_post_far" type="box" pos="{rear + 0.03} {far - 0.012} {deck + wall / 2}"
          size="0.03 0.026 {wall / 2}" material="truck_frame" {_SOLID}/>
    <geom name="truck_post_kerb" type="box" pos="{rear + 0.03} {kerb + 0.012} {deck + 0.30}"
          size="0.03 0.026 0.30" material="truck_frame" {_SOLID}/>
    <!-- One rear door swung right back against the body; the other is out of frame. -->
    <geom name="truck_door" type="box" pos="{rear - 0.42} {far - 0.06} {deck + wall / 2}"
          size="0.40 0.016 {wall / 2 - 0.01}" material="truck_door" {_SOLID}/>
    <geom name="truck_door_bar_0" type="box" pos="{rear - 0.55} {far - 0.082} {deck + wall / 2}"
          size="0.012 0.008 {wall / 2 - 0.06}" material="truck_frame" {_TRIM}/>
    <geom name="truck_door_bar_1" type="box" pos="{rear - 0.29} {far - 0.082} {deck + wall / 2}"
          size="0.012 0.008 {wall / 2 - 0.06}" material="truck_frame" {_TRIM}/>
    <geom name="dock_bumper_far" type="box" pos="{rear + 0.09} {far + 0.10} {deck / 2}"
          size="0.055 0.075 {deck / 2 + 0.02}" material="black" {_SOLID}/>
    <geom name="dock_bumper_kerb" type="box" pos="{rear + 0.09} {kerb - 0.10} {deck / 2}"
          size="0.055 0.075 {deck / 2 + 0.02}" material="black" {_SOLID}/>"""


def _industrial_scene_xml(*, simplified_graphics: bool) -> str:
    if simplified_graphics:
        return ""
    posts = "\n".join(
        f'<geom type="box" pos="{x} 0.92 0.76" size="0.035 0.035 0.76" material="safety_yellow" contype="0" conaffinity="0"/>'
        for x in (-1.65, -0.85, -0.05, 0.75, 1.55)
    )
    rails = "\n".join(
        f'<geom type="box" pos="0 0.92 {z}" size="1.65 0.018 0.012" material="fence" contype="0" conaffinity="0"/>'
        for z in (0.22, 0.47, 0.72, 0.97, 1.22, 1.47)
    )
    verticals = "\n".join(
        f'<geom type="box" pos="{x} 0.92 0.84" size="0.009 0.018 0.64" material="fence" contype="0" conaffinity="0"/>'
        for x in np.linspace(-1.55, 1.45, 16)
    )
    markings = "\n".join(
        f'<geom type="box" pos="{x} -0.72 0.003" size="0.11 0.025 0.003" '
        'euler="0 0 -0.55" material="safety_yellow" contype="0" conaffinity="0"/>'
        for x in np.linspace(-0.20, 1.40, 9)
    )
    return f"{posts}\n{rails}\n{verticals}\n{markings}"


def _cup_xml(*, simplified_graphics: bool) -> str:
    gripper = SuctionArray()
    cups: list[str] = []
    lip_z = gripper.flange_to_cup - gripper.cup_radius * 0.3
    stem_z = lip_z - 0.014
    for column, row, x, y in gripper.cup_offsets():
        if simplified_graphics:
            cups.append(
                f'<geom name="cup_{column}_{row}" type="cylinder" pos="{x} {y} {lip_z}" '
                f'size="{gripper.cup_radius} 0.012" rgba="0.10 0.85 0.90 1" '
                'contype="0" conaffinity="0"/>'
            )
        else:
            cups.append(
                f'<geom name="cup_stem_{column}_{row}" type="cylinder" pos="{x} {y} {stem_z}" '
                f'size="0.007 0.010" material="aluminium" contype="0" conaffinity="0"/>'
                f'<geom name="cup_{column}_{row}" type="cylinder" pos="{x} {y} {lip_z}" '
                f'size="{gripper.cup_radius} 0.006" rgba="0.08 0.12 0.14 1" '
                'contype="0" conaffinity="0"/>'
            )
    return "\n".join(cups)

def build_mjcf(
    scenario: Scenario,
    held: HeldPackage | None = None,
    *,
    simplified_graphics: bool = False,
) -> str:
    held_index = held.index if held is not None else -1
    packages = "\n".join(
        _package_xml(package, i, held, simplified_graphics=simplified_graphics)
        for i, package in enumerate(scenario.packages)
        if i != held_index
    )
    carried = "".join(
        _package_xml(package, i, held, simplified_graphics=simplified_graphics)
        for i, package in enumerate(scenario.packages)
        if i == held_index
    )
    # A carried package hangs off `tool`, so the weld that would hold it has nothing left
    # to constrain and is left out.
    welds = "\n".join(_weld_xml(i) for i, _package in enumerate(scenario.packages) if i != held_index)
    pallet_weld = (
        '<weld name="pallet_anchor" body1="pallet" solref="0.002 1" '
        'solimp="0.95 0.99 0.001 0.5 2"/>'
    )
    cups = _cup_xml(simplified_graphics=simplified_graphics)
    robot = _ur10e_xml(cups, carried, simplified_graphics=simplified_graphics)
    pallet_xml = _pallet_xml(scenario, simplified_graphics=simplified_graphics)
    beam_xml = _beam_xml(scenario)
    infeed_xml = _infeed_xml(scenario, simplified_graphics=simplified_graphics)
    truck_xml = _truck_xml(scenario, simplified_graphics=simplified_graphics)
    truck_materials = _TRUCK_MATERIALS if scenario.simulation.truck is not None else ""
    industrial_scene = _industrial_scene_xml(simplified_graphics=simplified_graphics)
    meshes = (
        ""
        if simplified_graphics
        else "\n".join(f'<mesh name="{Path(filename).stem}" file="{filename}"/>' for filename in UR10E_MESH_FILES)
    )
    return f"""<mujoco model="UR10e stable palletizing cell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <size njmax="4000" nconmax="800"/>
  <default>
    <default class="ur10e">
      <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
      <!-- Collision class 2 lets the arm collide with the cell but not with a
           package that is rigidly attached to its own vacuum tool. -->
      <geom friction="0.8 0.01 0.001" contype="2" conaffinity="2"/>
      <general biastype="affine" ctrlrange="-6.2831 6.2831"
               gainprm="9000" biasprm="0 -9000 -700"/>
      <default class="size4">
        <joint damping="10"/><general forcerange="-330 330"/>
      </default>
      <default class="size3_limited">
        <joint damping="5" range="-3.1415 3.1415"/>
        <general forcerange="-150 150" ctrlrange="-3.1415 3.1415"/>
      </default>
      <default class="size2">
        <joint damping="2"/><general forcerange="-56 56"/>
      </default>
      <default class="robot_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
    </default>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" specular="0.2 0.2 0.2"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.16 0.18 0.21"
             rgb2="0.22 0.24 0.27" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.08"/>
    <material name="ur_blue" rgba="0.49 0.678 0.8 1"/>
    <material name="ur_light" rgba="0.82 0.82 0.82 1"/>
    <material name="ur_joint" rgba="0.08 0.09 0.10 1"/>
    <material name="tool" rgba="0.10 0.13 0.16 1"/>
    <material name="black" rgba="0.033 0.033 0.033 1" specular="0.5" shininess="0.25"/>
    <material name="jointgray" rgba="0.278 0.278 0.278 1" specular="0.5" shininess="0.25"/>
    <material name="linkgray" rgba="0.82 0.82 0.82 1" specular="0.5" shininess="0.25"/>
    <material name="urblue" rgba="0.49 0.678 0.8 1" specular="0.5" shininess="0.25"/>
    <material name="aluminium" rgba="0.62 0.66 0.69 1" metallic="0.35" roughness="0.28"/>
    <material name="vacuum_blue" rgba="0.04 0.34 0.58 1" metallic="0.15" roughness="0.36"/>
    <material name="onrobot_orange" rgba="0.94 0.35 0.07 1" roughness="0.42"/>
    <material name="pallet_wood" rgba="0.62 0.39 0.18 1" roughness="0.82"/>
    <material name="pallet_wood_dark" rgba="0.45 0.27 0.12 1" roughness="0.88"/>
    <material name="frame" rgba="0.16 0.19 0.22 1" metallic="0.55" roughness="0.30"/>
    <material name="roller" rgba="0.31 0.34 0.36 1" metallic="0.75" roughness="0.22"/>
    <material name="fence" rgba="0.12 0.14 0.15 1" metallic="0.45" roughness="0.45"/>
    <material name="safety_yellow" rgba="0.95 0.68 0.04 1" roughness="0.55"/>
    {truck_materials}
    {meshes}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <camera name="overview" pos="3.05 -3.15 2.45" xyaxes="0.72 0.69 0 -0.34 0.35 0.87"/>
    <!-- Looks in through the open rear doors, so the load is visible while it comes out. -->
    <camera name="dock" pos="3.10 -2.85 1.95" xyaxes="0.530 0.848 0 -0.322 0.201 0.925"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat"
          friction="1.0 0.01 0.001" conaffinity="3"/>
    <geom name="robot_pedestal" type="cylinder" pos="0.60 -0.72 0.24" size="0.24 0.24"
          rgba="0.22 0.25 0.28 1" conaffinity="3"/>
    {pallet_xml}
    {beam_xml}
    {infeed_xml}
    {truck_xml}
    {industrial_scene}
    {robot}
    {packages}
  </worldbody>
  <equality>
    {pallet_weld}
    {welds}
  </equality>
  <contact>
    <exclude body1="world" body2="pallet"/>
  </contact>
  <actuator>
    <general class="size4" name="shoulder_pan" joint="shoulder_pan_joint"/>
    <general class="size4" name="shoulder_lift" joint="shoulder_lift_joint"/>
    <general class="size3_limited" name="elbow" joint="elbow_joint"/>
    <general class="size2" name="wrist_1" joint="wrist_1_joint"/>
    <general class="size2" name="wrist_2" joint="wrist_2_joint"/>
    <general class="size2" name="wrist_3" joint="wrist_3_joint"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0 0 0 -1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"
          ctrl="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"/>
  </keyframe>
  <sensor>
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>
  </sensor>
</mujoco>"""


def _placement_differs(first: Placement, second: Placement) -> bool:
    """Whether measuring the balance changed where the package ends up."""
    return (first.x, first.y, first.z, first.yaw) != (second.x, second.y, second.z, second.yaw)


class UnreachablePose(RuntimeError):
    pass


def _hide_viewer_mass_overlays(mujoco: Any, option: Any) -> None:
    """Hide MuJoCo's inertia and center-of-mass debug markers."""
    option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_COM] = False


class PalletizingSimulator:
    cup_gap = 0.0015
    # How long the cell settles after a fast-forward jump. Long enough for the servos to
    # take up the gravity sag and for a carried carton to stop ringing on the cups.
    fast_forward_settle = 0.06
    # When the arm counts as parked at the end of a move: joint speed below this, held
    # for long enough that a zero crossing mid-wobble cannot pass for stillness.
    rest_speed = 0.008
    rest_steps = 25
    rest_timeout = 800
    # Servo droop the Cartesian correction cannot take out is a few millimetres; a pose
    # the arm cannot hold misses by hundreds. Anything between the two is still a move
    # that did not happen.
    reach_tolerance = 0.02
    ur10e_payload_kg = 12.5
    tool_mass_kg = 2.55
    euro_pallet_mass_kg = EURO_PALLET_MASS_KG
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    pallet_joint_names = ("pallet_x", "pallet_y", "pallet_z", "pallet_rx", "pallet_ry")

    def __init__(
        self,
        scenario: Scenario,
        *,
        viewer: bool = False,
        video_path: str | Path | None = None,
        seed: int = 7,
        measure_com: bool = True,
        precise_com: bool = False,
        simplified_graphics: bool = False,
        controls: ViewerControls | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo is required; run `uv sync --extra video --extra dev`") from exc

        self.mujoco = mujoco
        self.scenario = scenario
        self.simplified_graphics = simplified_graphics
        self.controls = controls if controls is not None else ViewerControls()
        self.mesh_assets = None if simplified_graphics else load_ur10e_mesh_assets()
        model = mujoco.MjModel.from_xml_string(
            build_mjcf(scenario, simplified_graphics=simplified_graphics), assets=self.mesh_assets
        )
        self.bind_model(model, mujoco.MjData(model))
        self.planner = StablePalletPlanner(scenario.planner)
        self.suction = SuctionArray()
        self.measure_com = measure_com
        self.precise_com = precise_com
        # What the cell believes about each package. With measurement on, the centre of
        # mass starts unknown -- assumed centred, which is what a cell without a sensor
        # has to do -- and each pick replaces the guess with what the wrist actually felt.
        self.known_packages: list[Package] = [
            replace(package, com=(0.0, 0.0, 0.0)) if measure_com else package
            for package in scenario.packages
        ]
        self.rng = np.random.default_rng(seed)
        # How the trailer turned up loaded. The same seed that draws the parcels decides
        # how they were stacked, so a run is reproducible end to end.
        self.truck = scenario.simulation.truck
        self.truck_slots: tuple[TruckSlot, ...] = (
            () if self.truck is None else plan_truck_load(scenario.packages, self.truck, seed=seed)
        )
        self.palletized: list[int] = []
        self._staged_centers: dict[int, np.ndarray] = {}
        self.load_drift = 0.0
        self.camera_name = "dock" if self.truck is not None else "overview"
        self.video_path = Path(video_path) if video_path else None
        self.frames: list[np.ndarray] = []
        self._next_frame_time = 0.0
        self._frame_interval = 1 / 30
        self._baseline: Any | None = None
        self.renderer: Any | None = None
        self.viewer: Any | None = None
        self.com_markers: ComMarkers | None = None
        self.playback: Playback | None = None
        # What the tool is carrying, what the overlay says and where the planner decided
        # each carton goes: the markers and the recorder read all three, and none of them
        # can be recovered from `MjData` alone.
        self.held_package: HeldPackage | None = None
        self.viewer_status: tuple[str, str] = ("", "")
        self.planned_placements: list[Placement] = []
        self._pacing_suspended = False
        self._viewer_syncs = 0
        if self.video_path:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self._viewer_thread: threading.Thread | None = None
        if viewer:
            import mujoco.viewer

            # `launch_passive` runs the window on a thread of its own and hands back no
            # way to wait for it. Spotting it here is what lets `close` join it; see the
            # comment there for why leaving it running is fatal.
            running = set(threading.enumerate())
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_thread = next(iter(set(threading.enumerate()) - running), None)
            with self.viewer.lock():
                _hide_viewer_mass_overlays(mujoco, self.viewer.opt)
            self.com_markers = ComMarkers(self)
            self.playback = Playback(self)

        self.data.qpos[self.arm_qpos] = self.home
        self.data.ctrl[:] = self.home
        mujoco.mj_forward(self.model, self.data)
        self.reference_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self._capture_frame(force=True)

    def bind_model(self, model: Any, data: Any) -> None:
        """Point the simulator at a model and cache its indices.

        Kept separate from `__init__` because grasping rebuilds the cell: the package
        stops being a free body and becomes part of the tool, which changes the model.
        """
        mujoco = self.mujoco
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.arm_qpos = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
        self.arm_dofs = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "suction_site")
        self.tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.pallet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pallet")
        self.pallet_weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "pallet_anchor")
        pallet_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.pallet_joint_names
        ]
        self.pallet_qpos = [model.jnt_qposadr[joint_id] for joint_id in pallet_joint_ids]
        self.pallet_dofs = [model.jnt_dofadr[joint_id] for joint_id in pallet_joint_ids]
        self.beam_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "beam")
        self.beam_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beam")
        self.beam_mocap_id = int(model.body_mocapid[self.beam_body_id])

    def package_world_poses(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """World pose of every package, whether it is free or carried by the tool."""
        poses = {}
        for index in range(len(self.scenario.packages)):
            body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
            poses[index] = (self.data.xpos[body].copy(), self.data.xquat[body].copy())
        return poses

    def _write_free_pose(self, index: int, position: np.ndarray, quaternion: np.ndarray) -> None:
        joint = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}")
        if joint < 0:
            return
        address = self.model.jnt_qposadr[joint]
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = quaternion
        self.data.qvel[self.model.jnt_dofadr[joint] : self.model.jnt_dofadr[joint] + 6] = 0.0

    def rebuild(self, held: HeldPackage | None) -> None:
        """Recompile the cell with a package carried rigidly, or with all of them free.

        Grasping and releasing change the kinematic tree, so the model object changes and
        every cached index with it. Arm configuration and package poses are carried across
        so the cell picks up exactly where it left off.
        """
        mujoco = self.mujoco
        arm = np.asarray(self.data.qpos[self.arm_qpos]).copy()
        control = np.asarray(self.data.ctrl).copy()
        pallet_q = np.asarray(self.data.qpos[self.pallet_qpos]).copy()
        pallet_locked = bool(self.data.eq_active[self.pallet_weld_id])
        poses = self.package_world_poses()
        appearance = {
            name: (
                int(self.model.geom_contype[geom]),
                int(self.model.geom_conaffinity[geom]),
                self.model.geom_rgba[geom].copy(),
            )
            for name, geom in ((name, self._geom_id(name)) for name in self._appearance_geom_names())
            if geom >= 0
        }

        model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.scenario, held, simplified_graphics=self.simplified_graphics),
            assets=self.mesh_assets,
        )
        self.bind_model(model, mujoco.MjData(model))
        self.data.qpos[self.arm_qpos] = arm
        self.data.ctrl[:] = control
        self.data.qpos[self.pallet_qpos] = pallet_q
        self.data.eq_active[self.pallet_weld_id] = pallet_locked
        for index, (position, quaternion) in poses.items():
            self._write_free_pose(index, position, quaternion)
        for name, (contype, conaffinity, rgba) in appearance.items():
            geom = self._geom_id(name)
            if geom < 0:
                continue
            self.model.geom_contype[geom] = contype
            self.model.geom_conaffinity[geom] = conaffinity
            self.model.geom_rgba[geom] = rgba
        mujoco.mj_forward(self.model, self.data)
        self.held_package = held
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self._reload_viewer()

    def _reload_viewer(self) -> None:
        """Keep the same GLFW window when the kinematic tree changes on grasp/release."""
        if self.viewer is None:
            return
        simulate = self.viewer._get_sim()
        if simulate is None:
            return
        simulate.load(self.model, self.data, "")
        with self.viewer.lock():
            _hide_viewer_mass_overlays(self.mujoco, self.viewer.opt)

    def set_viewer_status(self, left: str, right: str = "") -> None:
        self.viewer_status = (left, right)
        if self.viewer is None:
            return
        font = self.mujoco.mjtFontScale.mjFONTSCALE_150
        pos = self.mujoco.mjtGridPos.mjGRID_TOPLEFT
        self.viewer.set_texts((int(font), int(pos), left, right))

    @contextmanager
    def busy(self, label: str):
        """Announce a stretch of work that runs without stepping the cell.

        No frames are captured while the planner thinks, so the window holds the last
        one and the transport bar stops growing. Saying what is going on is the
        difference between a demonstrator that looks slow and one that looks broken.
        """
        previous_activity = self.controls.activity
        previous_status = self.viewer_status
        self.controls.activity = label
        self.set_viewer_status(label)
        try:
            yield
        finally:
            self.controls.activity = previous_activity
            self.set_viewer_status(*previous_status)

    @contextmanager
    def viewer_fast_forward(self):
        """Skip wall-clock pacing while the wrist probe settles through many steps."""
        previous = self._pacing_suspended
        self._pacing_suspended = True
        try:
            yield
        finally:
            self._pacing_suspended = previous
            self._capture_frame(force=True)

    def _appearance_geom_names(self) -> list[str]:
        """Geoms the cell recolours as it runs, so a rebuild has to carry their state over.

        Recompiling resets every geom to what the XML declares, which would hide the
        packages -- they are spawned transparent -- and forget which cups are pulling.
        """
        names = [f"package_geom_{index}" for index in range(len(self.scenario.packages))]
        if not self.simplified_graphics:
            names += [
                f"package_{part}_{index}"
                for index in range(len(self.scenario.packages))
                for part in ("tape", "label", "code")
            ]
        names += [f"cup_{column}_{row}" for column, row, _x, _y in SuctionArray().cup_offsets()]
        return names

    def _geom_id(self, name: str) -> int:
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name)

    def _package_geom(self, index: int) -> int:
        return self._geom_id(f"package_geom_{index}")

    def close(self) -> None:
        if self.video_path and self.frames:
            try:
                import imageio.v3 as iio
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Video export requires `uv sync --extra video`") from exc
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(self.video_path, self.frames, fps=30, codec="libx264", quality=8)
        if self.renderer is not None:
            self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
            # `Handle.close` only asks the render loop to stop; it returns while the
            # window is still swapping buffers. The interpreter then reaches the
            # `glfw.terminate` that `launch_passive` registered with atexit, tears the
            # EGL display out from under that thread and the process dies with a
            # segfault -- after the run is finished, so the artifact is on disk and the
            # summary is lost in a buffer. Waiting for the thread costs milliseconds.
            if self._viewer_thread is not None:
                self._viewer_thread.join(timeout=5.0)
                self._viewer_thread = None

    def sync_viewer(self) -> None:
        """Push the current state to the viewer window, markers included."""
        if self.viewer is None:
            return
        if self.com_markers is not None:
            self.com_markers.draw()
        self.viewer.sync()

    @property
    def frame_pause(self) -> float:
        """Wall-clock seconds to wait per step: a simulated second takes
        1 / (speed * SPEED_SCALE).

        Zero while the probe is settling or when the operator asked for no pacing: the
        cell then runs as fast as the machine allows, still executing every trajectory.
        """
        if self._pacing_suspended or not self.controls.paced:
            return 0.0
        return self.model.opt.timestep / (self.controls.speed * SPEED_SCALE)

    def _capture_frame(self, force: bool = False) -> None:
        if self.playback is not None:
            self.playback.gate()
        if self.controls.cancelled:
            raise RunCancelled("The operator stopped the run")
        if self.viewer is not None:
            self._viewer_syncs += 1
            pause = self.frame_pause
            interval = 1 if pause > 0.0 else 20
            if force or self._viewer_syncs % interval == 0:
                self.sync_viewer()
            if pause > 0.0:
                time.sleep(pause)
        if self.playback is not None:
            self.playback.record()
        if self.renderer is not None and (force or self.data.time >= self._next_frame_time):
            self.renderer.update_scene(self.data, camera=self.camera_name)
            self.frames.append(self.renderer.render().copy())
            self._next_frame_time = self.data.time + self._frame_interval

    def _step(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self.mujoco.mj_step(self.model, self.data)
            self._capture_frame()

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])

    @staticmethod
    def _rotation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        error_matrix = desired @ current.T
        cosine = max(-1.0, min(1.0, (np.trace(error_matrix) - 1) / 2))
        angle = math.acos(cosine)
        skew = np.array(
            [
                error_matrix[2, 1] - error_matrix[1, 2],
                error_matrix[0, 2] - error_matrix[2, 0],
                error_matrix[1, 0] - error_matrix[0, 1],
            ]
        )
        if angle < 1e-7:
            return skew / 2
        return skew * (angle / (2 * math.sin(angle)))

    def _grasp_offset_world(self, offset: tuple[float, float], yaw_deg: float) -> tuple[float, float]:
        """World XY of the engaged cup block, measured from the tool origin.

        The cups are laid out in the tool frame, which points down: `reference_rotation`
        is diag(1, -1, -1), so its y axis runs against the world's. Cancelling the offset
        straight in world coordinates doubles it on that axis instead of removing it, and
        the outer row of cups then hangs past the edge of the carton.
        """
        local = self.reference_rotation[:2, :2] @ np.asarray(offset, dtype=float)
        angle = math.radians(yaw_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        return (
            float(local[0] * cosine - local[1] * sine),
            float(local[0] * sine + local[1] * cosine),
        )

    def _solve_ik(self, xyz: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
        target = np.asarray(xyz, dtype=float)
        desired_rotation = self._rotation_z(math.radians(yaw_deg)) @ self.reference_rotation
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        current = np.asarray(self.data.qpos[self.arm_qpos]).copy()
        seeds = (current, self.home.copy())
        best_error = float("inf")
        for seed in seeds:
            q = seed.copy()
            for _ in range(300):
                self.ik_data.qpos[self.arm_qpos] = q
                self.mujoco.mj_forward(self.model, self.ik_data)
                position_error = target - self.ik_data.site_xpos[self.site_id]
                rotation_error = self._rotation_error(
                    desired_rotation, self.ik_data.site_xmat[self.site_id].reshape(3, 3)
                )
                best_error = min(best_error, float(np.linalg.norm(position_error)))
                if (
                    np.linalg.norm(position_error) < 0.0015
                    and np.linalg.norm(rotation_error) < 0.015
                ):
                    return q
                self.mujoco.mj_jacSite(
                    self.model,
                    self.ik_data,
                    jacobian_position,
                    jacobian_rotation,
                    self.site_id,
                )
                jacobian = np.vstack(
                    (jacobian_position[:, self.arm_dofs], jacobian_rotation[:, self.arm_dofs])
                )
                error = np.concatenate((position_error, rotation_error))
                damping = 0.025
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(6), error
                )
                q += np.clip(delta, -0.12, 0.12)
                q[2] = np.clip(q[2], -3.13, 3.13)
                q = (q + math.pi) % (2 * math.pi) - math.pi

        raise UnreachablePose(
            f"UR10e cannot reach {tuple(round(value, 3) for value in xyz)} "
            f"at yaw {yaw_deg:.0f}° (best residual {best_error * 1000:.1f} mm)"
        )

    def _move_joints(
        self,
        target_q: np.ndarray,
        seconds: float = 0.85,
        *,
        rest_speed: float | None = None,
        rest_steps: int | None = None,
    ) -> None:
        """Smoothstep the arm in joint space, then wait until it is still.

        A move that only has to carry on afterwards is done once the arm has stopped.
        Weighing a carton is not: the fitted moment is read against the pose the wrist
        ends on, so the probe asks for a slower threshold and a longer dwell, and pays
        for them. The two arguments are how still, and for how long, the caller needs.
        """
        start_q = np.asarray(self.data.qpos[self.arm_qpos], dtype=float).copy()
        target_q = np.asarray(target_q, dtype=float)
        steps = max(1, round(seconds / self.model.opt.timestep))
        for step in range(steps):
            phase = (step + 1) / steps
            smooth = phase * phase * (3 - 2 * phase)
            self.data.ctrl[:] = start_q + (target_q - start_q) * smooth
            self.mujoco.mj_step(self.model, self.data)
            self._capture_frame()
        self.data.ctrl[:] = target_q
        # Waiting for the joints to reach the commanded pose waits forever: a position
        # servo holds against gravity with a standing error of several milliradians, and
        # a carton on the cups pushes it past fifteen. That droop is exactly what the
        # Cartesian correction in `_move_tool` measures and takes out, so this loop only
        # has to answer the other question -- has the arm stopped moving.
        limit = self.rest_speed if rest_speed is None else rest_speed
        dwell = self.rest_steps if rest_steps is None else rest_steps
        still = 0
        for _ in range(self.rest_timeout):
            if np.max(np.abs(self.data.qvel[self.arm_dofs])) < limit:
                still += 1
                if still >= dwell:
                    break
            else:
                still = 0
            self.mujoco.mj_step(self.model, self.data)
            self._capture_frame()

    def _move_tool(
        self, x: float, y: float, z: float, yaw_deg: float, seconds: float = 0.85
    ) -> None:
        desired_position = np.array([x, y, z], dtype=float)
        commanded_position = desired_position.copy()
        for correction in range(3):
            target_q = self._solve_ik(tuple(commanded_position), yaw_deg)
            if self.controls.fast_forward:
                self._snap_to(target_q)
            else:
                self._move_joints(target_q, seconds if correction == 0 else 0.28)
            cartesian_error = desired_position - self.data.site_xpos[self.site_id]
            if np.linalg.norm(cartesian_error) < 0.0025:
                return
            # Compensate the static deflection caused by gravity and payload.
            commanded_position += cartesian_error
        # A pose the solver accepts is not a pose the arm can hold: folded in tight the
        # links stand on the pedestal and the servos stop short. Sealing the cups on a
        # carton the tool never arrived over is how a cell picks one up by its corner
        # and puts it down somewhere else, so say so instead.
        residual = float(np.linalg.norm(desired_position - self.data.site_xpos[self.site_id]))
        if residual > self.reach_tolerance:
            raise UnreachablePose(
                f"UR10e stopped {residual * 1000:.0f} mm short of "
                f"{tuple(round(value, 3) for value in desired_position)} at yaw {yaw_deg:.0f}°"
            )

    def _snap_to(self, target_q: np.ndarray) -> None:
        """Put the arm on a solved pose without travelling there.

        Fast-forward skips the transit, not the physics that decides the outcome: the
        cell still settles on arrival, so the servo sag, the contacts and the release all
        happen exactly as they would after a real move. A package held by the vacuum is
        carried across too -- it is a free body attached by a weld, and leaving it behind
        would snap it back across the cell.
        """
        self.data.qpos[self.arm_qpos] = target_q
        self.data.qvel[self.arm_dofs] = 0.0
        self.data.ctrl[:] = target_q
        self.mujoco.mj_forward(self.model, self.data)
        self._carry_welded_package()
        self._step(self.fast_forward_settle)

    def _welded_package(self) -> int | None:
        """Index of the package currently held by an active suction weld, if any."""
        for index in range(len(self.scenario.packages)):
            equality = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{index}"
            )
            if equality >= 0 and self.data.eq_active[equality]:
                return index
        return None

    def _carry_welded_package(self) -> None:
        index = self._welded_package()
        if index is None:
            return
        equality = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{index}"
        )
        tool_position = self.data.xpos[self.tool_body_id]
        tool_rotation = self.data.xmat[self.tool_body_id].reshape(3, 3)
        tool_quaternion = self.data.xquat[self.tool_body_id]
        relative_position = self.model.eq_data[equality, 3:6]
        relative_quaternion = self.model.eq_data[equality, 6:10]
        world_quaternion = np.empty(4)
        self.mujoco.mju_mulQuat(world_quaternion, tool_quaternion, relative_quaternion)
        self._write_free_pose(
            index, tool_position + tool_rotation @ relative_position, world_quaternion
        )
        self.mujoco.mj_forward(self.model, self.data)

    def _package_top_center(self, index: int) -> np.ndarray:
        package = self.known_packages[index]
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}"
        )
        address = self.model.jnt_qposadr[joint_id]
        center = np.asarray(self.data.qpos[address : address + 3]).copy()
        center[2] += package.size[2] / 2
        return center

    def _show_active_cups(self, active_cups: tuple[tuple[float, float], ...]) -> None:
        active = {(round(x, 4), round(y, 4)) for x, y in active_cups}
        for column, row, x, y in self.suction.cup_offsets():
            geom_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"cup_{column}_{row}"
            )
            if (round(x, 4), round(y, 4)) in active:
                self.model.geom_rgba[geom_id] = [0.10, 0.85, 0.90, 1.0]
            else:
                self.model.geom_rgba[geom_id] = [0.18, 0.22, 0.24, 0.22]

    def _set_package_pose(
        self, index: int, xyz: tuple[float, float, float], yaw: float = 0.0
    ) -> None:
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}"
        )
        qpos_address = self.model.jnt_qposadr[joint_id]
        qvel_address = self.model.jnt_dofadr[joint_id]
        half = math.radians(yaw) / 2
        self.data.qpos[qpos_address : qpos_address + 7] = [
            *xyz,
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ]
        self.data.qvel[qvel_address : qvel_address + 6] = 0
        geom_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"package_geom_{index}"
        )
        self.model.geom_contype[geom_id] = 1
        self.model.geom_conaffinity[geom_id] = 1
        main_color = _rgba(index) if self.simplified_graphics else _cardboard_rgba(index)
        self.model.geom_rgba[geom_id] = np.fromstring(main_color, sep=" ")
        if not self.simplified_graphics:
            visual_colors = {
                f"package_tape_{index}": (0.73, 0.61, 0.40, 1.0),
                f"package_label_{index}": (0.94, 0.95, 0.91, 1.0),
                f"package_code_{index}": tuple(np.fromstring(_rgba(index), sep=" ")),
            }
            for name, color in visual_colors.items():
                visual_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name
                )
                self.model.geom_rgba[visual_id] = color
        self.mujoco.mj_forward(self.model, self.data)

    def _set_suction(self, index: int, active: bool) -> None:
        equality_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{index}"
        )
        if active:
            package_body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"package_{index}"
            )
            tool_rotation = self.data.xmat[self.tool_body_id].reshape(3, 3)
            relative_position = tool_rotation.T @ (
                self.data.xpos[package_body_id] - self.data.xpos[self.tool_body_id]
            )
            inverse_tool = np.empty(4)
            relative_quaternion = np.empty(4)
            self.mujoco.mju_negQuat(inverse_tool, self.data.xquat[self.tool_body_id])
            self.mujoco.mju_mulQuat(
                relative_quaternion, inverse_tool, self.data.xquat[package_body_id]
            )
            self.model.eq_data[equality_id, 3:6] = relative_position
            self.model.eq_data[equality_id, 6:10] = relative_quaternion
        self.data.eq_active[equality_id] = active
        self.mujoco.mj_forward(self.model, self.data)

    def _read_pose(self, index: int, planned_yaw: int) -> tuple[Placement, float]:
        package = self.known_packages[index]
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}"
        )
        address = self.model.jnt_qposadr[joint_id]
        x, y, center_z, qw, qx, qy, qz = self.data.qpos[address : address + 7]
        yaw = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
        snapped_yaw = 90 if abs(abs(yaw) - 90) < min(abs(yaw), abs(abs(yaw) - 180)) else 0
        width, depth, height = package.oriented_size(snapped_yaw)
        origin_x, origin_y = self.scenario.simulation.pallet_origin
        bottom = center_z - height / 2 - self.scenario.simulation.pallet_height
        placement = Placement(
            package,
            round(float(x - width / 2 - origin_x), 4),
            round(float(y - depth / 2 - origin_y), 4),
            round(max(0.0, float(bottom)), 4),
            snapped_yaw,
        )
        return placement, tilt

    def _verified_state(self, order: list[int], planned: list[Placement]) -> tuple[StackState, list[float]]:
        """Re-read the stack from the scene, in the order the cartons were actually laid.

        Out of a trailer the cell decides that order as it goes, so it is no longer the
        order the packages were declared in.
        """
        placements: list[Placement] = []
        tilts: list[float] = []
        for position, index in enumerate(order):
            measured, tilt = self._read_pose(index, planned[position].yaw)
            placements.append(measured)
            tilts.append(tilt)
        return StackState(self.scenario.pallet, placements), tilts

    def _park_package(self, index: int) -> None:
        height = self.scenario.packages[index].size[2]
        self._set_package_pose(index, (-3.0 - index * 0.7, 0.0, height / 2 + 0.002))

    # -- infeed: conveyor or loaded trailer ----------------------------------------------

    def stage_truck_load(self) -> None:
        """Put the whole delivery in the trailer and let it settle onto its own contacts.

        Everything the cell has to move is in the scene from the first frame, which is the
        point: there is no belt handing over one carton at a time, so the cell has to work
        out for itself which one it is allowed to take next.
        """
        if self.truck is None:
            return
        for slot in self.truck_slots:
            self._set_package_pose(slot.index, slot.center, slot.yaw)
        self._step(0.30)
        self._staged_centers = {
            slot.index: np.asarray(self._package_top_center(slot.index), dtype=float).copy()
            for slot in self.truck_slots
        }
        self.set_viewer_status(
            f"Trailer loaded: {len(self.truck_slots)} cartons",
            f"{len({slot.column for slot in self.truck_slots})} columns",
        )

    def _package_yaw(self, index: int) -> float:
        """Yaw of a carton as it actually lies, in degrees.

        Stands in for what an RGB-D pose estimate would return. A hand-stacked column is
        never square to the trailer, and the tool has to line up with the carton it is
        about to seal onto, not with the bay.
        """
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}"
        )
        address = self.model.jnt_qposadr[joint_id]
        qw, qx, qy, qz = self.data.qpos[address + 3 : address + 7]
        return math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def next_pick(self, remaining: list[int]) -> int:
        """Which carton the cell may take next.

        Out of a trailer that is whichever one is highest: a carton resting on another
        always has the higher top of the two, so the highest top left in the bay is
        always one with nothing standing on it. Off a conveyor there is no choice --
        whatever the belt presents is next.
        """
        if self.truck is None:
            return remaining[0]
        return max(remaining, key=lambda index: (float(self._package_top_center(index)[2]), -index))

    def unload_report(self, order: list[int]) -> dict[str, Any] | None:
        """What the trailer held and the order the cell actually emptied it in."""
        if self.truck is None:
            return None
        report = load_summary(self.truck_slots, self.truck)
        report["pick_order"] = [self.scenario.packages[index].id for index in order]
        report["followed_plan"] = report["pick_order"] == report["planned_pick_order"]
        report["max_load_disturbance_mm"] = self.load_drift * 1_000
        return report

    def record_load_drift(self, remaining: list[int]) -> float:
        """How far the cartons still in the trailer have moved since it was loaded.

        Taking the top carton out is only correct if the rest of the load stays where it
        was. This is the number that says so, measured rather than assumed, and it is the
        one that grows first when a pick starts dragging its neighbours.
        """
        drift = max(
            (
                float(np.linalg.norm(self._package_top_center(index) - self._staged_centers[index]))
                for index in remaining
                if index in self._staged_centers
            ),
            default=0.0,
        )
        self.load_drift = max(self.load_drift, drift)
        return drift

    def _unload_sequence(self) -> list[int]:
        """The whole pick order up front, for the runs that skip the arm."""
        if self.truck is None:
            return list(range(len(self.scenario.packages)))
        return [slot.index for slot in unload_order(self.truck_slots)]

    def _pick_forecast(self, remaining: list[int]) -> list[int]:
        """The order the rest of the load will come out in, as things stand.

        A conveyor only ever shows the next carton, so this is where seeing the whole
        trailer pays off twice: the same look that says which carton may be taken now
        also tells the planner what is behind it, which is exactly what its lookahead
        needs. Cartons lower down can be re-ordered by what lands on the pallet, so this
        is recomputed on every cycle.
        """
        if self.truck is None:
            return list(remaining)
        return sorted(
            remaining, key=lambda index: (-float(self._package_top_center(index)[2]), index)
        )

    def _pick_pose(self, index: int, remaining: list[int]) -> PickPose:
        """Measure where the next carton is and what it must be lifted clear of."""
        sim = self.scenario.simulation
        if self.truck is None:
            infeed_x, infeed_y = sim.infeed_position
            package = self.known_packages[index]
            self._set_package_pose(index, (infeed_x, infeed_y, sim.infeed_height + package.size[2] / 2 + 0.001))
            self._step(0.20)
            obstacle_top = sim.infeed_height
            yaw = 0.0
        else:
            others = [other for other in remaining if other != index]
            obstacle_top = max(
                (float(self._package_top_center(other)[2]) for other in others),
                default=self.truck.floor_height,
            )
            yaw = self._package_yaw(index)
        top = self._package_top_center(index)
        if self.truck is not None:
            # Everything the cell picks is measured, so a carton that is no longer in the
            # bay is a carton it cannot go and get. Saying so beats letting the arm chase
            # it and fail on an unreachable pose several metres away.
            min_x, max_x, min_y, max_y = self.truck.bounds
            if not (min_x - 0.2 <= top[0] <= max_x + 0.2 and min_y - 0.2 <= top[1] <= max_y + 0.2):
                raise RuntimeError(
                    f"{self.scenario.packages[index].id} is not in the trailer bay: measured at "
                    f"({top[0]:.2f}, {top[1]:.2f}, {top[2]:.2f}) m, outside "
                    f"x {min_x:.2f}..{max_x:.2f}, y {min_y:.2f}..{max_y:.2f}. "
                    "Resetting the scene in the viewer empties the trailer like this."
                )
        return PickPose(index, float(top[0]), float(top[1]), float(top[2]), yaw, obstacle_top)

    def _set_placement(self, index: int, placement: Placement, *, lift: float = 0.0) -> None:
        sim = self.scenario.simulation
        center_x, center_y, center_z = placement.center
        self._set_package_pose(
            index,
            (
                sim.pallet_origin[0] + center_x,
                sim.pallet_origin[1] + center_y,
                sim.pallet_height + center_z + lift,
            ),
            yaw=placement.yaw,
        )

    def apply_stack(self, placements: list[Placement], *, lift: float = 0.0) -> list[int]:
        """Park every package, then teleport the given stack onto the pallet."""
        id_to_index = {package.id: index for index, package in enumerate(self.scenario.packages)}
        for index in range(len(self.scenario.packages)):
            self._park_package(index)
        placed: list[int] = []
        for placement in placements:
            index = id_to_index[placement.package.id]
            self._set_placement(index, placement, lift=lift)
            placed.append(index)
        self.planned_placements = list(placements)
        self.palletized = list(placed)
        self.mujoco.mj_forward(self.model, self.data)
        return placed

    def measure_true_pallet_com(self, indices: list[int]) -> PalletCom:
        """Physical CoM of the chosen package bodies, in pallet coordinates."""
        masses = []
        world_coms = []
        for index in indices:
            body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
            masses.append(float(self.model.body_mass[body]))
            world_coms.append(self.data.xipos[body].copy())
        sim = self.scenario.simulation
        return com_from_world_points(
            masses, world_coms, sim.pallet_origin, sim.pallet_height, self.scenario.pallet
        )

    def _placed_package_indices(self) -> list[int]:
        """Packages standing on the pallet, in the order the cell put them there.

        With a trailer in the cell the whole delivery is visible from the first frame,
        so being visible no longer means being on the pallet: a transport trial has to
        shake what was stacked, not what is still in the bay.
        """
        if self.palletized:
            return list(self.palletized)
        return [
            index
            for index in range(len(self.scenario.packages))
            if self.model.geom_rgba[self._package_geom(index)][3] > 0.5
        ]

    def _package_relative_to_pallet(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"package_{index}")
        pallet_pos = self.data.xpos[self.pallet_body_id]
        pallet_mat = self.data.xmat[self.pallet_body_id].reshape(3, 3)
        relative = pallet_mat.T @ (self.data.xpos[body] - pallet_pos)
        inverse = np.empty(4)
        relative_quaternion = np.empty(4)
        self.mujoco.mju_negQuat(inverse, self.data.xquat[self.pallet_body_id])
        self.mujoco.mju_mulQuat(relative_quaternion, inverse, self.data.xquat[body])
        return relative, relative_quaternion

    def _tilt_deg(self, quaternion: np.ndarray) -> float:
        _qw, qx, qy, _qz = quaternion
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))

    def _go_home(self, seconds: float = 1.0) -> None:
        """Lift off the stack, retreat over the infeed, then fold the arm."""
        sim = self.scenario.simulation
        retreat_x, retreat_y = sim.truck.center if sim.truck is not None else sim.infeed_position
        site = np.asarray(self.data.site_xpos[self.site_id], dtype=float)
        clear_z = max(1.08, float(site[2]) + 0.12)
        self._move_tool(float(site[0]), float(site[1]), clear_z, 0, 0.45)
        self._move_tool(retreat_x, retreat_y, clear_z, 0, 0.80)
        if self.controls.fast_forward:
            self._snap_to(self.home)
            return
        self._move_joints(self.home, seconds)
        self._step(0.20)

    def capture_baseline(self) -> None:
        """Freeze the loaded pallet exactly as placement left it."""
        snapshot = self.mujoco.MjData(self.model)
        self.mujoco.mj_copyData(snapshot, self.model, self.data)
        self._baseline = snapshot

    def restore_baseline(self) -> None:
        """Rewind pallet and packages to the saved post-placement state."""
        if self._baseline is None:
            raise RuntimeError("No baseline stack has been captured")
        clock = float(self.data.time)
        self.mujoco.mj_copyData(self.data, self.model, self._baseline)
        self.data.time = clock
        self.data.xfrc_applied[:] = 0
        self._configure_pallet_joints(beam=False)
        self._park_beam()
        self._capture_frame(force=True)

    def _hover_force(self, placed: list[int]) -> np.ndarray:
        """Force that holds the loaded pallet up once the world weld is released."""
        mass = self.euro_pallet_mass_kg + sum(self.known_packages[index].mass for index in placed)
        return np.array([0.0, 0.0, mass * GRAVITY])

    def _step_with_pallet_force(self, seconds: float, force: np.ndarray) -> None:
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self.data.xfrc_applied[self.pallet_body_id, :3] = force
            self.mujoco.mj_step(self.model, self.data)
            self._capture_frame()

    def _package_outcomes(
        self,
        placed: list[int],
        before: dict[int, tuple[np.ndarray, np.ndarray]],
        config: ShakeConfig,
    ) -> tuple[list[dict[str, Any]], float, bool]:
        outcomes: list[dict[str, Any]] = []
        max_shift = 0.0
        any_fell = False
        for index in placed:
            relative, quaternion = self._package_relative_to_pallet(index)
            previous, _ = before[index]
            shift = float(np.linalg.norm(relative - previous))
            fell = package_fell_off(
                relative,
                self.scenario.pallet.width,
                self.scenario.pallet.depth,
                self.scenario.simulation.pallet_height,
            )
            max_shift = max(max_shift, shift)
            any_fell = any_fell or fell
            outcomes.append(
                {
                    "package_id": self.scenario.packages[index].id,
                    "displacement_mm": shift * 1_000,
                    "horizontal_shift_mm": float(np.linalg.norm((relative - previous)[:2])) * 1_000,
                    "vertical_drop_mm": float(previous[2] - relative[2]) * 1_000,
                    "tilt_deg": self._tilt_deg(quaternion),
                    "fell_off": fell,
                    "baseline_relative_center_m": [float(value) for value in previous],
                    "relative_center_m": [float(value) for value in relative],
                }
            )
        return outcomes, max_shift, (not any_fell) and max_shift <= config.max_shift_m

    def _apply_jolt(self, axis: str, peak_accel_g: float, config: ShakeConfig) -> dict[str, Any]:
        """One directional pulse from the current (baseline) stack."""
        placed = self._placed_package_indices()
        hover = self._hover_force(placed)
        direction = axis_unit(axis)
        total_mass = float(hover[2] / GRAVITY)
        peak_force = total_mass * peak_accel_g * GRAVITY
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]

        self.data.eq_active[self.pallet_weld_id] = 0
        self._configure_pallet_joints(beam=False)
        self.data.xfrc_applied[self.pallet_body_id, :3] = hover
        self.mujoco.mj_forward(self.model, self.data)
        self._step_with_pallet_force(0.08, hover)

        before = {index: self._package_relative_to_pallet(index) for index in placed}
        pallet_before = self.data.xpos[self.pallet_body_id].copy()

        duration = config.duration
        steps = max(1, round(duration / self.model.opt.timestep))
        start_time = float(self.data.time)
        measured_peak_accel = 0.0
        previous_velocity = self.data.qvel[self.pallet_dofs].copy()
        for _ in range(steps):
            elapsed = float(self.data.time) - start_time
            force = hover + sine_jolt_force(elapsed, duration, peak_force, direction)
            self.data.xfrc_applied[self.pallet_body_id, :3] = force
            self.mujoco.mj_step(self.model, self.data)
            self._capture_frame()
            velocity = self.data.qvel[self.pallet_dofs].copy()
            accel = np.linalg.norm((velocity - previous_velocity) / self.model.opt.timestep)
            measured_peak_accel = max(measured_peak_accel, float(accel))
            previous_velocity = velocity

        self._step_with_pallet_force(config.settle_seconds, hover)
        self.data.xfrc_applied[self.pallet_body_id] = 0

        delta = self.data.xpos[self.pallet_body_id] - pallet_before
        outcomes, max_shift, held = self._package_outcomes(placed, before, config)
        return {
            "axis": axis,
            "peak_accel_g": peak_accel_g,
            "reason": transport_reason(peak_accel_g),
            "peak_force_n": peak_force,
            "measured_peak_accel_g": measured_peak_accel / GRAVITY,
            "duration_s": duration,
            "pallet_mass_kg": self.euro_pallet_mass_kg,
            "loaded_mass_kg": total_mass,
            "pallet_travel_mm": float(np.linalg.norm(delta)) * 1_000,
            "axis_travel_mm": abs(float(delta[axis_index])) * 1_000,
            "max_package_shift_mm": max_shift * 1_000,
            "held": held,
            "packages": outcomes,
        }

    def run_transport_trials(
        self, config: ShakeConfig | None = None, *, retract_arm: bool = True
    ) -> dict[str, Any]:
        """Sweep transport jolts, restoring the saved stack before every trial."""
        config = config or self.scenario.simulation.shake
        if retract_arm:
            self._go_home()
        self._step(config.hold_seconds)
        self.capture_baseline()

        trials: list[dict[str, Any]] = []
        for level_index, peak_g in enumerate(config.levels_g, start=1):
            for axis in config.axes:
                self.restore_baseline()
                self._step(config.rest_seconds)
                trial = self._apply_jolt(axis, peak_g, config)
                trial["level"] = level_index
                trials.append(trial)
        self.restore_baseline()
        attach_trial_scores(trials, config.max_shift_m)
        return {
            "levels_g": list(config.levels_g),
            "axes": list(config.axes),
            "duration_s": config.duration,
            "max_shift_mm": config.max_shift_m * 1_000,
            "trials": trials,
            "summary": summarise_trials(trials, config.axes),
        }

    def _joint_id(self, name: str) -> int:
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)

    def _set_joint_locked(self, name: str, locked: bool) -> None:
        joint = self._joint_id(name)
        if locked:
            self.model.jnt_limited[joint] = 1
            self.model.jnt_range[joint] = (-1.0e-4, 1.0e-4)
        else:
            self.model.jnt_limited[joint] = 0
            self.model.jnt_range[joint] = (-1.6, 1.6)

    def _configure_pallet_joints(self, *, beam: bool = False) -> None:
        """Lock pallet DOFs that the current trial must not use.

        Transport jolts leave the three slides free and freeze both hinges so the deck
        stays level. A beam trial frees the hinges: the loaded pallet has to balance on
        the fulcrum, and tips if the centre of mass walks off the ridge.
        """
        self._set_joint_locked("pallet_x", False)
        self._set_joint_locked("pallet_y", False)
        self._set_joint_locked("pallet_z", False)
        self._set_joint_locked("pallet_rx", not beam)
        self._set_joint_locked("pallet_ry", not beam)

    def _park_beam(self) -> None:
        self.data.mocap_pos[self.beam_mocap_id] = (0.0, 0.0, -1.0)
        self.data.mocap_quat[self.beam_mocap_id] = (1.0, 0.0, 0.0, 0.0)
        self.model.geom_rgba[self.beam_geom_id, 3] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _stage_beam(self, axis: str, config: ShakeConfig) -> None:
        center = self.data.xpos[self.pallet_body_id]
        height = config.beam_height
        if axis == "x":
            quat = (1.0, 0.0, 0.0, 0.0)
        elif axis == "y":
            quat = (0.70710678, 0.0, 0.0, 0.70710678)
        else:
            raise ValueError(f"Beam axis must be x or y, not {axis!r}")
        self.data.mocap_pos[self.beam_mocap_id] = (float(center[0]), float(center[1]), height / 2)
        self.data.mocap_quat[self.beam_mocap_id] = quat
        self.model.geom_rgba[self.beam_geom_id] = (0.78, 0.28, 0.14, 1.0)
        self.mujoco.mj_forward(self.model, self.data)

    def _lift_loaded_pallet(self, lift: float) -> None:
        self.data.qpos[self.pallet_qpos[2]] += lift
        self.data.qvel[self.pallet_dofs] = 0
        for index in self._placed_package_indices():
            joint = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, f"package_joint_{index}"
            )
            if joint < 0:
                continue
            address = self.model.jnt_qposadr[joint]
            dof = self.model.jnt_dofadr[joint]
            self.data.qpos[address + 2] += lift
            self.data.qvel[dof : dof + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _apply_beam(self, axis: str, config: ShakeConfig) -> dict[str, Any]:
        """Seat the saved stack on a narrow beam along `axis` and see if it stays up."""
        placed = self._placed_package_indices()
        self.data.eq_active[self.pallet_weld_id] = 0
        self._lift_loaded_pallet(config.beam_height + 0.003)
        self._stage_beam(axis, config)
        self._configure_pallet_joints(beam=True)
        self.data.xfrc_applied[self.pallet_body_id] = 0
        self.mujoco.mj_forward(self.model, self.data)

        before = {index: self._package_relative_to_pallet(index) for index in placed}
        self._step(config.beam_settle_seconds)

        xmat = self.data.xmat[self.pallet_body_id]
        tilt_x = pallet_tilt_deg(xmat, "x")
        tilt_y = pallet_tilt_deg(xmat, "y")
        tilt = tilt_x if axis == "x" else tilt_y
        pallet_z = float(self.data.xpos[self.pallet_body_id][2])
        outcomes, max_shift, _packages_held = self._package_outcomes(placed, before, config)
        any_fell = any(item["fell_off"] for item in outcomes)
        seated = pallet_z > config.beam_height * 0.35
        held = (
            seated
            and not any_fell
            and abs(tilt_x) <= config.max_tilt_deg
            and abs(tilt_y) <= config.max_tilt_deg
        )
        self._park_beam()
        self._configure_pallet_joints(beam=False)
        return {
            "kind": "beam",
            "axis": axis,
            "reason": f"narrow beam along {axis}",
            "beam_width_mm": config.beam_width * 1_000,
            "beam_height_mm": config.beam_height * 1_000,
            "tilt_deg": tilt,
            "tilt_x_deg": tilt_x,
            "tilt_y_deg": tilt_y,
            "max_tilt_deg": config.max_tilt_deg,
            "pallet_z_m": pallet_z,
            "seated": seated,
            "max_package_shift_mm": max_shift * 1_000,
            "held": held,
            "packages": outcomes,
        }

    def run_beam_trials(self, config: ShakeConfig | None = None, *, retract_arm: bool = False) -> dict[str, Any]:
        """Rest the loaded pallet on a narrow beam, first along X and then along Y."""
        config = config or self.scenario.simulation.shake
        if self._baseline is None:
            if retract_arm:
                self._go_home()
            self._step(config.hold_seconds)
            self.capture_baseline()

        trials: list[dict[str, Any]] = []
        for axis in ("x", "y"):
            self.restore_baseline()
            self._step(config.rest_seconds)
            trials.append(self._apply_beam(axis, config))
        self.restore_baseline()
        return {
            "beam_width_mm": config.beam_width * 1_000,
            "beam_height_mm": config.beam_height * 1_000,
            "max_tilt_deg": config.max_tilt_deg,
            "trials": trials,
            "summary": summarise_beam(trials),
        }

    def place_from_plan(self) -> dict[str, Any]:
        """Teleport every package to its planned pose so the shake can be watched without the arm."""
        sim = self.scenario.simulation
        state = StackState(self.scenario.pallet)
        planned = self.planned_placements
        planned.clear()
        self.stage_truck_load()
        order = self._unload_sequence()
        for position, index in enumerate(order):
            incoming = [self.known_packages[item] for item in order[position:]]
            with self.busy(f"Planificando la caja {position + 1}/{len(order)}…"):
                placement = self.planner.plan_next(state, incoming).placement
            world_x = sim.pallet_origin[0] + placement.x + placement.width / 2
            world_y = sim.pallet_origin[1] + placement.y + placement.depth / 2
            world_z = sim.pallet_height + placement.z + placement.height / 2 + 0.001
            self._set_package_pose(index, (world_x, world_y, world_z), yaw=placement.yaw)
            planned.append(placement)
            state = state.with_placement(placement)
            self._capture_frame(force=True)
        self.palletized = list(order)
        self._step(sim.settle_seconds)
        state, tilts = self._verified_state(order, planned)
        final_report = validate_stack(
            state,
            self.scenario.planner.minimum_support_ratio,
            self.scenario.planner.minimum_tipping_margin,
        )
        return {
            "scenario": self.scenario.name,
            "robot": "Universal Robots UR10e",
            "placement_mode": "instant",
            "source": "truck" if self.truck is not None else "conveyor",
            "truck": self.unload_report(order),
            "eoat": self.suction.name,
            "graphics_mode": "simplified" if self.simplified_graphics else "detailed",
            "ur10e_payload_kg": self.ur10e_payload_kg,
            "tool_mass_kg": self.tool_mass_kg,
            "success": final_report.stable,
            "stability_reason": final_report.reason,
            "minimum_support_ratio": final_report.minimum_support_ratio,
            "minimum_tipping_margin_m": final_report.minimum_margin,
            "state": state.as_dict(),
            "com_measurements": [],
            "measurements": [
                {
                    "package_id": planned[index].package.id,
                    "position_error_mm": math.dist(planned[index].center, state.placements[index].center) * 1_000,
                    "yaw_error_deg": abs(state.placements[index].yaw - planned[index].yaw),
                    "tilt_deg": tilts[index],
                    "active_cups": 0,
                    "grip_capacity_ratio": 0.0,
                }
                for index in range(len(planned))
            ],
        }

    def _run_palletizing(self) -> dict[str, Any]:
        sim = self.scenario.simulation
        state = StackState(self.scenario.pallet)
        planned = self.planned_placements
        planned.clear()
        measurements: list[PlacementMeasurement] = []
        com_readings: list[ComMeasurement] = []

        # The trailer is there from the first frame; the empty tool is weighed in front
        # of it, before anything has been taken out.
        self.stage_truck_load()
        probe = None
        tare = None
        if self.measure_com:
            # Imported here because `com_probe` builds on this module.
            from .com_probe import ComProbe, ProbeConfig

            probe = ComProbe(self, ProbeConfig(precise=self.precise_com))
            tare = probe.calibrate()
        remaining = list(range(len(self.scenario.packages)))
        order: list[int] = []
        while remaining:
            # Out of a trailer the cell chooses; off a conveyor it takes what arrives.
            self.record_load_drift(remaining)
            index = self.next_pick(remaining)
            remaining.remove(index)
            order.append(index)
            truth = self.scenario.packages[index]
            package = self.known_packages[index]
            with self.busy(f"Planificando la caja {len(order)}/{len(self.scenario.packages)}…"):
                # Seeing the whole load means the lookahead is over the cartons that
                # really are coming next, in the order the trailer will let them out.
                incoming = [index, *self._pick_forecast(remaining)]
                candidate: Candidate = self.planner.plan_next(
                    state, [self.known_packages[item] for item in incoming]
                )
            placement = candidate.placement
            # The tool always lines up with the carton before it seals, so the cups are
            # laid out over its own footprint; package and tool only turn together after.
            grip = self.suction.plan(package, 0)
            if not grip.feasible:
                raise RuntimeError(
                    f"{self.suction.name} cannot safely lift {package.id}: "
                    f"capacity ratio {grip.capacity_newtons / grip.required_newtons:.2f}"
                )
            if package.mass + self.tool_mass_kg > self.ur10e_payload_kg:
                raise RuntimeError(f"{package.id} exceeds the UR10e payload with the EOAT")

            pick = self._pick_pose(index, [index, *remaining])
            if self.truck is not None:
                self.set_viewer_status(
                    f"Taking {package.id}: highest carton at {pick.top_z * 1_000:.0f} mm",
                    f"{len(remaining)} left in the trailer",
                )
            self._show_active_cups(grip.active_cups)
            # The engaged cups, not the housing, are what has to sit over the carton.
            # For anything narrower than the array their block is off-centre, so the
            # arm cancels that offset on every move while this package is held.
            offset_x, offset_y = self._grasp_offset_world(grip.tool_offset, pick.yaw)
            pick_z = pick.top_z + self.cup_gap
            approach_z = max(0.96, pick_z + 0.18)
            self._move_tool(pick.x - offset_x, pick.y - offset_y, approach_z, pick.yaw)
            self._move_tool(pick.x - offset_x, pick.y - offset_y, pick_z, pick.yaw, 0.60)
            if probe is not None and tare is not None:
                blind_placement = placement
                probe.grasp(index)
                self._step(0.15)
                # Straight up and clear of what is still stacked around it before the
                # wrist tilts: weighing swings the carton about a third of a metre below
                # the tool, which is a column over if it is done inside the bay.
                self._move_tool(
                    pick.x - offset_x,
                    pick.y - offset_y,
                    max(approach_z, pick.obstacle_top + 0.14),
                    pick.yaw,
                    0.55,
                )
                reading = probe.measure(index, tare)
                package = reading.as_package(package)
                self.known_packages[index] = package
                # The balance is known now, so decide again where this one goes. A
                # package whose weight sits off to one side often wants a different
                # slot from the one chosen while assuming it was centred.
                with self.busy("Replanificando con el peso medido…"):
                    placement = self.planner.plan_next(
                        state, [package, *(self.known_packages[item] for item in incoming[1:])]
                    ).placement
                error_xy_mm = math.hypot(
                    reading.com_local[0] - truth.com[0],
                    reading.com_local[1] - truth.com[1],
                ) * 1_000
                com_readings.append(
                    ComMeasurement(
                        package_id=package.id,
                        measured_mass=reading.mass,
                        measured_com=reading.com_local,
                        declared_com=truth.com,
                        error_mm=float(math.dist(reading.com_local, truth.com) * 1000),
                        error_xy_mm=float(error_xy_mm),
                        normalised=reading.com_normalised,
                        trustworthy=reading.trustworthy,
                        replanned=_placement_differs(blind_placement, placement),
                        hidden_mm=reading.hidden_m * 1_000,
                        duration_s=reading.duration_s,
                        method=reading.method,
                    )
                )
                dx, dy, dz = (value * 1_000 for value in reading.com_local)
                self.set_viewer_status(
                    f"{package.id}  CoM ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) mm",
                    f"error {com_readings[-1].error_xy_mm:.1f} mm XY / "
                    f"{com_readings[-1].error_mm:.1f} mm 3D",
                )
            else:
                self._set_suction(index, True)
                self._step(0.15)

            world_x = sim.pallet_origin[0] + placement.x + placement.width / 2
            world_y = sim.pallet_origin[1] + placement.y + placement.depth / 2
            place_z = sim.pallet_height + placement.z + placement.height + self.cup_gap
            safe_z = max(0.94, place_z + 0.18, sim.pallet_height + state.max_height + 0.22)
            # The grasp offset is fixed in the tool frame, so it turns with the wrist.
            turned_x, turned_y = self._grasp_offset_world(grip.tool_offset, placement.yaw)
            tool_x, tool_y = world_x - turned_x, world_y - turned_y
            # Lift out vertically first: nothing else may be dragged out with it.
            self._move_tool(pick.x - offset_x, pick.y - offset_y, safe_z, pick.yaw, 0.65)
            self._move_tool(tool_x, tool_y, safe_z, placement.yaw, 1.00)
            self._move_tool(tool_x, tool_y, place_z + 0.002, placement.yaw, 0.65)

            if sim.position_noise_std > 0:
                noise = self.rng.normal(0, sim.position_noise_std, size=2)
                world_x += float(noise[0])
                world_y += float(noise[1])
                tool_x, tool_y = world_x - turned_x, world_y - turned_y
                self._move_tool(tool_x, tool_y, place_z + 0.002, placement.yaw, 0.25)
            self._step(0.15)
            if probe is not None:
                probe.release()
            else:
                self._set_suction(index, False)
            self._step(sim.settle_seconds)
            self._move_tool(tool_x, tool_y, min(1.08, place_z + 0.20), placement.yaw, 0.55)

            planned.append(placement)
            self.palletized = list(order)
            state, tilts = self._verified_state(order, planned)
            actual = state.placements[-1]
            error = math.dist(placement.center, actual.center)
            measurements.append(
                PlacementMeasurement(
                    package.id,
                    error,
                    abs(actual.yaw - placement.yaw),
                    tilts[-1],
                    len(grip.active_cups),
                    grip.capacity_newtons / grip.required_newtons,
                )
            )

        final_report = validate_stack(
            state,
            self.scenario.planner.minimum_support_ratio,
            self.scenario.planner.minimum_tipping_margin,
        )
        pallet_com = compare_pallet_com(
            estimate_pallet_com(planned, self.scenario.pallet),
            self.measure_true_pallet_com(order),
        ).as_dict()
        return {
            "scenario": self.scenario.name,
            "robot": "Universal Robots UR10e",
            "eoat": self.suction.name,
            "placement_mode": "robot",
            "source": "truck" if self.truck is not None else "conveyor",
            "truck": self.unload_report(order),
            "graphics_mode": "simplified" if self.simplified_graphics else "detailed",
            "ur10e_payload_kg": self.ur10e_payload_kg,
            "tool_mass_kg": self.tool_mass_kg,
            "success": final_report.stable,
            "stability_reason": final_report.reason,
            "minimum_support_ratio": final_report.minimum_support_ratio,
            "minimum_tipping_margin_m": final_report.minimum_margin,
            "pallet_com": pallet_com,
            "state": state.as_dict(),
            "com_measurements": [
                {
                    "package_id": item.package_id,
                    "measured_mass_kg": item.measured_mass,
                    "measured_com_mm": [value * 1_000 for value in item.measured_com],
                    "declared_com_mm": [value * 1_000 for value in item.declared_com],
                    "error_mm": item.error_mm,
                    "error_xy_mm": item.error_xy_mm,
                    "normalised": list(item.normalised),
                    "trustworthy": item.trustworthy,
                    "replanned": item.replanned,
                    "hidden_mm": item.hidden_mm,
                    "duration_s": item.duration_s,
                    "method": item.method,
                }
                for item in com_readings
            ],
            "measurements": [
                {
                    "package_id": item.package_id,
                    "position_error_mm": item.position_error_m * 1_000,
                    "yaw_error_deg": item.yaw_error_deg,
                    "tilt_deg": item.tilt_deg,
                    "active_cups": item.active_cups,
                    "grip_capacity_ratio": item.grip_capacity_ratio,
                }
                for item in measurements
            ],
        }

    def run(
        self, *, shake: bool | ShakeConfig = False, instant_place: bool = False
    ) -> dict[str, Any]:
        try:
            result = self.place_from_plan() if instant_place else self._run_palletizing()
            if shake is False:
                return result
            config = self.scenario.simulation.shake if shake is True else shake
            result["shake"] = self.run_transport_trials(config, retract_arm=not instant_place)
            result["beam"] = self.run_beam_trials(config, retract_arm=False)
            result["palletizing_success"] = result["success"]
            result["success"] = bool(result["success"])
            return result
        finally:
            if self.playback is not None:
                self.playback.hold()
            self.close()


def save_result(result: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
