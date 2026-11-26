"""Scenario: how each object is presented, the motion sequence, and pass criteria."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .geometry import KeepOut
from .objects import G


@dataclass(frozen=True)
class Presentation:
    insertion_depth: float  # m: object top above the fingertip line (fingers parallel)
    lateral_offset: float  # m: object centre off the jaw centreline
    fixture: str  # "table" (box rests on a table) or "holder" (released after closing)
    insertion: str = "fixed"  # "fixed", "pad_registered" or "best"
    clearance: float = 0.005  # m, pad_registered: object top above the pad's proximal end
    max_insertion: float = 0.11  # m


@dataclass(frozen=True)
class Scenario:
    presentation: dict[str, Presentation]
    closed_limit: float  # rad, finger angle at full close
    closing_speed: float  # rad/s, sets the joint damping of the torque-limited drive
    close_time: float
    settle_time: float
    lift_height: float
    lift_time: float
    hold_time: float
    shake_time: float
    shake_lateral_g: float
    shake_lateral_hz: float
    shake_vertical_g: float
    shake_vertical_hz: float
    final_hold_time: float
    max_slip: float  # m
    max_rotation: float  # rad
    crease_safety_factor: float
    crease_width: str  # "pad" or "sheet": width that carries the bending moment at the pinch exit
    friction_sf: float
    pitch_sf: float
    torque_mode: str  # "fixed": one drive torque for every object
    pressure_sf: float = 1.0
    friction_scale: float = 1.0  # multiplies every material's pad friction (sensitivity runs)
    keep_out: KeepOut | None = None
    sim: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def lateral_accel(self) -> float:
        return self.shake_lateral_g * G

    @property
    def vertical_accel(self) -> float:
        return self.shake_vertical_g * G

    def for_kind(self, kind: str) -> Presentation:
        return self.presentation[kind]

    def summary(self) -> dict[str, Any]:
        return self.raw


def load_scenario(path: str | Path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    pres = {}
    for kind in ("box", "sheet"):
        e = raw["presentation"][kind]
        mode = str(e.get("insertion", "fixed"))
        if mode not in ("fixed", "pad_registered", "best"):
            raise ValueError(f"presentation.{kind}.insertion must be fixed, pad_registered or best, not {mode!r}")
        pres[kind] = Presentation(
            insertion_depth=float(e["insertion_depth_mm"]) / 1000.0,
            lateral_offset=float(e.get("lateral_offset_mm", 0.0)) / 1000.0,
            fixture=str(e.get("fixture", "table" if kind == "box" else "holder")),
            insertion=mode,
            clearance=float(e.get("clearance_mm", 5.0)) / 1000.0,
            max_insertion=float(e.get("max_insertion_mm", 110.0)) / 1000.0,
        )
    g = raw["gripper"]
    seq = raw["sequence"]
    sh = seq["shake"]
    pc = raw["pass_criteria"]
    an = raw.get("analytic", {}) or {}
    return Scenario(
        presentation=pres,
        closed_limit=math.radians(float(g.get("closed_limit_deg", -25.0))),
        closing_speed=float(g.get("closing_speed_rad_s", 1.0)),
        close_time=float(seq.get("close_s", 0.6)),
        settle_time=float(seq.get("settle_s", 0.25)),
        lift_height=float(seq.get("lift_height_mm", 80.0)) / 1000.0,
        lift_time=float(seq.get("lift_s", 0.5)),
        hold_time=float(seq.get("hold_s", 0.3)),
        shake_time=float(sh.get("duration_s", 2.0)),
        shake_lateral_g=float(sh["lateral"]["peak_accel_g"]),
        shake_lateral_hz=float(sh["lateral"]["freq_hz"]),
        shake_vertical_g=float(sh["vertical"]["peak_accel_g"]),
        shake_vertical_hz=float(sh["vertical"]["freq_hz"]),
        final_hold_time=float(seq.get("final_hold_s", 0.5)),
        max_slip=float(pc["max_slip_mm"]) / 1000.0,
        max_rotation=math.radians(float(pc["max_rotation_deg"])),
        crease_safety_factor=float(pc.get("crease_safety_factor", 1.0)),
        crease_width=str(pc.get("crease_width", "pad")),
        friction_sf=float(an.get("friction_safety_factor", 1.5)),
        pitch_sf=float(an.get("pitch_safety_factor", 1.5)),
        torque_mode=str(g.get("torque_mode", "fixed")),
        pressure_sf=float(pc.get("pressure_safety_factor", 1.0)),
        friction_scale=float(raw.get("friction_scale", 1.0)),
        keep_out=_keep_out(g.get("keep_out")),
        sim=dict(raw.get("sim", {}) or {}),
        raw=raw,
    )


def _keep_out(e: Mapping[str, Any] | None) -> KeepOut | None:
    if not e or not e.get("circles"):
        return None
    circles = []
    for c in e["circles"]:
        x, z = (float(v) for v in c["center_mm"])
        circles.append((x, z, float(c["radius_mm"])))
    return KeepOut(
        circles=tuple(circles),
        clearance=float(e.get("clearance_mm", 0.0)),
        min_ligament=float(e.get("min_ligament_mm", 5.0)),
        blocks_objects=bool(e.get("blocks_objects", True)),
        note=str(e.get("note", "")),
    )
