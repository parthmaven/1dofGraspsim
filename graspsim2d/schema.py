"""Typed design-variable schema for the 2D 1-DoF two-finger gripper.

The schema fixes which variables the geometry generator understands, their
canonical units and their nominal values. The bounding box (lower and upper
bound of each variable) is read from a YAML file, so a new bounding box is a
config change and a rerun.

Canonical units: mm, deg, N*m, N*m/rad, Shore A, 1/m, and plain fractions.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

# Accepted input units and their conversion factor to the canonical unit.
_UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    "mm": ("mm", 1.0),
    "cm": ("mm", 10.0),
    "m": ("mm", 1000.0),
    "in": ("mm", 25.4),
    "deg": ("deg", 1.0),
    "rad": ("deg", 180.0 / math.pi),
    "N*m": ("N*m", 1.0),
    "N.m": ("N*m", 1.0),
    "Nm": ("N*m", 1.0),
    "lbf*in": ("N*m", 0.1129848),
    "N*m/rad": ("N*m/rad", 1.0),
    "shoreA": ("shoreA", 1.0),
    "1/m": ("1/m", 1.0),
    "1": ("1", 1.0),
}


@dataclass(frozen=True)
class VariableDef:
    """Schema entry: what a variable means and its canonical unit."""

    name: str
    unit: str
    nominal: float
    description: str


# Every variable the geometry generator reads. Nominals describe the nominal
# gripper: pivots 38 mm apart, tip 150 mm forward of the pivots, straight
# fingers with a flat 40 mm pad.
SCHEMA: tuple[VariableDef, ...] = (
    VariableDef("pivot_spacing", "mm", 38.0, "Distance between the two finger pivot axes."),
    VariableDef("finger_reach", "mm", 150.0, "Fingertip distance forward of the pivot line, fingers parallel."),
    VariableDef("finger_thickness", "mm", 12.0, "In-plane depth of the finger beam."),
    VariableDef("crank_offset", "mm", 0.0, "Inward offset of the distal finger section (0 = straight finger)."),
    VariableDef("crank_position", "1", 0.45, "Where the crank starts, as a fraction of the reach from the pivot."),
    VariableDef("pad_length", "mm", 40.0, "Length of the (distal) pad face along the finger."),
    VariableDef("pad_thickness", "mm", 6.0, "Elastomer pad thickness (at the crown apex for crowned pads)."),
    VariableDef("pad_angle", "deg", 0.0, "Pad face angle to the finger axis; + makes the distal end stand further inward."),
    VariableDef("pad_tip_margin", "mm", 5.0, "Distance from the fingertip to the distal end of the pad."),
    VariableDef("pad_width", "mm", 40.0, "Out-of-plane width of pad and finger (the 2D extrusion width)."),
    VariableDef("pad_shore", "shoreA", 40.0, "Pad hardness; sets the elastomer modulus (Gent 1958)."),
    VariableDef("pad_crown_curvature", "1/m", 0.0, "Pad face curvature 1/R (0 = flat; 10 = 100 mm radius)."),
    VariableDef("finger_torque", "N*m", 5.0, "Drive torque at each finger pivot (torque-limited, backdrivable)."),
    VariableDef("max_opening", "deg", 35.0, "Opening angle of each finger at the start of a grasp."),
    VariableDef("pad2_offset", "mm", 50.0, "Dual-zone only: fingertip to the centre of the proximal box pad."),
    VariableDef("pad2_length", "mm", 30.0, "Dual-zone only: length of the proximal box pad."),
    VariableDef("pad2_angle", "deg", 6.0, "Dual-zone only: face angle of the proximal box pad."),
    VariableDef("pad2_protrusion", "mm", 0.0, "Dual-zone only: box-pad face inward of the tip-pad face (+ protrudes)."),
    VariableDef("pivot_depth", "mm", 8.0, "Self-aligning only: pad pivot pin depth behind the pad face."),
    VariableDef("pivot_stiffness", "N*m/rad", 0.3, "Self-aligning only: centring spring on the pad pivot."),
    VariableDef("pivot_range", "deg", 8.0, "Self-aligning only: pad rotation limit either side of neutral."),
)

SCHEMA_BY_NAME: dict[str, VariableDef] = {v.name: v for v in SCHEMA}
PAD_STYLES = ("flat", "crowned", "dual", "pivoting")


@dataclass(frozen=True)
class GripperDesign:
    """One gripper design: a concept (topology) plus a value for every variable."""

    concept: str
    pad_style: str
    pivot_spacing: float
    finger_reach: float
    finger_thickness: float
    crank_offset: float
    crank_position: float
    pad_length: float
    pad_thickness: float
    pad_angle: float
    pad_tip_margin: float
    pad_width: float
    pad_shore: float
    pad_crown_curvature: float
    finger_torque: float
    max_opening: float
    pad2_offset: float
    pad2_length: float
    pad2_angle: float
    pad2_protrusion: float
    pivot_depth: float
    pivot_stiffness: float
    pivot_range: float
    label: str = ""

    def values(self) -> dict[str, float]:
        return {v.name: float(getattr(self, v.name)) for v in SCHEMA}

    def to_dict(self) -> dict[str, Any]:
        d = {"concept": self.concept, "pad_style": self.pad_style, "label": self.label}
        d.update({k: round(v, 4) for k, v in self.values().items()})
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GripperDesign":
        missing = [v.name for v in SCHEMA if v.name not in d]
        if missing:
            raise ValueError(f"design is missing variables: {missing}")
        kwargs = {v.name: float(d[v.name]) for v in SCHEMA}
        style = d.get("pad_style", "flat")
        if style not in PAD_STYLES:
            raise ValueError(f"pad_style must be one of {PAD_STYLES}, got {style!r}")
        return cls(concept=str(d.get("concept", "custom")), pad_style=style, label=str(d.get("label", "")), **kwargs)

    def replace(self, **changes: Any) -> "GripperDesign":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class VariableRange:
    """Bounding-box entry for one variable, in canonical units."""

    name: str
    lower: float
    upper: float
    nominal: float
    placeholder: bool
    note: str = ""

    @property
    def fixed(self) -> bool:
        return self.upper - self.lower <= 1e-12

    def clip(self, x: float) -> float:
        return float(min(max(x, self.lower), self.upper))


@dataclass
class DesignSpace:
    """The bounding box: one range per schema variable."""

    ranges: dict[str, VariableRange]
    source: str = ""

    def __getitem__(self, name: str) -> VariableRange:
        return self.ranges[name]

    def nominal_values(self) -> dict[str, float]:
        return {n: r.clip(r.nominal) for n, r in self.ranges.items()}

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for v in SCHEMA:
            r = self.ranges[v.name]
            rows.append(
                {
                    "name": v.name,
                    "unit": v.unit,
                    "nominal": r.nominal,
                    "lower": r.lower,
                    "upper": r.upper,
                    "placeholder": r.placeholder,
                    "description": v.description,
                }
            )
        return rows


def _convert(value: float, unit: str, canonical: str, name: str) -> float:
    if unit not in _UNIT_CONVERSIONS:
        raise ValueError(f"{name}: unknown unit {unit!r}; accepted: {sorted(_UNIT_CONVERSIONS)}")
    target, factor = _UNIT_CONVERSIONS[unit]
    if target != canonical:
        raise ValueError(f"{name}: unit {unit!r} is not a {canonical} unit")
    return float(value) * factor


def load_design_space(path: str | Path) -> DesignSpace:
    """Read the bounding box YAML.

    Each entry gives ``range: [lower, upper]`` (or ``value`` to fix it), an
    optional ``unit`` (converted to the canonical unit) and an optional
    ``nominal``. Variables left out are fixed at the schema nominal. Unknown
    names are an error, so a typo cannot silently drop a variable.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("variables", {}) or {}
    unknown = sorted(set(entries) - set(SCHEMA_BY_NAME))
    if unknown:
        raise ValueError(f"{path}: unknown design variables {unknown}; valid names: {sorted(SCHEMA_BY_NAME)}")
    ranges: dict[str, VariableRange] = {}
    for v in SCHEMA:
        e = entries.get(v.name)
        if e is None:
            ranges[v.name] = VariableRange(v.name, v.nominal, v.nominal, v.nominal, placeholder=True, note="not in config; fixed at nominal")
            continue
        if not isinstance(e, Mapping):
            e = {"value": e}
        unit = str(e.get("unit", v.unit))
        if "range" in e:
            lo, hi = e["range"]
        elif "value" in e:
            lo = hi = e["value"]
        else:
            raise ValueError(f"{path}: {v.name} needs 'range: [lo, hi]' or 'value'")
        lo = _convert(lo, unit, v.unit, v.name)
        hi = _convert(hi, unit, v.unit, v.name)
        if hi < lo:
            raise ValueError(f"{path}: {v.name} upper bound {hi} is below lower bound {lo}")
        nominal = _convert(e["nominal"], unit, v.unit, v.name) if "nominal" in e else v.nominal
        if not lo - 1e-9 <= nominal <= hi + 1e-9:
            warnings.warn(f"{v.name}: nominal {nominal} {v.unit} lies outside [{lo}, {hi}]; nominal designs clip it")
        ranges[v.name] = VariableRange(
            v.name, lo, hi, nominal, placeholder=bool(e.get("placeholder", False)), note=str(e.get("note", ""))
        )
    return DesignSpace(ranges=ranges, source=str(path))


@dataclass(frozen=True)
class Concept:
    """A distinct finger-shape concept: a pad style plus which variables it searches.

    ``active`` variables are searched within the bounding box, optionally
    narrowed by ``narrow`` (never widened). Everything else takes ``fixed``
    or the bounding-box nominal.
    """

    name: str
    label: str
    pad_style: str
    active: tuple[str, ...]
    fixed: Mapping[str, float] = field(default_factory=dict)
    narrow: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    idea: str = ""

    def bounds(self, space: DesignSpace) -> list[tuple[float, float]]:
        out = []
        for name in self.active:
            r = space[name]
            lo, hi = r.lower, r.upper
            if name in self.narrow:
                nlo, nhi = self.narrow[name]
                lo, hi = max(lo, nlo), min(hi, nhi)
                if hi < lo:
                    # The bounding box excludes the concept's preferred range:
                    # fall back to the bounding box rather than leaving the box.
                    lo, hi = r.lower, r.upper
            out.append((lo, hi))
        return out

    def make(self, space: DesignSpace, x: Sequence[float] | None = None, label: str = "") -> GripperDesign:
        vals = space.nominal_values()
        for name, value in self.fixed.items():
            vals[name] = space[name].clip(float(value))
        if x is None:
            x = [vals[n] for n in self.active]
        if len(x) != len(self.active):
            raise ValueError(f"{self.name}: expected {len(self.active)} values, got {len(x)}")
        for (lo, hi), name, value in zip(self.bounds(space), self.active, x):
            vals[name] = float(min(max(value, lo), hi))
        return GripperDesign(concept=self.name, pad_style=self.pad_style, label=label or self.label, **vals)

    def from_unit(self, space: DesignSpace, u: Sequence[float], label: str = "") -> GripperDesign:
        b = np.asarray(self.bounds(space), dtype=float)
        x = b[:, 0] + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (b[:, 1] - b[:, 0])
        return self.make(space, x, label=label)

    def to_vector(self, design: GripperDesign) -> np.ndarray:
        return np.array([getattr(design, n) for n in self.active], dtype=float)


def load_concepts(path: str | Path, space: DesignSpace) -> list[Concept]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    concepts = []
    for name, e in (raw.get("concepts") or {}).items():
        style = e.get("pad_style", "flat")
        if style not in PAD_STYLES:
            raise ValueError(f"concept {name}: pad_style must be one of {PAD_STYLES}")
        active = tuple(e.get("active", []))
        bad = [a for a in active if a not in SCHEMA_BY_NAME]
        bad += [f for f in (e.get("fixed") or {}) if f not in SCHEMA_BY_NAME]
        if bad:
            raise ValueError(f"concept {name}: unknown variables {bad}")
        # A variable fixed in the bounding box is not searched.
        active = tuple(a for a in active if not space[a].fixed)
        narrow = {k: (float(v[0]), float(v[1])) for k, v in (e.get("narrow") or {}).items()}
        concepts.append(
            Concept(
                name=name,
                label=str(e.get("label", name)),
                pad_style=style,
                active=active,
                fixed={k: float(v) for k, v in (e.get("fixed") or {}).items()},
                narrow=narrow,
                idea=str(e.get("idea", "")).strip(),
            )
        )
    if not concepts:
        raise ValueError(f"{path}: no concepts defined")
    return concepts
