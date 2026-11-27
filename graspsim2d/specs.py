"""Concept specs (the review team's batch-1 template, revision 2) and the designs they define.

A spec draws each finger as an outline polygon in the finger-local (u, v) frame, with pads
on its faces, out-of-plane widths along u and labelled features. Any number in the finger
blocks may be a quoted arithmetic expression in the spec's variables: its
`search.new_variables` plus the existing design variables it lists under
`search.existing_variables` (finger_reach, finger_torque, pad_thickness, pad_shore, ...).
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .schema import SCHEMA, GripperDesign

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "atan": math.atan, "atan2": math.atan2, "asin": math.asin, "acos": math.acos, "radians": math.radians,
          "degrees": math.degrees, "hypot": math.hypot}
_CONSTS = {"pi": math.pi}


class SpecError(ValueError):
    pass


def evaluate(expr: Any, values: Mapping[str, float]) -> float:
    """A spec number: a YAML number, or a string holding an arithmetic expression in `values`."""
    if isinstance(expr, bool):
        raise SpecError(f"expected a number, got {expr!r}")
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str):
        raise SpecError(f"expected a number or an expression, got {expr!r}")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise SpecError(f"bad expression {expr!r}: {e}") from None

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name):
            if n.id in values:
                return float(values[n.id])
            if n.id in _CONSTS:
                return _CONSTS[n.id]
            raise SpecError(f"expression {expr!r} uses {n.id!r}, which is not a variable of this spec")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise SpecError(f"expression {expr!r}: {ast.dump(n)[:60]} is not allowed")

    return ev(tree)


def names_in(expr: Any) -> set[str]:
    if not isinstance(expr, str):
        return set()
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNCS and n.id not in _CONSTS}


@dataclass(frozen=True)
class SpecVariable:
    name: str
    lo: float
    hi: float
    nominal: float
    unit: str = "mm"
    meaning: str = ""
    source: str = "new"  # "new" (search.new_variables) or "existing" (search.existing_variables)

    @property
    def fixed(self) -> bool:
        return self.hi - self.lo <= 1e-12


@dataclass
class ConceptSpec:
    """One concept spec file."""

    id: str
    name: str
    author: str
    path: str
    sha256: str
    modified_utc: str
    raw: dict[str, Any]
    variables: dict[str, SpecVariable]
    readings: tuple[float, ...]
    symmetric: bool
    asymmetric_for: tuple[str, ...]
    outside: dict[str, Any]
    inside: dict[str, Any] | None  # None: the mirror image of `outside`
    issues: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return Path(self.path).stem

    def nominal(self) -> dict[str, float]:
        return {k: v.nominal for k, v in self.variables.items()}

    def free(self) -> list[SpecVariable]:
        return [v for v in self.variables.values() if not v.fixed]

    def finger_blocks(self) -> dict[str, dict[str, Any]]:
        """The right ('outside') and left ('inside') finger blocks; a symmetric spec's inside is its outside."""
        return {"outside": self.outside, "inside": self.inside if self.inside is not None else self.outside}

    def brake_ratings(self) -> list[float]:
        """Finger-side brake ratings (N*m) the spec names beyond the reference drive's, read from its notes."""
        import re

        text = " ".join(str(self.raw.get(k, "")) for k in ("notes",))
        out = []
        for m in re.finditer(r"brake[^.]{0,80}?at least\s+([0-9.]+)\s*N\s*[·.]?\s*m at the output", text):
            out.append(float(m.group(1)))
        return out

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "author": self.author, "file": Path(self.path).name, "sha256": self.sha256[:12],
                "modified_utc": self.modified_utc, "readings": list(self.readings), "symmetric": self.symmetric,
                "variables": {k: [v.lo, v.hi, v.nominal] for k, v in self.variables.items()}}


_EXISTING_DEFAULTS = {v.name: v.nominal for v in SCHEMA}


def _range(e: Any, name: str) -> tuple[float, float]:
    if isinstance(e, (int, float)):
        return float(e), float(e)
    if isinstance(e, Mapping) and "range" in e:
        e = e["range"]
    if not isinstance(e, (list, tuple)) or len(e) != 2:
        raise SpecError(f"variable {name}: expected [lo, hi], got {e!r}")
    lo, hi = float(e[0]), float(e[1])
    if hi < lo:
        raise SpecError(f"variable {name}: range {lo}..{hi} is reversed")
    return lo, hi


def load_spec(path: str | Path, reach_cap: float | None = None) -> ConceptSpec:
    """Read a concept spec. `reach_cap` replaces the top of finger_reach's range (the flagged 170 mm run)."""
    path = Path(path)
    text = path.read_text()
    raw = yaml.safe_load(text) or {}
    sha = hashlib.sha256(text.encode()).hexdigest()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    search = raw.get("search", {}) or {}
    ex = search.get("existing_variables", {}) or {}
    variables: dict[str, SpecVariable] = {}
    issues: list[str] = []
    for name, val in (ex.get("fixed") or {}).items():
        variables[name] = SpecVariable(name, float(val), float(val), float(val), source="existing")
    narrow = ex.get("narrow") or {}
    nominal = ex.get("nominal") or {}
    for name in list(ex.get("active") or []) + [n for n in narrow if n not in (ex.get("active") or [])]:
        if name in variables:
            continue
        if name in narrow:
            lo, hi = _range(narrow[name], name)
        else:
            d = _EXISTING_DEFAULTS.get(name)
            if d is None:
                raise SpecError(f"{path.name}: existing variable {name} has no range")
            lo = hi = float(d)
        nom = float(nominal.get(name, 0.5 * (lo + hi)))
        variables[name] = SpecVariable(name, lo, hi, nom, source="existing")
    for name, e in (search.get("new_variables") or {}).items():
        lo, hi = _range(e, name)
        nom = float(e.get("nominal", 0.5 * (lo + hi))) if isinstance(e, Mapping) else lo
        variables[name] = SpecVariable(name, lo, hi, nom, unit=str((e or {}).get("unit", "mm")) if isinstance(e, Mapping) else "mm",
                                       meaning=str((e or {}).get("meaning", "")) if isinstance(e, Mapping) else "", source="new")
    if reach_cap is not None and "finger_reach" in variables:
        v = variables["finger_reach"]
        variables["finger_reach"] = SpecVariable(v.name, v.lo, max(v.hi, reach_cap), v.nominal, v.unit,
                                                 v.meaning + f" (flagged run to {reach_cap:g} mm)", v.source)
    for v in variables.values():
        if not v.lo - 1e-9 <= v.nominal <= v.hi + 1e-9:
            issues.append(f"{v.name}: nominal {v.nominal:g} is outside its range {v.lo:g}..{v.hi:g}")
    fingers = raw.get("fingers", {}) or {}
    outside = fingers.get("outside")
    if not isinstance(outside, Mapping):
        raise SpecError(f"{path.name}: fingers.outside must be a finger block")
    inside_raw = fingers.get("inside", "same")
    inside = None if inside_raw in (None, "same") else dict(inside_raw)
    symmetric = bool(fingers.get("symmetric", inside is None))
    readings = tuple(float(r) for r in ((raw.get("stroke") or {}).get("readings") or [45, 90]))
    spec = ConceptSpec(id=str(raw.get("id", path.stem)), name=str(raw.get("name", "")), author=str(raw.get("author", "")),
                       path=str(path), sha256=sha, modified_utc=mtime, raw=raw, variables=variables, readings=readings,
                       symmetric=symmetric and inside is None, asymmetric_for=tuple(fingers.get("asymmetric_for") or []),
                       outside=dict(outside), inside=inside, issues=issues)
    _check_names(spec)
    return spec


def _check_names(spec: ConceptSpec) -> None:
    known = set(spec.variables)
    missing: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str):
            missing.update(names_in(x) - known)
        elif isinstance(x, Mapping):
            for k, v in x.items():
                if k not in ("material", "purpose", "where", "name", "role", "side", "type", "root"):
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    for block in spec.finger_blocks().values():
        walk({k: block.get(k) for k in ("outline", "body_width_mm", "pads", "features")})
    if missing:
        raise SpecError(f"{spec.id}: expressions use undefined variables {sorted(missing)}")


def load_specs(paths: list[str | Path], reach_caps: Mapping[str, float] | None = None) -> dict[str, ConceptSpec]:
    out = {}
    for p in paths:
        s = load_spec(p)
        cap = (reach_caps or {}).get(s.id)
        if cap is not None:
            s = load_spec(p, reach_cap=cap)
        out[s.id] = s
    return out


@dataclass(frozen=True)
class SpecDesign:
    """One design of a concept spec: a value for every variable, at one stroke reading."""

    spec_id: str
    values: tuple[tuple[str, float], ...]
    reading: float  # deg, the stroke reading this design runs at
    label: str = ""

    @classmethod
    def make(cls, spec: ConceptSpec, values: Mapping[str, float], reading: float, label: str = "") -> "SpecDesign":
        full = spec.nominal()
        full.update({k: float(v) for k, v in values.items() if k in spec.variables})
        return cls(spec.id, tuple(sorted(full.items())), float(reading), label)

    @property
    def v(self) -> dict[str, float]:
        return dict(self.values)

    @property
    def finger_reach(self) -> float:
        return self.v.get("finger_reach", 150.0)

    @property
    def finger_torque(self) -> float:
        return self.v.get("finger_torque", 3.0)

    def replace(self, **changes: float) -> "SpecDesign":
        d = self.v
        d.update(changes)
        return SpecDesign(self.spec_id, tuple(sorted(d.items())), self.reading, self.label)

    def key(self) -> str:
        return self.spec_id + ":" + ",".join(f"{k}={v:.4g}" for k, v in self.values) + f"@{self.reading:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec_id, "reading_deg": self.reading, "label": self.label, **{k: round(v, 4) for k, v in self.values}}

    def shim(self) -> GripperDesign:
        """A GripperDesign carrying the values the shared helpers read (pivots, reach, torque, stroke)."""
        base = {v.name: float(v.nominal) for v in SCHEMA}
        vals = self.v
        base.update({k: vals[k] for k in base if k in vals})
        base["max_opening"] = self.reading
        base["finger_thickness"] = 39.0
        base["crank_offset"] = 0.0
        return GripperDesign(concept=self.spec_id, pad_style="flat", label=self.label or self.spec_id, **base)


def lhs(spec: ConceptSpec, n: int, rng: np.random.Generator) -> list[dict[str, float]]:
    """Latin hypercube over the spec's free variables (the others at their nominal)."""
    free = spec.free()
    out = []
    if not free:
        return [spec.nominal() for _ in range(n)]
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, len(free)))
    pts = cut[:n, None] + u * (cut[1:] - cut[:n])[:, None]
    for j in range(len(free)):
        rng.shuffle(pts[:, j])
    for i in range(n):
        vals = spec.nominal()
        for j, var in enumerate(free):
            vals[var.name] = var.lo + pts[i, j] * (var.hi - var.lo)
        out.append(vals)
    return out
