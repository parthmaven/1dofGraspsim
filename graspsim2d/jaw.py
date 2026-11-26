"""Two-finger jaws for batch 1.

A jaw is the pair of fingers on the platform's pivots, drawn from a concept spec (asymmetric
where the spec says) or from the first pass's parametric finger, mirrored. Each finger's
surfaces are scored in three groups (test conditions rev 2.9, "Bare contact faces"):

* pads: elastomer;
* marked bare faces: printed faces a spec states as contacts (config/batch1/marked_faces.yaml),
  rigid against the object's own surface stiffness, scored as grip against the object's limits;
* the rest of the body, unmarked bare faces included: in T1 - T3 its load on the object counts
  toward the 1 N body limit; in T4 and T5 it is scored against the bin's and case's limits.

Every bare surface, marked or not, takes the friction table's bare value for the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import box as sbox
from shapely.ops import unary_union

from .geometry import Finger, Pad, build_finger
from .outline import LAYERS, SpecFinger, SpecGripper, _polys, width_at
from .schema import GripperDesign
from .specs import evaluate

BARE_INSET = 1.6  # mm the body is cut back under a bare face; its MuJoCo elements travel 1.5 mm
REFERENCE_BRAKE = 30.0  # N*m per gripper (test conditions rev 2.9)
SHORT_FACE = 20.0  # mm: faces shorter than this run at the fine element pitch


@dataclass(frozen=True)
class Face:
    pad: Pad
    side: int  # +1 right (the spec's outside finger), -1 left
    kind: str  # "pad", "marked" (a stated bare contact) or "bare"
    width: float  # mm out of plane at the face's middle
    width_table: tuple[tuple[float, float], ...] | None = None
    width_scale: float = 1.0  # 0.5 on a comb part that fills one layer and gives no width
    # (u0, u1, width, scale) along a face that crosses comb parts: a part's own width, or the table's times the scale
    width_runs: tuple[tuple[float, float, float | None, float], ...] | None = None

    @property
    def name(self) -> str:
        return self.pad.name

    @property
    def rigid(self) -> bool:
        return self.kind != "pad"

    @property
    def grip(self) -> bool:
        return self.kind != "bare"

    @property
    def curved(self) -> bool:
        return self.pad.profile is not None or self.pad.crowned

    def width_at(self, u: float) -> float:
        if self.width_runs:
            _, _, fixed, scale = min(self.width_runs, key=lambda r: max(r[0] - u, u - r[1], 0.0))
            if fixed is not None:
                return fixed
            return (width_at(list(self.width_table), u) if self.width_table else self.pad.width) * scale
        if self.width_table:
            return width_at(list(self.width_table), u) * self.width_scale
        return self.width


@dataclass
class JawSide:
    side: int
    faces: list[Face]
    body_mj: list[SPolygon]  # finger-local printed body, cut back under every bare face (MuJoCo)
    body_screen: list[SPolygon]  # cut back under the marked faces only (the screen's body test)
    outline: Any  # shapely: body plus pads, for drawing and table/hub checks
    widths: list[tuple[float, float]] | None
    default_width: float
    parts: list | None = None  # the spec's comb parts (outline.LayerPart), finger-local

    def width_at(self, u: float) -> float:
        return width_at(self.widths, u) if self.widths else self.default_width

    def layers_at(self, u: float, v: float) -> frozenset[str]:
        """The layers of the part nearest a finger-local point, as `_part_rule` finds it; both off a comb."""
        best, layers = 0.6, LAYERS
        for p in self.parts or ():
            d = p.poly.distance(Point(u, v))
            if d < best - 1e-6:
                best, layers = d, p.layers
        return layers

    @property
    def grip_faces(self) -> list[Face]:
        return [f for f in self.faces if f.grip]

    @property
    def pads(self) -> list[Face]:
        return [f for f in self.faces if f.kind == "pad"]

    @property
    def marked(self) -> list[Face]:
        return [f for f in self.faces if f.kind == "marked"]

    @property
    def unmarked(self) -> list[Face]:
        return [f for f in self.faces if f.kind == "bare"]


@dataclass
class Jaw:
    concept: str  # "S1".."S6", "P0" (the platform from the STEP) or a first-pass concept name
    label: str
    sides: dict[int, JawSide]
    phi_meet: float  # rad
    phi_max: float  # rad: the stroke's end at this reading, or a held stop
    torque: float  # N*m per finger motor
    reading: float  # deg
    stop_deg: float | None
    stop_type: str | None
    symmetric: bool
    brakes: tuple[float, ...]  # N*m per gripper
    reach: float  # mm
    issues: list[str] = field(default_factory=list)
    source: Any = None
    pivot_x: float = 19.0
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not any(i.startswith("infeasible") for i in self.issues)

    @property
    def right(self) -> JawSide:
        return self.sides[1]

    @property
    def left(self) -> JawSide:
        return self.sides[-1]

    def face_counts(self) -> dict[str, int]:
        out = {}
        for s, js in self.sides.items():
            tag = "R" if s > 0 else "L"
            for k in ("pad", "marked", "bare"):
                out[f"{tag}_{k}"] = sum(1 for f in js.faces if f.kind == k)
        return out


def convex_pieces(poly, min_area: float = 0.3, tol: float = 0.2) -> list[SPolygon]:
    """Split a finger-local polygon into convex pieces for MuJoCo: simplified to `tol` mm, cut into
    slabs across u at every vertex, then merged wherever two pieces share an edge and their union
    stays convex."""
    out: list[SPolygon] = []
    for g0 in _polys(poly):
        g = g0.simplify(tol, preserve_topology=True)
        if g.is_empty or g.area < min_area:
            continue
        coords = list(g.exterior.coords) + [c for r in g.interiors for c in r.coords]
        xs = sorted({round(float(c[0]), 6) for c in coords})
        pieces = []
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x1 - x0 < 1e-6:
                continue
            pieces += _polys(g.intersection(sbox(x0, -1e4, x1, 1e4)))
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(pieces):
                j = i + 1
                while j < len(pieces):
                    a, b = pieces[i], pieces[j]
                    if a.bounds[2] < b.bounds[0] - 1e-6 or b.bounds[2] < a.bounds[0] - 1e-6 or a.intersection(b).length < 1e-6:
                        j += 1
                        continue
                    u = unary_union([a, b])
                    if u.geom_type == "Polygon" and u.convex_hull.area - u.area <= 1e-4 * u.area + 1e-6:
                        pieces[i] = u.simplify(1e-6)
                        pieces.pop(j)
                        changed = True
                        continue
                    j += 1
                i += 1
        out += [m for m in pieces if m.area >= min_area]
    return out


def _strip(face: Pad, depth: float):
    pts = face.face_samples(max(9, 2 * len(face.profile or ())))[0]
    return LineString(pts).buffer(depth, cap_style="flat", join_style="mitre")


def _cut_back(body_polys: list[SPolygon], faces: list[Face], depth: float = BARE_INSET) -> list[SPolygon]:
    if not faces:
        return [p for p in body_polys if p.area > 0.3]
    cut = unary_union([_strip(f.pad, depth) for f in faces])
    out = []
    for p in body_polys:
        out += [g for g in _polys(p.difference(cut)) if g.area > 0.3]
    return out


def load_marks(path: str | Path) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _marked(pad: Pad, rules: list[Mapping[str, Any]], v: Mapping[str, float]) -> bool:
    pts = pad.face_samples(9)[0]
    steps = np.hypot(*np.diff(pts, axis=0).T)
    u_mid = float(np.interp(steps.sum() / 2, np.r_[0.0, np.cumsum(steps)], pts[:, 0]))
    for r in rules:
        lo, hi = (evaluate(x, v) for x in r["u"])
        if lo - 0.5 <= u_mid <= hi + 0.5:
            return True
    return False


def _part_rule(sf: SpecFinger, pt: Point) -> tuple[float | None, float]:
    """(width, scale) from the part nearest a point on a face, the first part on a tie: a comb part's own width, half
    the table's on a comb part that gives none, or the table's on the full finger."""
    best, rule = 0.6, (None, 1.0)
    for p in sf.parts:
        d = p.poly.distance(pt)
        if d < best - 1e-6:
            best = d
            if p.layers == LAYERS:
                rule = (None, 1.0)
            else:
                rule = (float(p.width), 1.0) if p.width is not None else (None, 0.5)
    return rule


def _layer_width(sf: SpecFinger, pad: Pad) -> tuple[float, tuple | None, float, tuple | None]:
    """Width of a bare face, point by point: the comb part under each point decides it. A face that crosses parts gets
    runs along u; one that stays on a part, or runs across u at one station, takes its middle's part."""
    table = tuple(sf.widths) if sf.widths else None
    # The end points sit on corners that the next part shares.
    pts = pad.face_samples(max(9, int(pad.length / 0.25) + 1))[0][1:-1]
    rules = [_part_rule(sf, Point(float(u), float(v))) for u, v in pts]
    fixed, scale = _part_rule(sf, Point(pad.face_u, pad.face_v))
    w_mid = fixed if fixed is not None else pad.width * scale
    u = pts[:, 0]
    if len(set(rules)) == 1 or float(np.ptp(u)) < 0.5:
        if fixed is not None:
            return w_mid, None, 1.0, None
        return w_mid, table, scale, None
    runs: list[list] = []
    for k, r in enumerate(rules):
        if runs and (runs[-1][2], runs[-1][3]) == r:
            runs[-1][1] = float(u[k])
            continue
        if runs:
            edge = 0.5 * (runs[-1][1] + float(u[k]))
            runs[-1][1] = edge
            runs.append([edge, float(u[k]), *r])
        else:
            runs.append([float(u[k]), float(u[k]), *r])
    return w_mid, table, 1.0, tuple(sorted((min(a, b), max(a, b), f, s) for a, b, f, s in runs))


def jaw_from_spec(g: SpecGripper, marks: Mapping[str, Any] | None = None, torque: float | None = None) -> Jaw:
    spec, design = g.spec, g.design
    v = design.v
    rules = (marks or {}).get(spec.id, {}) or {}
    sides = {}
    for side, sf in ((1, g.right), (-1, g.left)):
        key = "outside" if side > 0 else "inside"
        r = rules.get(key)
        if r is None and key == "inside" and spec.symmetric:
            r = rules.get("outside")
        r = r or []
        faces = [Face(p, side, "pad", p.width) for p in sf.pads]
        for b in sf.bare:
            w, table, scale, runs = _layer_width(sf, b)
            faces.append(Face(b, side, "marked" if _marked(b, r, v) else "bare", w, table, scale, runs))
        body = _polys(sf.body)
        bare_all = [f for f in faces if f.rigid]
        bare_marked = [f for f in faces if f.kind == "marked"]
        outline = unary_union([sf.body] + [SPolygon(p.outline()) for p in sf.pads])
        sides[side] = JawSide(side, faces, _cut_back(body, bare_all), _cut_back(body, bare_marked), outline,
                              list(sf.widths) if sf.widths else None, sf.width_at(60.0), list(sf.parts))
    tau = float(v.get("finger_torque", 3.0)) if torque is None else torque
    brakes = tuple(sorted({REFERENCE_BRAKE, *[float(b) for b in spec.brake_ratings()]}))
    notes = list(dict.fromkeys(g.right.merged + g.left.merged))
    return Jaw(spec.id, design.label or f"{spec.id} nominal", sides, g.phi_meet, g.phi_max, tau, design.reading,
               g.stop_deg, g.stop_type, spec.symmetric, brakes, design.finger_reach, list(g.issues), g, 19.0, notes)


def jaw_from_design(d: GripperDesign, sc, reading: float | None = None, concept: str | None = None) -> Jaw:
    """A first-pass parametric design (r2-09, the padded platform stand-in), mirrored. `reading`
    (deg of opening from the meet) replaces the design's own `max_opening`."""
    if reading is not None:
        d = d.replace(max_opening=float(reading))
    f: Finger = build_finger(d, sc.keep_out, sc.closed_limit)
    body_np = list(f.cut_body) if f.cut_body else [b.poly for b in f.body]
    body = [g for P in body_np for g in _polys(SPolygon(P).buffer(0))]
    faces = [Face(p, 1, "pad", p.width) for p in f.pads]
    w = max(p.width for p in f.pads) if f.pads else 28.0
    outline = unary_union(body + [SPolygon(p.outline()) for p in f.pads])
    js = JawSide(1, faces, body, body, outline, None, w)
    jl = JawSide(-1, [Face(x.pad, -1, x.kind, x.width) for x in faces], body, body, outline, None, w)
    return Jaw(concept or d.concept, d.label or d.concept, {1: js, -1: jl}, f.phi_meet, f.phi_max, d.finger_torque,
               float(d.max_opening), None, None, True, (REFERENCE_BRAKE,), d.finger_reach, list(f.issues), f,
               d.pivot_spacing / 2)


def element_layout(face: Face, pitch: float, fine_pitch: float, short: float = SHORT_FACE) -> dict[str, np.ndarray]:
    """Element centres along a face (finger-local mm), inward normals, lengths and widths, at the
    fine pitch on faces shorter than `short` or curved."""
    pad = face.pad
    p = fine_pitch if (pad.length < short or face.curved) else pitch
    n = max(int(round(pad.length / p)), 1)
    if pad.length < short:
        n = max(n, 5)
    pts, nrm, ds, thick = pad.face_samples(2 * n + 1)
    centres, normals, th = pts[1::2], nrm[1::2], thick[1::2]
    seg = np.full(n, pad.length / n)
    if pad.crowned or pad.profile is not None:
        seg = np.full(n, float(np.sum(ds)) / n)
    widths = np.array([face.width_at(float(c[0])) for c in centres])
    return {"c": centres, "n": normals, "seg": seg, "th": th, "w": widths}
