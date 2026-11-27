"""Fingers drawn from concept-spec outlines (batch 1).

Each finger is the spec's outline polygon unioned with the platform root (u <= 46),
cut clear of the hub over the stroke, with its pads and its bare contact faces:
every exposed edge of the printed body beyond the root that can face an object
(the inner side, steps and walls, and both sides of the last `TIP_REGION` mm) is a
rigid "bare pad" at the bare friction for the object (test conditions rev 2.8). The
rest of the body is ordinary finger body, held to the 1 N body-contact limit.

Comb concepts (S4, S5) interleave parts out of plane. A layout (config/batch1/layouts.yaml)
puts those parts in one half of the width, layer "a" or "b"; the rest of each finger fills
both. Finger-to-finger contact is checked per layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import box as sbox
from shapely.ops import unary_union

from .geometry import BodyPiece, Finger, KeepOut, Pad, to_local, to_view
from .specs import ConceptSpec, SpecDesign, evaluate

ROOT_U = 46.0  # the platform root, kept by every concept
FACE_V = 17.75
ROOT_BACK_V = -21.25
TIP_REGION = 30.0  # mm from the tip in which both sides of the finger are contact faces
BARE_THICKNESS = 1.0  # mm, notional depth of a bare pad block (inside the body)
MERGE_TOL_DEG = 20.0  # facet pads this close in angle and sharing an end merge into one even layer
LAYERS = frozenset({"a", "b"})


@dataclass(frozen=True)
class LayerPart:
    poly: Any  # shapely geometry, finger-local
    layers: frozenset[str]
    width: float | None = None  # mm out of plane, if the layout gives it


@dataclass
class SpecFinger:
    side: str  # "outside" (drawn as the right finger) or "inside" (left)
    finger: Finger
    uncut: Any  # shapely, outline plus root, before the hub cut
    body: Any  # shapely, after the hub cut
    pads: list[Pad]  # elastomer pads
    bare: list[Pad]  # bare contact faces
    widths: list[tuple[float, float]]
    parts: list[LayerPart]
    features: dict[str, Any]
    stop_type: str | None
    stop_deg: float | None
    merged: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def width_at(self, u: float) -> float:
        return width_at(self.widths, u)

    @property
    def contact_faces(self) -> list[Pad]:
        return list(self.pads) + list(self.bare)


@dataclass
class SpecGripper:
    spec: ConceptSpec
    design: SpecDesign
    right: SpecFinger
    left: SpecFinger
    phi_meet: float  # rad
    phi_max: float  # rad: the stroke's end at this reading (or the stop's angle)
    stop_deg: float | None  # opening (deg from the meet) of a hard stop within this reading
    stop_type: str | None
    finger_gap: float  # mm: least gap between the fingers over the stroke, per layer (negative: they overlap)
    finger_gap_phi: float  # deg from the meet
    issues: list[str]

    @property
    def feasible(self) -> bool:
        return not any(i.startswith("infeasible") for i in self.issues)

    @property
    def symmetric(self) -> bool:
        return self.spec.symmetric

    @property
    def pivot_x(self) -> float:
        return 19.0

    def fingers(self) -> list[tuple[int, SpecFinger]]:
        return [(1, self.right), (-1, self.left)]

    def drawn_length(self) -> float:
        return max(float(np.asarray(f.uncut.bounds)[2]) for f in (self.right, self.left))


def width_at(widths: list[tuple[float, float]], u: float) -> float:
    """Out-of-plane width (mm) at u from [u, width] stations; a repeated u is a step (the
    distal value applies beyond it)."""
    if not widths:
        return 28.0
    us = [w[0] for w in widths]
    if u <= us[0]:
        return widths[0][1]
    if u >= us[-1]:
        return widths[-1][1]
    for (u0, w0), (u1, w1) in zip(widths[:-1], widths[1:]):
        if u0 <= u <= u1:
            if u1 - u0 < 1e-9:
                continue
            return w0 + (w1 - w0) * (u - u0) / (u1 - u0)
    return widths[-1][1]


def _poly(pts: list[tuple[float, float]]):
    g = SPolygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    return g


def _largest(g):
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda p: p.area)
    return g


def _polys(g) -> list[SPolygon]:
    if g.is_empty:
        return []
    if isinstance(g, SPolygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, SPolygon) and p.area > 1e-6]


def _pad_from(e: Mapping[str, Any], v: Mapping[str, float], widths: list[tuple[float, float]]) -> Pad:
    (u0, v0), (u1, v1) = [(evaluate(a, v), evaluate(b, v)) for a, b in e["face"]]
    th = evaluate(e.get("thickness_mm", v.get("pad_thickness", 6.0)), v)
    shore = evaluate(e.get("shore_a", v.get("pad_shore", 50.0)), v)
    w = e.get("width_mm")
    width = evaluate(w, v) if w is not None else width_at(widths, 0.5 * (u0 + u1))
    cr = e.get("crown_radius_mm")
    crown = math.inf if cr in (None, "null") else evaluate(cr, v)
    ang = math.atan2(v1 - v0, u1 - u0)
    return Pad(name=str(e.get("name", "pad")), role=str(e.get("role", "all")), face_u=0.5 * (u0 + u1), face_v=0.5 * (v0 + v1),
               angle=ang, length=math.hypot(u1 - u0, v1 - v0), thickness=th, width=width, shore=shore, crown_radius=crown,
               seat="drawn")


def _ends(p: Pad) -> tuple[np.ndarray, np.ndarray]:
    if p.profile is not None:
        return np.asarray(p.profile[0]), np.asarray(p.profile[-1])
    h = p.length / 2
    return p.centre - h * p.tangent, p.centre + h * p.tangent


def merge_facets(pads: list[Pad]) -> tuple[list[Pad], list[str]]:
    """Merge runs of flat pads that share ends, thickness, hardness, width and role, and turn by
    less than MERGE_TOL_DEG at each joint, into one even layer whose face is a not-a-knot cubic
    through the facet vertices (exact for a parabolic or cubic crown)."""
    out: list[Pad] = []
    notes: list[str] = []
    run: list[Pad] = []

    def flush() -> None:
        if len(run) == 1:
            out.append(run[0])
        elif run:
            verts = [_ends(run[0])[0]] + [_ends(p)[1] for p in run]
            V = np.asarray(verts)
            if np.all(np.diff(V[:, 0]) > 1e-6):
                cs = CubicSpline(V[:, 0], V[:, 1], bc_type="not-a-knot")
                uu = np.linspace(V[0, 0], V[-1, 0], 8 * len(run) + 1)
                prof = np.stack([uu, cs(uu)], axis=1)
            else:
                prof = V
            seg = np.hypot(*np.diff(prof, axis=0).T)
            a, b = prof[0], prof[-1]
            base = run[0]
            merged = replace(base, name="+".join(p.name for p in run) if len(run) <= 2 else f"{run[0].name}..{run[-1].name}",
                             face_u=float(0.5 * (a[0] + b[0])), face_v=float(0.5 * (a[1] + b[1])),
                             angle=math.atan2(b[1] - a[1], b[0] - a[0]), length=float(seg.sum()),
                             profile=tuple((float(x), float(y)) for x, y in prof))
            out.append(merged)
            notes.append(f"pads {', '.join(p.name for p in run)} merged into one even {base.thickness:.1f} mm layer over the crown")
        run.clear()

    for p in pads:
        if run:
            q = run[-1]
            same = (abs(q.thickness - p.thickness) < 1e-6 and abs(q.shore - p.shore) < 1e-6 and abs(q.width - p.width) < 1e-6
                    and q.role == p.role and q.profile is None and p.profile is None and not math.isfinite(q.crown_radius)
                    and not math.isfinite(p.crown_radius))
            touch = np.hypot(*(_ends(q)[1] - _ends(p)[0])) < 1e-3
            turn = abs(math.degrees(p.angle - q.angle)) < MERGE_TOL_DEG
            if same and touch and turn:
                run.append(p)
                continue
            flush()
        run.append(p)
    flush()
    return out, notes


def _layout_parts(poly, rules: list[Mapping[str, Any]], v: Mapping[str, float], default_width: float | None = None) -> list[LayerPart]:
    rest = poly
    parts = []
    for r in rules:
        u0, u1 = (evaluate(x, v) for x in r["u"])
        v_lo = evaluate(r.get("v_above", -1e3), v)
        v_hi = evaluate(r.get("v_below", 1e3), v)
        region = sbox(u0, v_lo, u1, v_hi)
        part = rest.intersection(region)
        if not part.is_empty and part.area > 1e-6:
            w = r.get("width")
            parts.append(LayerPart(part, frozenset(r["layers"]), None if w is None else evaluate(w, v)))
            rest = rest.difference(region)
    if not rest.is_empty and rest.area > 1e-6:
        parts.append(LayerPart(rest, LAYERS, default_width))
    return parts


def _apply_relief(parts: list[LayerPart], region, keep: frozenset[str]) -> list[LayerPart]:
    out = []
    for p in parts:
        if p.layers != LAYERS:
            out.append(p)
            continue
        inside = p.poly.intersection(region)
        rest = p.poly.difference(region)
        if not inside.is_empty and inside.area > 1e-6:
            out.append(LayerPart(inside, keep, p.width))
        if not rest.is_empty and rest.area > 1e-6:
            out.append(LayerPart(rest, p.layers, p.width))
    return out


CHAIN_TURN_DEG = 15.0  # bare edges meeting at less than this join one curved face...
CHAIN_TOTAL_DEG = 45.0  # ...until the face has turned this far


def _bare_faces(body, pads: list[Pad], widths: list[tuple[float, float]], reach: float, exclude=None,
                min_len: float = 0.3) -> list[Pad]:
    """Bare contact faces: the exposed edges of body + pads beyond the root that face the inner
    side, a step or a wall, plus both sides of the tip region. Edges on the hub cut (`exclude`)
    stay plain body. Polyline arcs become one curved face with a normal per sample."""
    solid = unary_union([body] + [SPolygon(p.outline()) for p in pads]).buffer(0)
    masks = [LineString(p.face_samples(25)[0]).buffer(0.25) for p in pads] + [SPolygon(p.outline()).buffer(0.05) for p in pads]
    if exclude is not None and not exclude.is_empty:
        masks.append(exclude)
    covered = unary_union(masks) if masks else None
    chains: list[list[np.ndarray]] = []
    for poly in _polys(solid):
        ring = poly.exterior
        if not ring.is_ccw:
            ring = LineString(list(ring.coords)[::-1])
        coords = np.asarray(ring.coords)
        edges: list[tuple[np.ndarray, np.ndarray]] = []
        for a, b in zip(coords[:-1], coords[1:]):
            seg = LineString([a, b])
            for piece in ([seg] if covered is None else _lines(seg.difference(covered))):
                c = np.asarray(piece.coords)
                p0, p1 = c[0], c[-1]
                L = float(np.hypot(*(p1 - p0)))
                if L < min_len:
                    continue
                mid = 0.5 * (p0 + p1)
                if mid[0] < ROOT_U - 1e-6:
                    continue
                t = (p1 - p0) / L
                n_out = np.array([t[1], -t[0]])  # CCW ring: outward normal on the right
                if not (n_out[1] > -0.5 or mid[0] > reach - TIP_REGION):
                    continue
                edges.append((p0, p1))
        cur: list[np.ndarray] = []
        turned = 0.0
        for p0, p1 in edges:
            if cur:
                d0 = cur[-1] - cur[-2]
                d1 = p1 - p0
                turn = abs(math.degrees(math.atan2(d0[0] * d1[1] - d0[1] * d1[0], float(d0 @ d1))))
                if np.hypot(*(cur[-1] - p0)) < 1e-6 and turn < CHAIN_TURN_DEG and turned + turn <= CHAIN_TOTAL_DEG:
                    cur.append(p1)
                    turned += turn
                    continue
                chains.append(cur)
            cur, turned = [p0, p1], 0.0
        if cur:
            chains.append(cur)
        # the ring closes on itself: join the last chain to the first when they meet smoothly
        if len(chains) >= 2 and np.hypot(*(chains[-1][-1] - chains[0][0])) < 1e-6:
            d0 = chains[-1][-1] - chains[-1][-2]
            d1 = chains[0][1] - chains[0][0]
            turn = abs(math.degrees(math.atan2(d0[0] * d1[1] - d0[1] * d1[0], float(d0 @ d1))))
            if turn < CHAIN_TURN_DEG:
                chains[0] = chains.pop()[:-1] + chains[0]
    out: list[Pad] = []
    for k, ch in enumerate(chains):
        # Pad's normal (-sin a, cos a) is the left normal of its tangent: list the face against the ring.
        pts = np.asarray(ch[::-1], dtype=float)
        chord = pts[-1] - pts[0]
        steps = np.hypot(*np.diff(pts, axis=0).T)
        arc = float(np.sum(steps))
        mid = 0.5 * (pts[0] + pts[-1])
        u_arc = float(np.interp(arc / 2, np.r_[0.0, np.cumsum(steps)], pts[:, 0]))
        out.append(Pad(name=f"bare{k}", role="bare", face_u=float(mid[0]), face_v=float(mid[1]),
                       angle=math.atan2(chord[1], chord[0]), length=arc, thickness=BARE_THICKNESS,
                       width=width_at(widths, u_arc), shore=math.inf, rigid=True,
                       profile=tuple((float(a), float(b)) for a, b in pts) if len(pts) > 2 else None))
    return out


def _lines(g) -> list[LineString]:
    if g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    return [x for x in getattr(g, "geoms", []) if isinstance(x, LineString) and x.length > 1e-9]


def _swept(ko: KeepOut, pivot_x: float, phi_lo: float, phi_hi: float, n: int = 48):
    phis = np.linspace(phi_lo, phi_hi, n)
    parts = []
    for x, z, r in ko.radii():
        for xm in {x, -x}:
            pts = np.vstack([to_local(np.array([[xm, z]]), ph, pivot_x) for ph in phis])
            parts.append(LineString(pts).buffer(r, quad_segs=64))
    return unary_union(parts)


def _ligament(uncut, swept) -> float:
    lo, _, hi, _ = swept.bounds
    best = math.inf
    for u in np.arange(max(lo, 0.0), min(hi, uncut.bounds[2]) + 1e-9, 0.25):
        line = LineString([(u, -200.0), (u, 200.0)])
        if not line.intersects(swept):
            continue
        mat = line.intersection(uncut)
        if mat.is_empty:
            continue
        left = mat.difference(swept)
        if mat.length - left.length < 1e-6:
            continue
        best = min(best, float(left.length))
    return best


def _view(g, phi: float, side: int, pivot_x: float = 19.0):
    out = []
    for p in _polys(g):
        out.append(SPolygon(to_view(np.asarray(p.exterior.coords), phi, pivot_x, side)))
    return unary_union(out) if out else SPolygon()


def pair_gap(right: list[LayerPart], left: list[LayerPart], phi: float) -> float:
    """Least gap (mm) between the fingers at opening phi, per shared layer; an overlap counts as
    minus the square root of its area."""
    best = math.inf
    for lay in LAYERS:
        R = unary_union([_view(p.poly, phi, 1) for p in right if lay in p.layers])
        L = unary_union([_view(p.poly, phi, -1) for p in left if lay in p.layers])
        if R.is_empty or L.is_empty:
            continue
        g = -math.sqrt(R.intersection(L).area) if R.intersects(L) else R.distance(L)
        best = min(best, float(g))
    return best


def meet_angle(right: list[LayerPart], left: list[LayerPart], lower: float, upper: float = math.radians(70.0)) -> float:
    """Opening (rad) at which the closing fingers first touch, in any shared layer (or `lower`).
    Only the parts ahead of the pivot line count, as for the parametric concepts."""
    ahead = sbox(0.0, -1e4, 1e4, 1e4)
    R = [replace(p, poly=p.poly.intersection(ahead)) for p in right]
    L = [replace(p, poly=p.poly.intersection(ahead)) for p in left]
    if pair_gap(R, L, upper) <= 0.0:
        return upper
    hi = upper
    lo = None
    for ph in np.arange(upper, lower - 1e-9, -math.radians(0.5)):
        if pair_gap(R, L, ph) <= 0.0:
            lo = ph
            break
        hi = ph
    if lo is None:
        return lower
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if pair_gap(R, L, mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def _finger_geometry(spec: ConceptSpec, block: Mapping[str, Any], v: Mapping[str, float], side: str,
                     layout: list[Mapping[str, Any]] | None):
    issues: list[str] = []
    pts = [(evaluate(a, v), evaluate(b, v)) for a, b in block["outline"]]
    outline = _poly(pts)
    if not SPolygon(pts).is_valid:
        issues.append(f"{side}: the outline crosses itself; used its repaired area")
    root = sbox(0.0, ROOT_BACK_V, ROOT_U, FACE_V)
    uncut = _largest(unary_union([outline, root]).buffer(0))
    widths = [(evaluate(a, v), evaluate(b, v)) for a, b in (block.get("body_width_mm") or [])]
    pads = [_pad_from(e, v, widths) for e in (block.get("pads") or [])]
    pads, notes = merge_facets(pads)
    for p in pads:
        if p.thickness < 1.0 - 1e-9:
            issues.append(f"{side}: pad {p.name} is {p.thickness:.2f} mm thick, under the 1 mm minimum")
    feats = {}
    for k, e in (block.get("features") or {}).items():
        feats[k] = _eval_tree(e, v)
    stop = feats.get("hard_stop") or None
    stop_type = stop.get("type") if isinstance(stop, Mapping) else None
    stop_deg = float(stop.get("opening_deg")) if isinstance(stop, Mapping) and stop.get("opening_deg") is not None else None
    return uncut, widths, pads, notes, feats, stop_type, stop_deg, issues


def _eval_tree(e: Any, v: Mapping[str, float]) -> Any:
    if isinstance(e, Mapping):
        return {k: (_eval_tree(x, v) if k not in ("type", "side", "purpose", "where") else x) for k, x in e.items()}
    if isinstance(e, (list, tuple)):
        return [_eval_tree(x, v) for x in e]
    if isinstance(e, str):
        try:
            return evaluate(e, v)
        except Exception:
            return e
    return e


def build_gripper(spec: ConceptSpec, design: SpecDesign, ko: KeepOut | None, closed_limit: float = math.radians(-25.0),
                  layout: Mapping[str, Any] | None = None) -> SpecGripper:
    """Both fingers of a spec design at its stroke reading, cut clear of the hub over that stroke."""
    v = design.v
    blocks = spec.finger_blocks()
    lay = (layout or {}).get(spec.id, {}) or {}
    geo = {}
    for side in ("outside", "inside"):
        geo[side] = _finger_geometry(spec, blocks[side], v, side, lay.get(side))
    issues = [i for side in geo for i in geo[side][7]]
    reach = design.finger_reach

    reliefs: dict[str, Any] = {}

    def parts_of(side: str, g) -> list[LayerPart]:
        rules = lay.get(side) or []
        pads = geo[side][2]
        solid = unary_union([g] + [SPolygon(p.outline()) for p in pads])
        parts = _layout_parts(solid, rules, v)
        if side in reliefs:
            parts = _apply_relief(parts, *reliefs[side])
        return parts

    def _auto_relief(right: list[LayerPart], left: list[LayerPart], lo: float, hi: float):
        """Cut each finger away, in a layer, wherever the other finger's single-layer parts pass
        through it over the stroke ("the other finger is cut away there")."""
        notes = []
        out = {}
        for side, mine, other, s_mine, s_other in (("outside", right, left, 1, -1), ("inside", left, right, -1, 1)):
            inward = sbox(-1e4, FACE_V, 1e4, 1e4)
            for lay_other in LAYERS:
                solo = [LayerPart(p.poly.intersection(inward), p.layers, p.width) for p in other
                        if p.layers == frozenset({lay_other})]
                solo = [p for p in solo if not p.poly.is_empty and p.poly.area > 1.0]
                if not solo:
                    continue
                region = []
                phis = np.linspace(lo, hi, int(math.degrees(hi - lo) / 0.5) + 1)
                for p in solo:
                    for q in _polys(p.poly):
                        prev = None
                        for ph in phis:
                            view = to_view(np.asarray(q.exterior.coords), ph, 19.0, s_other)
                            if s_mine < 0:
                                view = view * np.array([-1.0, 1.0])
                            cur = SPolygon(to_local(view, ph, 19.0))
                            region.append(cur if prev is None else unary_union([prev, cur]).convex_hull.intersection(
                                unary_union([prev, cur]).buffer(2.0)))
                            prev = cur
                reg = unary_union(region).buffer(0.5)
                keep = LAYERS - {lay_other}
                out[side] = (reg, keep)
                hit = unary_union([p.poly for p in mine if p.layers == LAYERS]).intersection(reg)
                if not hit.is_empty and hit.area > 1.0:
                    b = hit.bounds
                    notes.append(f"the {side} finger is relieved in layer {lay_other} over u {b[0]:.0f} - {b[2]:.0f} mm, "
                                 f"where the other finger's step passes ({hit.area:.0f} mm2 of section)")
        reliefs.update(out)
        r2 = _apply_relief(right, *out["outside"]) if "outside" in out else right
        l2 = _apply_relief(left, *out["inside"]) if "inside" in out else left
        return r2, l2, "; ".join(notes)

    right0, left0 = parts_of("outside", geo["outside"][0]), parts_of("inside", geo["inside"][0])
    relief_note = ""
    if lay.get("relief") == "auto":
        right0, left0, relief_note = _auto_relief(right0, left0, math.radians(-5.0), math.radians(max(spec.readings) + 5.0))
        if relief_note:
            for side in ("outside", "inside"):
                geo[side][3].append(relief_note)
    meet = meet_angle(right0, left0, closed_limit)
    stop_types = [geo[s][5] for s in geo if geo[s][5]]
    stop_degs = [geo[s][6] for s in geo if geo[s][6] is not None]
    stop_type = stop_types[0] if stop_types else None
    stop_deg = min(stop_degs) if stop_degs else None
    stroke = design.reading
    if stop_deg is not None:
        if stop_type == "l_angle" and stroke > stop_deg + 1e-9:
            issues.append(f"infeasible: an L-angle stop at {stop_deg:g} deg caps the stroke below the {stroke:g} deg reading")
        stroke = min(stroke, stop_deg)
    held = stop_deg if (stop_deg is not None and stop_deg <= design.reading + 1e-9) else None
    phi_max = meet + math.radians(stroke)

    fingers = {}
    for side, gsign in (("outside", 1), ("inside", -1)):
        uncut, widths, pads, notes, feats, st, sd, _ = geo[side]
        f_issues: list[str] = []
        body = uncut
        lig = math.inf
        hub_gap, hub_phi = math.inf, math.nan
        if ko is not None and ko.circles:
            swept = _swept(ko, 19.0, meet, phi_max)
            for p in pads:
                if SPolygon(p.outline()).intersects(swept):
                    f_issues.append(f"infeasible: {side} pad {p.name} enters the hub keep-out during the stroke")
            body = _largest(uncut.difference(swept).buffer(0))
            lig = _ligament(uncut, swept)
            if lig < ko.min_ligament:
                f_issues.append(f"infeasible: the {side} finger keeps {lig:.1f} mm beside the hub cut (minimum {ko.min_ligament:.1f} mm)")
            solid = unary_union([body] + [SPolygon(p.outline()) for p in pads])
            for ph in np.linspace(meet, phi_max, 91):
                for x, z, r in ko.circles:
                    for xm in {x, -x}:
                        c = to_local(np.array([[xm, z]]), ph, 19.0)[0]
                        gap = solid.distance(Point(c)) - r
                        if gap < hub_gap:
                            hub_gap, hub_phi = gap, math.degrees(ph - meet)
        bare = _bare_faces(body, pads, widths, reach, exclude=swept.buffer(0.3) if ko is not None and ko.circles else None)
        pieces = tuple(BodyPiece(f"body{i}", np.asarray(p.exterior.coords)[:-1]) for i, p in enumerate(_polys(body)))
        fin = Finger(design=design.shim(), body=pieces, pads=tuple(pads) + tuple(bare), issues=tuple(f_issues),
                     cut_body=tuple(pc.poly for pc in pieces), ligament=lig, hub_clearance=hub_gap, hub_clearance_phi=hub_phi,
                     phi_meet=meet, phi_max=phi_max)
        parts = parts_of(side, body)
        fingers[side] = SpecFinger(side, fin, uncut, body, pads, bare, widths, parts, feats, st, sd, notes, f_issues)
        issues += f_issues
    if ko is not None and ko.brackets:
        for side, f in fingers.items():
            gap, where, name = _bracket_gap(f, ko, meet, phi_max)
            if gap < 0.0:
                issues.append(f"infeasible: the {side} finger meets the {name} at {where:.0f} deg of opening")
    gmin, gphi = math.inf, math.nan
    for ph in np.linspace(meet + math.radians(0.5), phi_max, 31):
        g = pair_gap(fingers["outside"].parts, fingers["inside"].parts, ph)
        if g < gmin:
            gmin, gphi = g, math.degrees(ph - meet)
    if gmin < -0.5:
        issues.append(f"infeasible: the fingers overlap in the same layer at {gphi:.0f} deg of opening")
    if design.finger_reach > 160.0 + 1e-6:
        issues.append(f"flagged: {design.finger_reach:.0f} mm reach is past the 160 mm length cap")
    return SpecGripper(spec, design, fingers["outside"], fingers["inside"], meet, phi_max, held, stop_type, gmin, gphi, issues)


def _bracket_gap(f: SpecFinger, ko: KeepOut, phi_lo: float, phi_hi: float, n: int = 61) -> tuple[float, float, str]:
    beyond = sbox(ko.root_length, -1e4, 1e4, 1e4)
    solid = unary_union([f.body.intersection(beyond)] + [SPolygon(p.outline()) for p in f.pads])
    solid = solid.difference(Point(0.0, 0.0).buffer(ko.hinge_radius, quad_segs=32))
    best = (math.inf, math.nan, "")
    if solid.is_empty:
        return best
    for ph in np.linspace(phi_lo, phi_hi, n):
        for br in ko.brackets:
            for geom in getattr(br.region, "geoms", [br.region]):
                local = SPolygon(to_local(np.asarray(geom.exterior.coords), ph, 19.0))
                g = -math.sqrt(solid.intersection(local).area) if solid.intersects(local) else solid.distance(local)
                if g < best[0]:
                    best = (float(g), math.degrees(ph - phi_lo), br.name)
    return best
