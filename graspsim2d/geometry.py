"""2D finger geometry for each concept, and the jaw kinematics.

Finger-local frame (u, v), in mm: origin at the finger's pivot, u along the
finger toward the tip, v inward toward the jaw centreline. A right-finger
point (u, v) at opening angle phi (rad, + opens) sits in the gripper plane at

    x = p/2 + u sin(phi) - v cos(phi)
    z = -u cos(phi) - v sin(phi)

with the gripper origin midway between the pivots and z pointing away from
the fingertips (up when the gripper points down). The left finger is the
mirror image (x -> -x). The jaw gap at a contact u, v is therefore
B = p + 2u sin(phi) - 2v cos(phi), as in lit review section 8.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union

from .schema import GripperDesign

MOUNT_MIN = 1.0  # mm of rigid mount left under the thinnest point of a pad
PAD_MIN_THICKNESS = 1.0  # mm of elastomer left at the ends of a crowned pad

@dataclass(frozen=True)
class Bracket:
    """Plan footprint of a fixed bracket beside the finger root, in gripper
    coordinates (mm), less the keep-out circles and the hinge discs, which are
    checked on their own."""

    name: str
    region: Any  # shapely geometry
    outline: Any = None  # the whole footprint, for drawing


@dataclass(frozen=True)
class KeepOut:
    """Region of the gripper plane that fingers and objects must stay out of,
    such as the footprint of a bracket above the jaw. Circles are (x, z, r) in
    mm in gripper coordinates (origin midway between the pivots, z = 0 on the
    pivot line, fingertips at negative z). Each finger gets a cutout that
    clears the circles over its whole stroke, from fully closed to fully open.

    Brackets are checked over the same stroke but not cut: a finger that meets
    one is infeasible. Within `hinge_radius` of each pivot the bracket and the
    finger's hinge interleave in height, and the first `root_length` mm of the
    finger runs below the brackets, as on the platform finger, so neither is
    checked."""

    circles: tuple[tuple[float, float, float], ...]
    clearance: float = 0.0  # mm added to every radius
    min_ligament: float = 5.0  # mm of finger material that must remain beside the cutout
    blocks_objects: bool = True  # objects may not enter the region either
    note: str = ""
    brackets: tuple[Bracket, ...] = ()
    hinge_radius: float = 0.0  # mm
    root_length: float = 0.0  # mm along the finger from the pivot

    def radii(self) -> list[tuple[float, float, float]]:
        return [(x, z, r + self.clearance) for x, z, r in self.circles]

    def max_object_top(self, x_lo: float, x_hi: float) -> float:
        """Highest z (mm) the top of an object spanning x_lo..x_hi may reach."""
        top = math.inf
        if not self.blocks_objects:
            return top
        for x, z, r in self.radii():
            dx = 0.0 if x_lo <= x <= x_hi else min(abs(x - x_lo), abs(x - x_hi))
            if dx < r:
                top = min(top, z - math.sqrt(r * r - dx * dx))
        return top


def rect_polygon(cu: float, cv: float, angle: float, hl: float, ht: float) -> np.ndarray:
    """Corners (CCW in u-v) of a rectangle centred at (cu, cv), long axis at `angle`."""
    t = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-math.sin(angle), math.cos(angle)])
    c = np.array([cu, cv])
    return np.array([c - hl * t - ht * n, c + hl * t - ht * n, c + hl * t + ht * n, c - hl * t + ht * n])


@dataclass(frozen=True)
class BodyPiece:
    """Rigid finger part: a convex polygon, plus its rectangle parameters if it is one."""

    name: str
    poly: np.ndarray
    rect: tuple[float, float, float, float, float] | None = None  # cu, cv, angle, hl, ht


@dataclass(frozen=True)
class Pad:
    """An elastomer pad on the finger.

    The face passes through (face_u, face_v) (the apex for a crowned pad)
    with tangent (cos a, sin a); a > 0 tilts the distal end inward. The
    inward normal is (-sin a, cos a).
    """

    name: str
    role: str  # "all", "sheet" or "box": the object kind the pad is meant for
    face_u: float
    face_v: float
    angle: float  # rad
    length: float  # mm
    thickness: float  # mm (apex thickness if crowned)
    width: float  # mm, out of plane
    shore: float
    crown_radius: float = math.inf  # mm
    pivot_depth: float | None = None  # mm behind the face; None = rigid mount
    pivot_stiffness: float = 0.0  # N*m/rad
    pivot_range: float = 0.0  # rad

    @property
    def tangent(self) -> np.ndarray:
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def normal(self) -> np.ndarray:
        return np.array([-math.sin(self.angle), math.cos(self.angle)])

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.face_u, self.face_v])

    @property
    def crowned(self) -> bool:
        return math.isfinite(self.crown_radius)

    @property
    def pivot(self) -> np.ndarray | None:
        if self.pivot_depth is None:
            return None
        return self.centre - self.pivot_depth * self.normal

    def sagitta(self) -> float:
        if not self.crowned:
            return 0.0
        r, h = self.crown_radius, self.length / 2
        return r - math.sqrt(max(r * r - h * h, 0.0))

    def face_samples(self, n: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Points on the face (proximal to distal), inward normals, arc length per
        point (mm) and local elastomer thickness (mm)."""
        t, nn, c = self.tangent, self.normal, self.centre
        if not self.crowned:
            s = np.linspace(-self.length / 2, self.length / 2, n)
            pts = c + s[:, None] * t
            normals = np.repeat(nn[None, :], n, axis=0)
            ds = np.full(n, self.length / (n - 1))
            ds[[0, -1]] *= 0.5
            thick = np.full(n, self.thickness)
            return pts, normals, ds, thick
        r = self.crown_radius
        th_max = math.asin(min(self.length / (2 * r), 1.0))
        th = np.linspace(-th_max, th_max, n)
        centre = c - r * nn
        dirs = np.sin(th)[:, None] * t + np.cos(th)[:, None] * nn
        pts = centre + r * dirs
        ds = np.full(n, r * (2 * th_max) / (n - 1))
        ds[[0, -1]] *= 0.5
        thick = self.thickness - r * (1 - np.cos(th))
        return pts, dirs, ds, np.maximum(thick, PAD_MIN_THICKNESS)

    def outline(self, n: int = 24) -> np.ndarray:
        """Closed outline of the pad block (face plus a flat back), CCW."""
        pts, _, _, _ = self.face_samples(n)
        back_c = self.centre - self.thickness * self.normal
        t = self.tangent
        half = self.length / 2
        back = np.array([back_c + half * t, back_c - half * t])
        return np.vstack([pts, back])

    def rotated(self, psi: float) -> "Pad":
        """The pad turned by psi (rad) about its pivot (self-aligning pads)."""
        if self.pivot_depth is None or psi == 0.0:
            return self
        pv = self.pivot
        rel = self.centre - pv
        cs, sn = math.cos(psi), math.sin(psi)
        new_c = pv + np.array([cs * rel[0] - sn * rel[1], sn * rel[0] + cs * rel[1]])
        return replace(self, face_u=float(new_c[0]), face_v=float(new_c[1]), angle=self.angle + psi)


@dataclass(frozen=True)
class Finger:
    """Geometry of the right finger; the left finger mirrors it."""

    design: GripperDesign
    body: tuple[BodyPiece, ...]
    pads: tuple[Pad, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)
    # Body outline after the keep-out cutout (finger-local polygons, for drawing),
    # the least finger material left beside the cutout (mm), and the least gap
    # between the finger and the keep-out's own radius over the stroke (mm,
    # negative if they overlap) with the opening where it occurs (deg, measured
    # from where the fingers meet).
    cut_body: tuple[np.ndarray, ...] = field(default_factory=tuple)
    ligament: float = math.inf
    hub_clearance: float = math.inf
    hub_clearance_phi: float = math.nan
    # Least gap to the brackets over the stroke (mm), where (deg) and which bracket.
    bracket_clearance: float = math.inf
    bracket_clearance_phi: float = math.nan
    bracket_name: str = ""
    # Jaw angle (rad) where the fingers meet, and the full stroke from there.
    phi_meet: float = 0.0
    phi_max: float = math.radians(90.0)

    @property
    def pivot_x(self) -> float:
        return self.design.pivot_spacing / 2

    @property
    def feasible(self) -> bool:
        return not any(i.startswith("infeasible") for i in self.issues)

    def pad_for(self, kind: str) -> Pad:
        for pad in self.pads:
            if pad.role in (kind, "all"):
                return pad
        return self.pads[0]


def to_view(pts: np.ndarray, phi: float, pivot_x: float, side: int = 1) -> np.ndarray:
    """Finger-local points (n, 2) to gripper-plane (x, z), mm."""
    pts = np.atleast_2d(pts)
    u, v = pts[:, 0], pts[:, 1]
    s, c = math.sin(phi), math.cos(phi)
    x = pivot_x + u * s - v * c
    z = -u * c - v * s
    return np.stack([side * x, z], axis=1)


def dir_to_view(d: np.ndarray, phi: float, side: int = 1) -> np.ndarray:
    d = np.atleast_2d(d)
    u, v = d[:, 0], d[:, 1]
    s, c = math.sin(phi), math.cos(phi)
    return np.stack([side * (u * s - v * c), -u * c - v * s], axis=1)


def _mount_polygon(pad: Pad, finger_face_v: float, extra_angle: float = 0.0) -> np.ndarray:
    """Rigid block between the finger's inner face and the back of the pad."""
    t, n = pad.tangent, pad.normal
    back_c = pad.centre - pad.thickness * n
    half = pad.length / 2
    if pad.pivot_depth is not None:
        # A clevis under the pivot pin, shorter than the pad so the pad can rock.
        pv = pad.pivot
        w = max(0.3 * pad.length, 6.0)
        top = pv[1] + 2.0
        return np.array(
            [[pv[0] - w / 2, finger_face_v - 1.0], [pv[0] + w / 2, finger_face_v - 1.0], [pv[0] + w / 2, top], [pv[0] - w / 2, top]]
        )
    b_prox, b_dist = back_c - half * t, back_c + half * t
    base = finger_face_v - 1.0
    return np.array([[b_prox[0], base], [b_dist[0], base], b_dist, b_prox])


def build_finger(d: GripperDesign, keep_out: KeepOut | None = None, closed_limit: float = math.radians(-25.0)) -> Finger:
    """Build the right finger for a design, with the cutout a keep-out region
    needs. Geometric conflicts are reported in `issues`; entries starting with
    'infeasible' make the design unusable."""
    f = _build_finger(d)
    meet = closed_angle(f, closed_limit)
    f = replace(f, phi_meet=meet, phi_max=meet + math.radians(d.max_opening))
    if keep_out is None or not keep_out.circles:
        return f
    return apply_keep_out(f, keep_out, closed_limit)


def _build_finger(d: GripperDesign) -> Finger:
    issues: list[str] = []
    L, h, c = d.finger_reach, d.finger_thickness, max(d.crank_offset, 0.0)
    uc = d.crank_position * L
    body: list[BodyPiece] = []

    if c > 1e-6:
        dcr = max(c, 0.6 * h)
        u_d = uc + dcr
        body.append(BodyPiece("proximal", rect_polygon((uc - h / 2) / 2, 0.0, 0.0, (uc + h / 2) / 2, h / 2), ((uc - h / 2) / 2, 0.0, 0.0, (uc + h / 2) / 2, h / 2)))
        ang = math.atan2(c, dcr)
        hl = math.hypot(dcr, c) / 2 + h / 4
        r = (uc + dcr / 2, c / 2, ang, hl, h / 2)
        body.append(BodyPiece("crank", rect_polygon(*r), r))
        r = ((u_d + L) / 2, c, 0.0, (L - u_d) / 2 + h / 4, h / 2)
        body.append(BodyPiece("distal", rect_polygon(*r), r))
    else:
        u_d = 0.0
        r = ((L - h / 2) / 2, 0.0, 0.0, (L + h / 2) / 2, h / 2)
        body.append(BodyPiece("beam", rect_polygon(*r), r))
    v_f = c + h / 2  # inner face of the distal section

    pads: list[Pad] = []
    t_p = d.pad_thickness

    def place(name: str, role: str, u_end: float, length: float, angle: float, extra: float = 0.0, protrude_from: float | None = None,
              crown: float = math.inf, pivot: dict | None = None) -> Pad:
        a_eff = abs(angle) + extra
        face_v_min = v_f + MOUNT_MIN + t_p * math.cos(a_eff) + (length / 2) * math.sin(a_eff)
        face_v = face_v_min
        if protrude_from is not None:
            face_v = protrude_from
            if face_v < face_v_min - 1e-9:
                issues.append(f"{name}: protrusion clamped to fit the mount")
                face_v = face_v_min
        face_u = u_end - (length / 2) * math.cos(angle)
        return Pad(name, role, face_u, face_v, angle, length, t_p, d.pad_width, d.pad_shore, crown, **(pivot or {}))

    alpha = math.radians(d.pad_angle)
    u_end = L - d.pad_tip_margin
    if d.pad_style in ("flat", "pivoting", "crowned"):
        crown = math.inf
        if d.pad_style == "crowned" and d.pad_crown_curvature > 1e-9:
            crown = 1000.0 / d.pad_crown_curvature
            half = d.pad_length / 2
            smax = t_p - PAD_MIN_THICKNESS
            if smax <= 0.05:
                issues.append("infeasible: pad too thin to crown")
            else:
                r_min = (half * half + smax * smax) / (2 * smax)
                if crown < r_min:
                    issues.append(f"crown radius clamped {crown:.0f} -> {r_min:.0f} mm by pad thickness")
                    crown = r_min
        pivot = None
        extra = 0.0
        if d.pad_style == "pivoting":
            depth = d.pivot_depth
            if depth < t_p + 1.0:
                issues.append(f"pivot depth raised {depth:.1f} -> {t_p + 1.0:.1f} mm to sit behind the elastomer")
                depth = t_p + 1.0
            pivot = {"pivot_depth": depth, "pivot_stiffness": d.pivot_stiffness, "pivot_range": math.radians(d.pivot_range)}
            extra = math.radians(d.pivot_range)
        pads.append(place("pad", "all", u_end, d.pad_length, alpha, extra=extra, crown=crown, pivot=pivot))
    elif d.pad_style == "paddle":
        # Paddle, then a recessed notch of tab_gap, then a narrower tab that
        # continues the same face to the fingertip, all on one mount sized for
        # the whole face line.
        has_tab = d.tab_width > 1e-6 and d.pad_tip_margin > d.tab_gap + 2.0
        cos_a = max(math.cos(alpha), 1e-6)
        span = d.pad_length + ((L - u_end) / cos_a if has_tab else 0.0)
        line = place("paddle", "all", L if has_tab else u_end, span, alpha)
        p0 = np.array([line.face_u, line.face_v]) - (span / 2) * line.tangent
        pc = p0 + (d.pad_length / 2) * line.tangent
        pads.append(replace(line, face_u=float(pc[0]), face_v=float(pc[1]), length=d.pad_length))
        if has_tab:
            s0 = d.pad_length + d.tab_gap
            tc = p0 + (s0 + (span - s0) / 2) * line.tangent
            pads.append(Pad("tab", "all", float(tc[0]), float(tc[1]), alpha, span - s0, t_p, d.tab_width, d.pad_shore))
    elif d.pad_style == "dual":
        tip = place("tip", "sheet", u_end, d.pad_length, alpha)
        pads.append(tip)
        alpha2 = math.radians(d.pad2_angle)
        u2c = L - d.pad2_offset
        u2_end = u2c + (d.pad2_length / 2) * math.cos(alpha2)
        box = place("box", "box", u2_end, d.pad2_length, alpha2, protrude_from=tip.face_v + d.pad2_protrusion)
        tip_prox = tip.face_u - (tip.length / 2) * math.cos(tip.angle)
        if u2_end > tip_prox - 2.0:
            issues.append("infeasible: box pad overlaps tip pad")
        pads.append(box)
    else:
        raise ValueError(f"unknown pad style {d.pad_style}")

    for pad in pads:
        prox = pad.face_u - (pad.length / 2) * math.cos(pad.angle)
        if prox < u_d + 1.0:
            issues.append(f"infeasible: {pad.name} pad overlaps the crank or pivot hub")
        if pad.face_u + (pad.length / 2) * math.cos(pad.angle) > L + 1e-6:
            issues.append(f"infeasible: {pad.name} pad runs past the fingertip")
        body.append(BodyPiece(f"mount_{pad.name}", _mount_polygon(pad, v_f)))
    return Finger(d, tuple(body), tuple(pads), tuple(issues))


def finger_outline_view(f: Finger, phi: float, side: int = 1) -> dict[str, list[np.ndarray]]:
    """Polygons of one finger in the gripper plane, for drawing."""
    body = f.cut_body if f.cut_body else tuple(p.poly for p in f.body)
    out = {"body": [to_view(P, phi, f.pivot_x, side) for P in body], "pads": [to_view(p.outline(), phi, f.pivot_x, side) for p in f.pads]}
    return out


def to_local(pts: np.ndarray, phi: float, pivot_x: float) -> np.ndarray:
    """Gripper-plane (x, z) points to the right finger's local (u, v), mm."""
    pts = np.atleast_2d(pts)
    dx, z = pts[:, 0] - pivot_x, pts[:, 1]
    s, c = math.sin(phi), math.cos(phi)
    return np.stack([dx * s - z * c, -dx * c - z * s], axis=1)


def closed_angle(f: Finger, lower: float) -> float:
    """Opening angle at which the closing fingers meet on the centreline (or `lower`)."""
    polys = [p.outline() for p in f.pads] + [b.poly for b in f.body]

    def min_x(phi: float) -> float:
        return min(float(to_view(P, phi, f.pivot_x)[:, 0].min()) for P in polys)

    hi = math.radians(60.0)
    if min_x(lower) >= 0.0:
        return lower
    if min_x(hi) <= 0.0:
        return hi
    lo = lower
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if min_x(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def swept_keep_out(f: Finger, ko: KeepOut, phi_lo: float, phi_hi: float, n: int = 48):
    """Keep-out region seen from the right finger over its stroke (finger-local shapely geometry).
    The left finger is the mirror image, so mirrored circles are included."""
    phis = np.linspace(phi_lo, phi_hi, n)
    parts = []
    for x, z, r in ko.radii():
        for xm in {x, -x}:
            pts = np.vstack([to_local(np.array([[xm, z]]), ph, f.pivot_x) for ph in phis])
            parts.append(LineString(pts).buffer(r, quad_segs=64))
    return unary_union(parts)


def apply_keep_out(f: Finger, ko: KeepOut, closed_limit: float) -> Finger:
    """Cut the finger body clear of the keep-out over the full stroke. Pads and
    pad mounts may not be cut; the beam must keep `min_ligament` of material."""
    issues = list(f.issues)
    phi_lo, phi_hi = f.phi_meet, f.phi_max
    swept = swept_keep_out(f, ko, phi_lo, phi_hi)
    for pad in f.pads:
        if SPolygon(pad.outline()).intersects(swept):
            issues.append(f"infeasible: the {pad.name} pad enters the bracket keep-out during the stroke")
    rects = [p for p in f.body if p.rect is not None]
    for p in f.body:
        if p.rect is None and SPolygon(p.poly).intersects(swept):
            issues.append(f"infeasible: the {p.name} block enters the bracket keep-out during the stroke")
    body_union = unary_union([SPolygon(p.poly) for p in rects])
    ligament = math.inf
    for p in rects:
        cu, cv, ang, hl, ht = p.rect
        t = np.array([math.cos(ang), math.sin(ang)])
        nrm = np.array([-math.sin(ang), math.cos(ang)])
        for s in np.linspace(-hl, hl, max(int(2 * hl), 8)):
            c = np.array([cu, cv]) + s * t
            seg = LineString([c - 1.5 * ht * nrm, c + 1.5 * ht * nrm])
            if not seg.intersects(swept):
                continue
            material = seg.intersection(body_union)
            if material.is_empty:
                continue
            ligament = min(ligament, material.difference(swept).length)
    if ligament < ko.min_ligament:
        issues.append(f"infeasible: the cutout around the bracket leaves {ligament:.1f} mm of finger (minimum {ko.min_ligament:.1f} mm)")
    cut = []
    for p in f.body:
        g = SPolygon(p.poly).difference(swept) if p.rect is not None else SPolygon(p.poly)
        for poly in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            if not poly.is_empty and poly.area > 1e-6:
                cut.append(np.asarray(poly.exterior.coords)[:-1])
    solid = unary_union([SPolygon(P) for P in cut] + [SPolygon(pad.outline()) for pad in f.pads])
    clearance, phi_min = hub_clearance(f, ko, solid, phi_lo, phi_hi)
    b_gap, b_phi, b_name = math.inf, math.nan, ""
    if ko.brackets:
        b_gap, b_phi, b_name = bracket_clearance(f, ko, phi_lo, phi_hi, swept)
        if b_gap < 0.0:
            issues.append(f"infeasible: the finger meets the {b_name} at {b_phi:.0f} deg of opening")
    return replace(f, issues=tuple(issues), cut_body=tuple(cut), ligament=ligament,
                   hub_clearance=clearance, hub_clearance_phi=phi_min,
                   bracket_clearance=b_gap, bracket_clearance_phi=b_phi, bracket_name=b_name)


def bracket_height_solid(f: Finger, ko: KeepOut, swept=None):
    """The parts of the finger at the brackets' heights (finger-local shapely geometry): the
    beam beyond the root (less the keep-out cutout), every pad and pad mount, less the hinge disc."""
    beyond_root = SPolygon([(ko.root_length, -1e4), (1e4, -1e4), (1e4, 1e4), (ko.root_length, 1e4)])
    parts = []
    for p in f.body:
        g = SPolygon(p.poly)
        if p.rect is not None:
            g = g.intersection(beyond_root)
            if swept is not None:
                g = g.difference(swept)
        parts.append(g)
    parts += [SPolygon(pad.outline()) for pad in f.pads]
    return unary_union(parts).difference(Point(0.0, 0.0).buffer(ko.hinge_radius, quad_segs=32))


def bracket_clearance(f: Finger, ko: KeepOut, phi_lo: float, phi_hi: float, swept=None, n: int = 91) -> tuple[float, float, str]:
    """Least gap (mm) between the finger and the brackets over the stroke, the opening
    (deg, from where the fingers meet) where it occurs, and the bracket's name.
    A negative gap means they overlap."""
    solid = bracket_height_solid(f, ko, swept)
    if solid.is_empty:
        return math.inf, math.nan, ""
    best = (math.inf, math.nan, "")
    for ph in np.linspace(phi_lo, phi_hi, n):
        for br in ko.brackets:
            gap = _region_gap(solid, br.region, ph, f.pivot_x)
            if gap < best[0]:
                best = (gap, math.degrees(ph - phi_lo), br.name)
    return best


def _region_gap(solid, region, phi: float, pivot_x: float) -> float:
    """Gap (mm) from the finger-local `solid` to a gripper-frame region at opening `phi`;
    an overlap counts as minus the square root of its area. Holes in the region are ignored."""
    gap = math.inf
    for geom in getattr(region, "geoms", [region]):
        local = SPolygon(to_local(np.asarray(geom.exterior.coords), phi, pivot_x))
        g = -math.sqrt(solid.intersection(local).area) if solid.intersects(local) else solid.distance(local)
        gap = min(gap, float(g))
    return gap


def clearance_profile(f: Finger, ko: KeepOut, openings: np.ndarray) -> dict[str, np.ndarray]:
    """Gap (mm) to the keep-out circles ("hub") and to each bracket at each opening (rad,
    from where the fingers meet), for a finger from build_finger."""
    phis = f.phi_meet + np.asarray(openings, dtype=float)
    body = f.cut_body or tuple(p.poly for p in f.body)
    solid = unary_union([SPolygon(P) for P in body] + [SPolygon(p.outline()) for p in f.pads])
    out = {"hub": np.array([min(solid.distance(Point(to_local(np.array([[xm, z]]), ph, f.pivot_x)[0])) - r
                                for x, z, r in ko.circles for xm in {x, -x}) for ph in phis])}
    if ko.brackets:
        high = bracket_height_solid(f, ko, swept_keep_out(f, ko, f.phi_meet, f.phi_max))
        for br in ko.brackets:
            out[br.name] = np.array([_region_gap(high, br.region, ph, f.pivot_x) for ph in phis])
    return out


def hub_clearance(f: Finger, ko: KeepOut, solid, phi_lo: float, phi_hi: float, n: int = 145) -> tuple[float, float]:
    """Least gap (mm) between the finger solid (finger-local shapely geometry) and the
    keep-out circles at their own radius, over the stroke, and the opening (deg, from
    phi_lo, where the fingers meet) where it occurs."""
    best, best_phi = math.inf, math.nan
    for ph in np.linspace(phi_lo, phi_hi, n):
        for x, z, r in ko.circles:
            for xm in {x, -x}:
                c = to_local(np.array([[xm, z]]), ph, f.pivot_x)[0]
                gap = solid.distance(Point(c)) - r
                if gap < best:
                    best, best_phi = gap, math.degrees(ph - phi_lo)
    return float(best), float(best_phi)
