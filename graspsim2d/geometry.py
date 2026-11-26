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

import numpy as np

from .schema import GripperDesign

MOUNT_MIN = 1.0  # mm of rigid mount left under the thinnest point of a pad
PAD_MIN_THICKNESS = 1.0  # mm of elastomer left at the ends of a crowned pad


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


def build_finger(d: GripperDesign) -> Finger:
    """Build the right finger for a design. Geometric conflicts are reported in
    `issues`; entries starting with 'infeasible' make the design unusable."""
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
    out = {"body": [to_view(p.poly, phi, f.pivot_x, side) for p in f.body], "pads": [to_view(p.outline(), phi, f.pivot_x, side) for p in f.pads]}
    return out
