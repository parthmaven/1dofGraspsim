"""Level 0: quasi-static analytic screen of one design on one object.

1. Fit: with the jaw at its maximum opening, the fingers clear the object at
   the presented insertion depth.
2. Touch: close until a finger first touches the object face. If the first
   touch is on the finger body, or the fingertips meet above a thin sheet, the
   grasp fails.
3. Squeeze: keep closing until the drive torque is balanced by the pad
   pressure. The pad and the object surface are springs in series on a
   Winkler foundation (lit review 3.1), sampled along the pad face, so pad
   angle, crown, partial overlap with the box top and the pad's own pivot
   (self-aligning pads) all fall out of one calculation.
4. Margins against the lift and shake loads: friction hold (including the
   jaw-opening moment of the friction force and squeeze-out at a box corner),
   pitch moment, peak pressure, unsupported-panel force, sheet creasing and
   pad rotation. Every margin must be >= 1 for a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import Finger, Pad, build_finger, to_view
from .objects import G, GraspObject
from .pads import foundation_stiffness, series
from .scenario import Scenario
from .schema import GripperDesign

N_FACE = 41
CORNER_ZONE = 2.0  # mm below the box top treated as the corner contact
FIT_CLEARANCE = 1.0  # mm
TABLE_CLEARANCE = 2.0  # mm
BODY_TOLERANCE = 0.5  # mm of finger-body interference ignored (surface texture, box bulge)
MAX_PAD_STRAIN = 0.8


@dataclass
class GripResult:
    design: GripperDesign
    object_id: str
    ok: bool
    reason: str = ""
    margins: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.ok or not self.margins:
            return 0.0
        return float(min(self.margins.values()))

    @property
    def passed(self) -> bool:
        return self.ok and self.score >= 1.0

    @property
    def limiting(self) -> str:
        if not self.ok:
            return self.reason
        if not self.margins:
            return ""
        return min(self.margins, key=self.margins.get)


def _fail(design: GripperDesign, obj: GraspObject, reason: str, **info: Any) -> GripResult:
    return GripResult(design, obj.id, ok=False, reason=reason, info=info)


def band_min_x(polys: list[np.ndarray], z_lo: float, z_hi: float) -> float:
    """Smallest x any polygon reaches inside the band z_lo <= z <= z_hi."""
    best = math.inf
    for P in polys:
        x, z = P[:, 0], P[:, 1]
        inside = (z >= z_lo) & (z <= z_hi)
        if inside.any():
            best = min(best, float(x[inside].min()))
        x2, z2 = np.roll(x, -1), np.roll(z, -1)
        for zl in (z_lo, z_hi):
            if not math.isfinite(zl):
                continue
            cross = (z - zl) * (z2 - zl) < 0
            if cross.any():
                t = (zl - z[cross]) / (z2[cross] - z[cross])
                best = min(best, float((x[cross] + t * (x2[cross] - x[cross])).min()))
    return best


def registered_depth(f: Finger, obj: GraspObject, sc: Scenario) -> float:
    """Insertion (m) that puts the object top `clearance` above the proximal end of its pad."""
    pres = sc.for_kind(obj.kind)
    pad = f.pad_for(obj.kind)
    u_prox = pad.face_u - (pad.length / 2) * math.cos(pad.angle)
    return (f.design.finger_reach - u_prox) / 1000.0 + pres.clearance


def _depth_limits(obj: GraspObject, sc: Scenario) -> float:
    pres = sc.for_kind(obj.kind)
    hi = pres.max_insertion
    if obj.kind == "box" and pres.fixture == "table":
        hi = min(hi, obj.height - 2 * TABLE_CLEARANCE / 1000.0)
    return hi


def candidate_depths(f: Finger, obj: GraspObject, sc: Scenario) -> list[float]:
    """Insertion depths (m) tried for this design and object."""
    pres = sc.for_kind(obj.kind)
    hi = _depth_limits(obj, sc)
    if pres.insertion == "fixed":
        return [max(min(pres.insertion_depth, hi), 0.005)]
    reg = max(min(registered_depth(f, obj, sc), hi), 0.005)
    if pres.insertion == "pad_registered" or hi - reg < 0.002:
        return [reg]
    n = max(int(sc.raw.get("presentation", {}).get(obj.kind, {}).get("insertion_scan", 5)), 2)
    return [float(x) for x in np.linspace(reg, hi, n)]


def finger_polys(f: Finger, phi: float, pads: tuple[Pad, ...] | None = None, body: bool = True) -> list[np.ndarray]:
    pads = f.pads if pads is None else pads
    out = [to_view(p.outline(), phi, f.pivot_x) for p in pads]
    if body:
        out += [to_view(b.poly, phi, f.pivot_x) for b in f.body]
    return out


@dataclass
class _PadModel:
    pad: Pad
    pts: np.ndarray  # local mm
    ds: np.ndarray  # m
    k: np.ndarray  # Pa/m per sample, pad and object in series
    k_pad: np.ndarray  # Pa/m per sample, pad alone
    thick: np.ndarray  # mm

    def rotated(self, theta_view: float) -> "_PadModel":
        """Rotate a pivoting pad by theta (rad, CCW in the gripper plane)."""
        if self.pad.pivot_depth is None or theta_view == 0.0:
            return self
        psi = -theta_view  # the finger-local frame is mirrored relative to the plane
        pv = self.pad.pivot
        cs, sn = math.cos(psi), math.sin(psi)
        rel = self.pts - pv
        pts = pv + np.stack([cs * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + cs * rel[:, 1]], axis=1)
        return _PadModel(self.pad.rotated(psi), pts, self.ds, self.k, self.k_pad, self.thick)


def _pad_models(f: Finger, k_obj: float) -> list[_PadModel]:
    out = []
    for pad in f.pads:
        pts, _, ds, thick = pad.face_samples(N_FACE)
        lut = {t: foundation_stiffness(pad.shore, pad.length * 1e-3, pad.width * 1e-3, t * 1e-3) for t in np.unique(thick)}
        k_pad = np.array([lut[t] for t in thick])
        k = np.array([series(kp, k_obj) for kp in k_pad])
        out.append(_PadModel(pad, pts, ds * 1e-3, k, k_pad, thick))
    return out


@dataclass
class _Load:
    phi: float
    models: list[_PadModel]
    z: np.ndarray  # mm
    p: np.ndarray  # Pa
    fx: np.ndarray  # N, force on the pad (+x pushes the right finger open)
    owner: np.ndarray  # pad index per sample
    pad_strain: np.ndarray

    @property
    def N(self) -> float:
        return float(self.fx.sum())

    def torque(self) -> float:
        return float(np.sum(-self.z * 1e-3 * self.fx))


def _load_at(f: Finger, models: list[_PadModel], phi: float, x_face: float, z_lo: float, z_hi: float, b: float) -> _Load:
    zs, ps, fs, own, st = [], [], [], [], []
    for i, m in enumerate(models):
        v = to_view(m.pts, phi, f.pivot_x)
        inside = (v[:, 1] >= z_lo) & (v[:, 1] <= z_hi)
        delta = np.where(inside, np.maximum(x_face - v[:, 0], 0.0), 0.0) * 1e-3
        p = m.k * delta
        zs.append(v[:, 1])
        ps.append(p)
        fs.append(p * b * m.ds)
        own.append(np.full(len(p), i))
        st.append(p / m.k_pad / (m.thick * 1e-3))
    cat = np.concatenate
    return _Load(phi, models, cat(zs), cat(ps), cat(fs), cat(own), cat(st))


def _align_pivoting(f: Finger, base: list[_PadModel], phi: float, x_face: float, z_lo: float, z_hi: float, b: float) -> _Load:
    """Self-aligning pads turn about their pin until contact moment and spring balance."""
    idx = [i for i, m in enumerate(base) if m.pad.pivot_depth is not None]
    if not idx:
        return _load_at(f, base, phi, x_face, z_lo, z_hi, b)
    i = idx[0]
    pad = base[i].pad
    pv = to_view(pad.pivot[None, :], phi, f.pivot_x)[0]
    rng, k = pad.pivot_range, max(pad.pivot_stiffness, 1e-6)

    def state(theta: float) -> tuple[float, _Load]:
        models = list(base)
        models[i] = base[i].rotated(theta)
        ld = _load_at(f, models, phi, x_face, z_lo, z_hi, b)
        sel = ld.owner == i
        m_contact = float(np.sum(-(ld.z[sel] - pv[1]) * 1e-3 * ld.fx[sel]))
        return m_contact - k * theta, ld

    g_lo, ld_lo = state(-rng)
    if g_lo <= 0:
        return ld_lo
    g_hi, ld_hi = state(rng)
    if g_hi >= 0:
        return ld_hi
    lo, hi = -rng, rng
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        g, _ = state(mid)
        if g > 0:
            lo = mid
        else:
            hi = mid
    return state(0.5 * (lo + hi))[1]


def _pad_rotation(ld: _Load) -> float:
    for m in ld.models:
        if m.pad.pivot_depth is not None:
            return m.pad.angle
    return 0.0


def solve_grip(design: GripperDesign, obj: GraspObject, sc: Scenario, torque: float | None = None,
               finger: Finger | None = None, depth: float | None = None) -> GripResult:
    """Best grasp over the scenario's candidate insertion depths (or at `depth`)."""
    f = finger or build_finger(design)
    if not f.feasible:
        return _fail(design, obj, "geometry: " + "; ".join(i for i in f.issues if i.startswith("infeasible")))
    depths = [depth] if depth is not None else candidate_depths(f, obj, sc)
    best: GripResult | None = None
    for dp in depths:
        r = _solve_at_depth(f, obj, sc, dp, design.finger_torque if torque is None else torque)
        if best is None or _rank(r) > _rank(best):
            best = r
    return best


def _rank(r: GripResult) -> tuple[float, float]:
    if not r.ok:
        return (-1.0, 0.0)
    return (min(r.score, 1.0), sum(min(v, 3.0) for v in r.margins.values()))


def _solve_at_depth(f: Finger, obj: GraspObject, sc: Scenario, depth: float, tau: float) -> GripResult:
    d = f.design
    pres = sc.for_kind(obj.kind)
    L = d.finger_reach
    x_face = obj.thickness * 1000.0 / 2
    z_top = -L + depth * 1000.0
    z_bot = z_top - obj.height * 1000.0
    b = d.pad_width * 1e-3
    phi_open, phi_lo = math.radians(d.max_opening), sc.closed_limit
    info0 = {"insertion_mm": depth * 1000.0}

    if z_top >= -d.finger_thickness / 2:
        return _fail(d, obj, "insertion depth puts the object above the finger pivots", **info0)
    if band_min_x(finger_polys(f, phi_open), z_bot, z_top) < x_face + FIT_CLEARANCE:
        return _fail(d, obj, "does not fit between the open fingers", **info0)

    def touching(phi: float) -> bool:
        return band_min_x(finger_polys(f, phi), z_bot, z_top) <= x_face

    if not touching(phi_lo):
        return _fail(d, obj, "fingers close fully without touching the object", **info0)
    lo, hi = phi_lo, phi_open
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if touching(mid):
            lo = mid
        else:
            hi = mid
    phi_touch = hi
    pads_touch = band_min_x(finger_polys(f, lo, body=False), z_bot, z_top) <= x_face
    body_touch = band_min_x([to_view(bp.poly, lo, f.pivot_x) for bp in f.body], z_bot, z_top) <= x_face - BODY_TOLERANCE
    if body_touch or not pads_touch:
        return _fail(d, obj, "finger body touches the object before the pads", phi_touch=math.degrees(phi_touch), **info0)
    if band_min_x(finger_polys(f, phi_touch), z_top + 1e-6, math.inf) <= 0.0:
        return _fail(d, obj, "fingertips meet above the object before gripping it", phi_touch=math.degrees(phi_touch), **info0)

    models = _pad_models(f, obj.material.k_surface)
    pivoting = any(m.pad.pivot_depth is not None for m in models)

    def load(phi: float) -> _Load:
        if pivoting:
            return _align_pivoting(f, models, phi, x_face, z_bot, z_top, b)
        return _load_at(f, models, phi, x_face, z_bot, z_top, b)

    lo, hi = phi_touch - math.radians(25.0), phi_touch
    if load(lo).torque() < tau:
        return _fail(d, obj, "pads bottom out before the drive torque is reached", **info0)
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if load(mid).torque() > tau:
            lo = mid
        else:
            hi = mid
    ld = load(0.5 * (lo + hi))
    phi = ld.phi
    N = ld.N
    if N <= 1e-9:
        return _fail(d, obj, "no pad force developed", **info0)

    body_x = band_min_x([to_view(bp.poly, phi, f.pivot_x) for bp in f.body], z_bot, z_top)
    if body_x <= x_face - BODY_TOLERANCE:
        return _fail(d, obj, "finger body reaches the object while squeezing", **info0)
    if float(ld.pad_strain.max()) > MAX_PAD_STRAIN:
        return _fail(d, obj, f"pad compressed beyond {MAX_PAD_STRAIN:.0%} of its thickness", **info0)

    if obj.kind == "box" and pres.fixture == "table":
        polys_all = [b_.poly for b_ in f.body] + [p.outline() for p in f.pads]
        zmin = math.inf
        for ph in (phi, phi_open, math.atan2(max(d.crank_offset, 0.0) + d.finger_thickness / 2, L)):
            if phi <= ph <= phi_open:
                zmin = min(zmin, min(float(to_view(P, ph, f.pivot_x)[:, 1].min()) for P in polys_all))
        if zmin < z_bot + TABLE_CLEARANCE:
            return _fail(d, obj, "fingertips hit the table", **info0)

    loaded = ld.fx > 0
    zc = ld.z[loaded]
    p_max = float(ld.p.max())
    cop_z = float(np.sum(ld.z * ld.fx) / N)  # mm
    contact_len = max(float(zc.max() - zc.min()), 0.5) * 1e-3  # m
    arm = tau / N
    used_pads = tuple(ld.models[i].pad for i in sorted(set(ld.owner[loaded].tolist())))

    # Box-corner contact: the pressure peaks at the top edge of the box and the
    # pad face there is tilted, so the reaction follows the pad normal and
    # pushes the object out of the jaw (lit review 4.4).
    f_sq = 0.0
    if obj.kind == "box":
        corner = loaded & (ld.z >= z_top - CORNER_ZONE)
        if corner.any():
            top_i = int(np.argmax(np.where(loaded, ld.z, -np.inf)))
            if ld.p[top_i] >= 0.6 * p_max:
                pad = ld.models[ld.owner[top_i]].pad
                tv = to_view(np.array([[0.0, 0.0], pad.tangent]), phi, 0.0)
                tv = tv[1] - tv[0]
                f_sq = float(ld.fx[corner].sum()) * abs(tv[0]) / max(abs(tv[1]), 1e-9)

    m, mu = obj.mass, obj.material.friction * sc.friction_scale
    ax, az = sc.lateral_accel, sc.vertical_accel
    px = d.pivot_spacing / 2 * 1e-3
    r_t = x_face * 1e-3 - px  # > 0: contact outboard of the pivot, friction closes the jaw

    t_dem = m * (G + az) / 2 + f_sq
    denom = arm - mu * r_t
    t_cap = math.inf if denom <= 1e-9 else mu * tau / denom
    margins: dict[str, float] = {"hold": t_cap / (sc.friction_sf * t_dem)}
    n_dem = (tau + r_t * min(t_dem, t_cap)) / arm

    z_com = z_top - obj.com_below_top * 1000.0
    h = abs(cop_z - z_com) * 1e-3
    m_dem = m * ax * h
    pivot_pad = next((p for p in used_pads if p.pivot_depth is not None), None)
    e_max = contact_len / 3
    rot = 0.0
    m_fric = max(0.0, 2 * mu * n_dem - m * (G + az)) * x_face * 1e-3
    if pivot_pad is not None:
        k_pp, rng = max(pivot_pad.pivot_stiffness, 1e-6), pivot_pad.pivot_range
        used = abs(_pad_rotation(ld) - math.radians(d.pad_angle))
        spare = max(rng - used, 0.0)
        theta_allow = min(spare, sc.max_rotation)
        m_norm = 2 * k_pp * theta_allow + (2 * n_dem * e_max if spare <= sc.max_rotation else 0.0)
        if m_dem > m_fric:
            rot = min(spare, (m_dem - m_fric) / (2 * k_pp))
    else:
        m_norm = 2 * n_dem * e_max
    m_cap = m_norm + m_fric
    margins["pitch"] = m_cap / (sc.pitch_sf * m_dem) if m_dem > 1e-12 else math.inf
    if pivot_pad is not None:
        margins["rotation"] = sc.max_rotation / rot if rot > 1e-9 else math.inf

    share = m_norm / m_cap if m_cap > 0 else 1.0
    m_pad = min(share * m_dem, m_cap) / 2
    p_dem = p_max * n_dem / N + 6 * m_pad / (b * contact_len**2)
    margins["pressure"] = obj.material.allow_pressure / (sc.pressure_sf * p_dem)

    f_panel = 0.0
    if obj.kind == "box" and math.isfinite(obj.unsupported_force):
        fill_line = z_top - obj.headspace * 1000.0
        f_panel = float(ld.fx[ld.z > fill_line].sum()) * n_dem / N
        if f_panel > 1e-9:
            margins["panel"] = obj.unsupported_force / f_panel
    if obj.kind == "sheet":
        b_eff = b if sc.crease_width == "pad" else obj.width
        cap = obj.material.bending_strength * b_eff
        margins["crease"] = cap / (sc.crease_safety_factor * m_dem) if m_dem > 1e-12 else math.inf

    info = {
        **info0,
        "phi_touch_deg": math.degrees(phi_touch),
        "phi_deg": math.degrees(phi),
        "normal_force_N": N,
        "normal_force_loaded_N": n_dem,
        "peak_pressure_kPa": p_max / 1e3,
        "loaded_pressure_kPa": p_dem / 1e3,
        "contact_length_mm": contact_len * 1e3,
        "cop_below_top_mm": z_top - cop_z,
        "panel_force_N": f_panel,
        "arm_mm": arm * 1e3,
        "squeeze_out_N": f_sq,
        "pitch_demand_Nm": m_dem,
        "pitch_capacity_Nm": m_cap,
        "pads": ",".join(p.name for p in used_pads),
        "pad_rotation_deg": math.degrees(rot),
        "max_pad_strain": float(ld.pad_strain.max()),
    }
    return GripResult(d, obj.id, ok=True, margins=margins, info=info)


def screen_design(design: GripperDesign, objects: list[GraspObject], sc: Scenario) -> list[GripResult]:
    f = build_finger(design)
    return [solve_grip(design, o, sc, finger=f) for o in objects]
