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
   jaw-opening moment of the friction force and the squeeze-out of a box by a
   tilted pad face),
   pitch moment, peak pressure, unsupported-panel force, sheet creasing and
   pad rotation. Every margin must be >= 1 for a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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
BODY_TOLERANCE = 0.0  # mm of finger-body interference ignored; the dynamic check treats any overlap as rigid contact
MAX_PAD_STRAIN = 0.8
TIP_BAND = 0.01  # mm above the lowest point of the finger counted as its tip


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


def tip_half_gap(f: Finger, phi: float) -> float:
    """Distance (mm) from the jaw centreline to the inside of the fingertip at jaw angle phi."""
    polys = finger_polys(f, phi)
    z_low = min(float(P[:, 1].min()) for P in polys)
    return band_min_x(polys, z_low, z_low + TIP_BAND)


def pass_opening(f: Finger, obj: GraspObject, sc: Scenario) -> float:
    """Smallest jaw angle (rad) at which the object passes between the fingertips, or nan if it never does."""
    half = obj.thickness * 1000.0 / 2
    lo, hi = sc.closed_limit, f.phi_max
    if tip_half_gap(f, hi) < half:
        return math.nan
    if tip_half_gap(f, lo) >= half:
        return lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if tip_half_gap(f, mid) >= half else (mid, hi)
    return hi


def approach_opening(f: Finger, obj: GraspObject, sc: Scenario, depth: float) -> float:
    """Smallest jaw angle (rad) at which the fingers clear the object by FIT_CLEARANCE all the
    way to `depth` (m) of insertion, i.e. how far the jaw must open to come down over it;
    nan if the stroke is too short."""
    x_need = obj.thickness * 1000.0 / 2 + FIT_CLEARANCE
    z_top = -f.design.finger_reach + depth * 1000.0
    z_bot = z_top - obj.height * 1000.0

    def clear(phi: float) -> bool:
        return band_min_x(finger_polys(f, phi), z_bot, z_top) >= x_need

    lo, hi = f.phi_meet, f.phi_max
    if not clear(hi):
        return math.nan
    if clear(lo):
        return lo
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if clear(mid) else (mid, hi)
    return hi


def _depth_limits(f: Finger, obj: GraspObject, sc: Scenario) -> float:
    pres = sc.for_kind(obj.kind)
    hi = pres.max_insertion
    if obj.kind == "box" and pres.fixture == "table":
        hi = min(hi, obj.height - 2 * TABLE_CLEARANCE / 1000.0)
    if sc.keep_out is not None:
        off, half = pres.lateral_offset * 1000.0, obj.thickness * 1000.0 / 2
        top = sc.keep_out.max_object_top(off - half, off + half)
        if math.isfinite(top):
            hi = min(hi, (f.design.finger_reach + top) / 1000.0)
    return hi


def candidate_depths(f: Finger, obj: GraspObject, sc: Scenario) -> list[float]:
    """Insertion depths (m) tried for this design and object."""
    pres = sc.for_kind(obj.kind)
    hi = _depth_limits(f, obj, sc)
    if pres.insertion == "fixed":
        return [max(min(pres.insertion_depth, hi), 0.005)]
    reg = max(min(registered_depth(f, obj, sc), hi), 0.005)
    if pres.insertion == "pad_registered":
        return [reg]
    lo = reg
    if hi - reg < 0.002:
        # The pad-registered depth is out of reach (keep-out, table, max
        # insertion): scan shallower instead, down to the object top at mid-pad.
        lo = max(min(reg, (f.design.finger_reach - f.pad_for(obj.kind).face_u) / 1000.0), 0.005)
        if hi - lo < 0.002:
            return [reg]
    n = max(int(sc.raw.get("presentation", {}).get(obj.kind, {}).get("insertion_scan", 5)), 2)
    return [float(x) for x in np.linspace(lo, hi, n)]


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
    w: float  # m, contact width: the pad's width or the object's, whichever is narrower

    def rotated(self, theta_view: float) -> "_PadModel":
        """Rotate a pivoting pad by theta (rad, CCW in the gripper plane)."""
        if self.pad.pivot_depth is None or theta_view == 0.0:
            return self
        psi = -theta_view  # the finger-local frame is mirrored relative to the plane
        pv = self.pad.pivot
        cs, sn = math.cos(psi), math.sin(psi)
        rel = self.pts - pv
        pts = pv + np.stack([cs * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + cs * rel[:, 1]], axis=1)
        return _PadModel(self.pad.rotated(psi), pts, self.ds, self.k, self.k_pad, self.thick, self.w)


def _pad_models(f: Finger, obj: GraspObject) -> list[_PadModel]:
    out = []
    for pad in f.pads:
        w = min(pad.width * 1e-3, obj.width)
        pts, _, ds, thick = pad.face_samples(N_FACE)
        lut = {t: foundation_stiffness(pad.shore, pad.length * 1e-3, w, t * 1e-3) for t in np.unique(thick)}
        k_pad = np.array([lut[t] for t in thick])
        k = np.array([series(kp, obj.material.k_surface) for kp in k_pad])
        out.append(_PadModel(pad, pts, ds * 1e-3, k, k_pad, thick, w))
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
    slope: np.ndarray  # dx/dz of the pad face per sample; < 0 where the face opens toward the fingertips

    @property
    def N(self) -> float:
        return float(self.fx.sum())

    def torque(self) -> float:
        return float(np.sum(-self.z * 1e-3 * self.fx))


def _load_at(f: Finger, models: list[_PadModel], phi: float, x_face: float, z_lo: float, z_hi: float) -> _Load:
    zs, ps, fs, own, st, sl = [], [], [], [], [], []
    for i, m in enumerate(models):
        v = to_view(m.pts, phi, f.pivot_x)
        inside = (v[:, 1] >= z_lo) & (v[:, 1] <= z_hi)
        delta = np.where(inside, np.maximum(x_face - v[:, 0], 0.0), 0.0) * 1e-3
        p = m.k * delta
        zs.append(v[:, 1])
        ps.append(p)
        fs.append(p * m.w * m.ds)
        own.append(np.full(len(p), i))
        st.append(p / m.k_pad / (m.thick * 1e-3))
        dz = np.gradient(v[:, 1])
        sl.append(np.gradient(v[:, 0]) / np.where(np.abs(dz) < 1e-9, 1e-9, dz))
    cat = np.concatenate
    return _Load(phi, models, cat(zs), cat(ps), cat(fs), cat(own), cat(st), cat(sl))


def _align_pivoting(f: Finger, base: list[_PadModel], phi: float, x_face: float, z_lo: float, z_hi: float) -> _Load:
    """Self-aligning pads turn about their pin until contact moment and spring balance."""
    idx = [i for i, m in enumerate(base) if m.pad.pivot_depth is not None]
    if not idx:
        return _load_at(f, base, phi, x_face, z_lo, z_hi)
    i = idx[0]
    pad = base[i].pad
    pv = to_view(pad.pivot[None, :], phi, f.pivot_x)[0]
    rng, k = pad.pivot_range, max(pad.pivot_stiffness, 1e-6)

    def state(theta: float) -> tuple[float, _Load]:
        models = list(base)
        models[i] = base[i].rotated(theta)
        ld = _load_at(f, models, phi, x_face, z_lo, z_hi)
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
    f = finger or build_finger(design, sc.keep_out, sc.closed_limit)
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
    phi_lo = sc.closed_limit
    info0: dict[str, float] = {"insertion_mm": depth * 1000.0}

    if z_top >= -d.finger_thickness / 2:
        return _fail(d, obj, "insertion depth puts the object above the finger pivots", **info0)
    if sc.keep_out is not None:
        off = pres.lateral_offset * 1000.0
        if z_top > sc.keep_out.max_object_top(off - x_face, off + x_face) + 1e-6:
            return _fail(d, obj, "object top would enter the bracket keep-out", **info0)
    phi_open = approach_opening(f, obj, sc, depth)
    if not math.isfinite(phi_open):
        return _fail(d, obj, "does not fit between the open fingers", **info0)
    info0["approach_opening_deg"] = math.degrees(phi_open)  # jaw angle, like phi_deg
    info0["approach_stroke_deg"] = math.degrees(phi_open - f.phi_meet)  # opening from where the fingers meet

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

    models = _pad_models(f, obj)
    pivoting = any(m.pad.pivot_depth is not None for m in models)

    def load(phi: float) -> _Load:
        if pivoting:
            return _align_pivoting(f, models, phi, x_face, z_bot, z_top)
        return _load_at(f, models, phi, x_face, z_bot, z_top)

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
    if ld.torque() <= 0.0:
        ld = load(lo)
    # The pad face is sampled at points, so on a stiff contact the torque can jump from zero to
    # several times the drive's within the last step; scale the load to the drive torque.
    s = tau / ld.torque() if ld.torque() > 0.0 else 0.0
    ld = replace(ld, p=ld.p * s, fx=ld.fx * s, pad_strain=ld.pad_strain * s)
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
    used = sorted(set(ld.owner[loaded].tolist()))
    used_pads = tuple(ld.models[i].pad for i in used)
    b = min(ld.models[i].w for i in used)  # narrowest loaded contact carries the moment

    # Squeeze-out (lit review 4.4): where the pad face is tilted to the box
    # side, the contact normal tilts by the box's share of the overlap. At the
    # top edge, which meets the pad face edge-on, that share is all of it; on
    # the side below it is the pad-and-box series split. A face that opens
    # toward the fingertips pushes the box out of the jaw; one that closes
    # toward them gets no credit for wedging it in.
    f_sq, tilt = 0.0, math.nan
    if obj.kind == "box":
        k_pad = np.concatenate([m.k_pad for m in ld.models])
        share = np.where(ld.z >= z_top - CORNER_ZONE, 1.0, k_pad / (k_pad + obj.material.k_surface))
        f_sq = max(0.0, float(np.sum(ld.fx * share * -ld.slope)))
        top_i = int(np.argmax(np.where(loaded, ld.z, -np.inf)))
        tilt = math.atan(abs(float(ld.slope[top_i])))

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

    # Least grip per finger that holds at margin 1, with and without the safety factors, keeping this
    # contact's geometry: hold scales the squeeze-out with the grip; pitch also gains the friction couple.
    sq_ratio = f_sq / N
    mu_eff = mu / (1.0 - mu * r_t / arm) if arm > mu * r_t else math.inf
    w_half = m * (G + az) / 2

    def grip_for_hold(sf: float) -> float:
        net = mu_eff - sf * sq_ratio
        return sf * w_half / net if net > 1e-9 else math.inf

    def pitch_capacity(n: float) -> float:
        fric = max(0.0, 2 * mu * n - m * (G + az)) * x_face * 1e-3
        if pivot_pad is not None:
            return 2 * k_pp * theta_allow + (2 * n * e_max if spare <= sc.max_rotation else 0.0) + fric
        return 2 * n * e_max + fric

    def grip_for_pitch(sf: float) -> float:
        need = sf * m_dem
        if need <= pitch_capacity(0.0):
            return 0.0
        lo, hi = 0.0, 1.0
        while pitch_capacity(hi) < need:
            hi *= 2.0
            if hi > 1e6:
                return math.inf
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if pitch_capacity(mid) < need else (lo, mid)
        return hi

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
        "contact_face_tilt_deg": math.degrees(tilt),  # pad face to the box side at the topmost loaded point
        "squeeze_out_ratio": f_sq / N,  # friction coefficient the squeeze-out alone needs; tan(tilt) for a contact at the box's top edge only
        "pitch_demand_Nm": m_dem,
        "pitch_capacity_Nm": m_cap,
        "pads": ",".join(p.name for p in used_pads),
        "pad_align_deg": math.degrees(_pad_rotation(ld) - math.radians(d.pad_angle)) if pivot_pad is not None else 0.0,
        "pad_pitch_rotation_deg": math.degrees(rot),
        "max_pad_strain": float(ld.pad_strain.max()),
        "min_grip_hold_N": grip_for_hold(1.0),
        "min_grip_hold_sf_N": grip_for_hold(sc.friction_sf),
        "min_grip_pitch_N": grip_for_pitch(1.0),
        "min_grip_pitch_sf_N": grip_for_pitch(sc.pitch_sf),
    }
    return GripResult(d, obj.id, ok=True, margins=margins, info=info)


def screen_design(design: GripperDesign, objects: list[GraspObject], sc: Scenario) -> list[GripResult]:
    f = build_finger(design, sc.keep_out, sc.closed_limit)
    return [solve_grip(design, o, sc, finger=f) for o in objects]
