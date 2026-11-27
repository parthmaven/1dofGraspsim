"""Batch-1 screen (level 0): a jaw on a box or a sheet, to test conditions revision 2.10.

Built on the first pass's screen (analytic.py, which PR #2's pipeline keeps), with these changes:

* Grip faces are the pads plus the marked bare faces. A bare face is rigid, so the object's own
  surface stiffness sets its pressure, at the bare friction for the object.
* The hold term resolves each sample's force on its own contact normal: the box's normal turned by
  the sample's share of the face tilt (all of it at the box's top corner and on rigid faces, the
  pad-and-box series split elsewhere), with friction along that face. Friction's moment about the
  pivot and its vertical share follow the face, not the box side.
* A box's insertion is scanned in 3 mm steps from the shallowest grip to the cap. Body contact on
  the box top ends the scan as a depth stop, found to 0.2 mm. The chosen depth is the one whose
  +-3 mm neighbours (the placement error) score best, at most 3 mm short of the stop, so the deep
  corner lands on it.
* Box limits per finger: the top band (15 N, x50 stiffness over its top 4 mm), the empty panel
  (10 N, interpolated across the fill line) and the bed (25 kPa, only below the fill line).
* The shake's pitching moment goes to the friction couple first, so the bed's pressure carries only
  the excess, m_pad = max(0, m_dem - m_fric) / 2 per pad. A grip face under 20 mm gets pitch from its
  pressure couple only up to that couple's stiffness times the rotation limit (SIM, round 9).
* At a sheet's 20 mm pinch, "pads bottom out" is reported for MuJoCo, not as a verdict.

Symmetric jaws only: an asymmetric jaw (S4) is scored in MuJoCo alone. Its insertion scan screens each finger as a
mirrored pair, and the jaw's grip at a depth is the worse finger's, stopped where either finger stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from shapely.geometry import Polygon as SPolygon

from .geometry import dir_to_view, to_view
from .jaw import BARE_INSET, Jaw, JawSide, _polys, _strip
from .objects import G, GraspObject
from .pads import foundation_stiffness
from .scenario import Scenario

SAMPLE_PITCH = 0.5  # mm between face samples
N_MIN = 21
CORNER_ZONE = 2.0  # mm below the object's top edge: the contact normal follows the face there
TABLE_CLEARANCE = 2.0  # mm
MAX_PAD_STRAIN = 0.85  # MuJoCo's compression stop
K_RIGID = 1.7e9 / 2e-3  # Pa/m: a PA12 face (E 1.7 GPa) over 2 mm
STOP_TOP = 3.0  # mm: body contact this close to the box top is a depth stop
SHORT_PAD = 20.0  # mm: a grip face this short gets pitch only from its pressure couple's stiffness within the rotation limit


@dataclass
class Limits:
    """Per-run values from config/batch1/scenario.yaml (`batch1:`)."""

    clearance_box: float = 7.0
    clearance_sheet: float = 4.0
    band_mm: float = 15.0
    band_force: float = 15.0
    band_stiff_mm: float = 4.0
    band_factor: float = 50.0
    panel_force: float = 10.0
    bed_pressure: float = 25e3  # Pa
    humid: float = 0.6
    shake_g: float = 0.5
    step: float = 3.0
    placement: float = 3.0
    friction_scale: float = 1.0
    robust_friction: float = 0.8

    @classmethod
    def from_config(cls, b1: Mapping[str, Any], task: str = "T1", **over: Any) -> "Limits":
        bl = b1.get("box_limits", {})
        ac = b1.get("approach_clearance_mm", {})
        out = cls(clearance_box=float(ac.get("box", 7.0)), clearance_sheet=float(ac.get("sheet", 4.0)),
                  band_mm=float(bl.get("band_mm", 15.0)), band_force=float(bl.get("band_force_N", 15.0)),
                  band_stiff_mm=float(bl.get("band_stiff_mm", 4.0)), band_factor=float(bl.get("band_stiff_factor", 50.0)),
                  panel_force=float(bl.get("panel_force_N", 10.0)), bed_pressure=float(bl.get("bed_pressure_kPa", 25.0)) * 1e3,
                  humid=float(b1.get("humid_factor", 0.6)), shake_g=float(b1.get("shake_g", {}).get(task, 0.5)),
                  step=float(b1.get("insertion_step_mm", 3.0)), placement=float(b1.get("placement_mm", 3.0)))
        for k, v in over.items():
            setattr(out, k, v)
        return out


@dataclass
class Grip:
    obj: str
    ok: bool
    reason: str = ""
    margins: dict[str, float] = field(default_factory=dict)
    demand: dict[str, float] = field(default_factory=dict)  # what each margin was scored on, in its unit
    info: dict[str, Any] = field(default_factory=dict)
    status: str = ""  # "stop" (a depth stop), "mujoco" (bottomed at a 20 mm pinch), "nogrip"

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
        return min(self.margins, key=self.margins.get) if self.margins else ""


@dataclass
class _Side:
    pts: np.ndarray  # finger-local mm
    nrm: np.ndarray
    ds: np.ndarray  # m
    w: np.ndarray  # m
    k_pad: np.ndarray  # Pa/m
    thick: np.ndarray  # mm
    mu: np.ndarray
    rigid: np.ndarray
    face: np.ndarray
    names: list[str]
    grip_np: list[np.ndarray]  # finger-local polygons: pads and the strips under marked faces
    body_np: list[np.ndarray]  # the rest of the body, unmarked bare faces included
    face_len: np.ndarray = field(default_factory=lambda: np.zeros(0))  # mm, per grip face


def _geometry(js: JawSide) -> tuple[list[np.ndarray], list[np.ndarray]]:
    geo = getattr(js, "_screen_geo", None)
    if geo is None:
        grip = [f.pad.outline() for f in js.pads]
        for f in js.marked:
            for g in _polys(_strip(f.pad, BARE_INSET).intersection(js.outline)):
                if g.area > 0.05:
                    grip.append(np.asarray(g.exterior.coords)[:-1])
        body = [np.asarray(g.exterior.coords)[:-1] for g in js.body_screen]
        geo = (grip, body)
        js._screen_geo = geo
    return geo


def _side(js: JawSide, obj: GraspObject, friction_scale: float = 1.0) -> _Side:
    P, Nn, D, W, K, T, M, R, F = [], [], [], [], [], [], [], [], []
    names, lens = [], []
    for i, f in enumerate(js.grip_faces):
        pad = f.pad
        n = max(N_MIN, min(241, int(pad.length / SAMPLE_PITCH) + 1))
        pts, nrm, ds, th = pad.face_samples(n)
        w = np.minimum(np.array([f.width_at(float(u)) for u in pts[:, 0]]), obj.width * 1e3) * 1e-3
        if f.rigid:
            k = np.full(n, K_RIGID)
            mu = obj.material.bare_friction
        else:
            w_mean = float(np.mean(w))
            lut = {t: foundation_stiffness(pad.shore, pad.length * 1e-3, w_mean, t * 1e-3) for t in np.unique(th)}
            k = np.array([lut[t] for t in th])
            mu = obj.material.friction
        P.append(pts), Nn.append(nrm), D.append(ds * 1e-3), W.append(w), K.append(k), T.append(th)
        M.append(np.full(n, mu * friction_scale)), R.append(np.full(n, f.rigid)), F.append(np.full(n, i))
        names.append(f.name)
        lens.append(float(pad.length))
    grip, body = _geometry(js)
    cat = np.concatenate
    if not P:
        z = np.zeros(0)
        return _Side(np.zeros((0, 2)), np.zeros((0, 2)), z, z, z, z, z, z.astype(bool), z.astype(int), names, grip, body)
    return _Side(cat(P), cat(Nn), cat(D), cat(W), cat(K), cat(T), cat(M), cat(R).astype(bool), cat(F).astype(int), names, grip, body,
                 np.array(lens))


def _band_min(polys: list[np.ndarray], z_lo: float, z_hi: float) -> tuple[float, float]:
    """Smallest x any polygon reaches within z_lo <= z <= z_hi, and the z where it does."""
    best, zb = math.inf, math.nan
    for P in polys:
        x, z = P[:, 0], P[:, 1]
        inside = (z >= z_lo) & (z <= z_hi)
        if inside.any():
            i = int(np.argmin(np.where(inside, x, np.inf)))
            if x[i] < best:
                best, zb = float(x[i]), float(z[i])
        x2, z2 = np.roll(x, -1), np.roll(z, -1)
        for zl in (z_lo, z_hi):
            if not math.isfinite(zl):
                continue
            cross = (z - zl) * (z2 - zl) < 0
            if cross.any():
                t = (zl - z[cross]) / (z2[cross] - z[cross])
                xc = x[cross] + t * (x2[cross] - x[cross])
                j = int(np.argmin(xc))
                if xc[j] < best:
                    best, zb = float(xc[j]), zl
    return best, zb


def _view(polys: list[np.ndarray], phi: float, px: float) -> list[np.ndarray]:
    return [to_view(P, phi, px) for P in polys]


def approach_opening(jaw: Jaw, S: _Side, x_need: float, z_bot: float, z_top: float) -> tuple[float, float]:
    """Smallest jaw angle (rad) at which every part of the finger clears x_need over the object's
    height, or nan; and, when it never does, the z of the part that blocks it at full stroke."""
    allp = S.grip_np + S.body_np
    px = jaw.pivot_x

    def clear(phi: float) -> bool:
        return _band_min(_view(allp, phi, px), z_bot, z_top)[0] >= x_need

    lo, hi = jaw.phi_meet, jaw.phi_max
    if not clear(hi):
        return math.nan, _band_min(_view(allp, hi, px), z_bot, z_top)[1]
    if clear(lo):
        return lo, math.nan
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if clear(mid) else (mid, hi)
    return hi, math.nan


def _series(k_pad: np.ndarray, k_obj: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 / k_pad + 1.0 / k_obj)


def _load(S: _Side, phi: float, px: float, x_face: float, z_lo: float, z_hi: float, k_obj: float, band_top: float,
          band_factor: float) -> dict[str, np.ndarray]:
    v = to_view(S.pts, phi, px)
    x, z = v[:, 0], v[:, 1]
    inside = (z >= z_lo) & (z <= z_hi)
    delta = np.where(inside, np.maximum(x_face - x, 0.0), 0.0) * 1e-3
    k_o = np.where(z >= band_top, k_obj * band_factor, k_obj)
    p = _series(S.k_pad, k_o) * delta
    N = p * S.w * S.ds
    strain = np.where(S.rigid, 0.0, p / S.k_pad / np.maximum(S.thick * 1e-3, 1e-6))
    return {"x": x, "z": z, "p": p, "N": N, "k_o": k_o, "strain": strain}


def _torque_box(ld: dict[str, np.ndarray]) -> float:
    return float(np.sum(-ld["z"] * 1e-3 * ld["N"]))


def _frames(S: _Side, ld: dict[str, np.ndarray], phi: float, px: float, x_face: float, z_top: float):
    nv = dir_to_view(S.nrm, phi)
    fx, fz = -nv[:, 0], -nv[:, 1]
    theta = np.arctan2(fz, fx)
    share = np.where(S.rigid | (ld["z"] >= z_top - CORNER_ZONE), 1.0, S.k_pad / (S.k_pad + ld["k_o"]))
    a = share * theta
    ex, ez = np.cos(a), np.sin(a)
    tx, tz = ez, -ex
    rx = (x_face - px) * 1e-3
    rz = ld["z"] * 1e-3
    Me = rx * ez - rz * ex
    Mt = rx * tz - rz * tx
    return ex, ez, tx, tz, Me, Mt, theta


def _hold(w: np.ndarray, mu: np.ndarray, ez: np.ndarray, tz: np.ndarray, Me: np.ndarray, Mt: np.ndarray, tau: float,
          W_half: float, sf: float) -> dict[str, float]:
    """Hold per finger with every sample's force on its own frame (see module docstring).
    Returns the capacity margin, the loaded normal force and the friction share used."""
    A = float(np.sum(w * -ez))
    B = float(np.sum(w * mu * -tz))
    C = float(np.sum(w * Me))
    Dm = float(np.sum(w * mu * Mt))
    if C + Dm <= 1e-12:
        cap = math.inf
    else:
        cap = max(tau / (C + Dm) * (A + B), 0.0)
    margin = cap / (sf * W_half) if W_half > 0 else math.inf
    den = tau * B - W_half * Dm
    if den <= 1e-12:
        kappa = 1.0 if W_half * C > tau * A else -1.0
    else:
        kappa = (W_half * C - tau * A) / den
    k_use = min(max(kappa, -1.0), 1.0)
    lam = tau / max(C + k_use * Dm, 1e-12)
    return {"margin": margin, "capacity_N": cap, "n_load": lam, "kappa": kappa, "A": A, "B": B, "C": C, "D": Dm}


def solve(jaw: Jaw, obj: GraspObject, sc: Scenario, lim: Limits, depth: float, S: _Side | None = None,
          tau: float | None = None) -> Grip:
    """One grasp at `depth` mm (a box's top, or a sheet's free edge, above the fingertip line)."""
    if not jaw.symmetric:
        return Grip(obj.id, False, "asymmetric jaw: MuJoCo only", status="nogrip")
    S = S or _side(jaw.right, obj, lim.friction_scale)
    tau = jaw.torque if tau is None else tau
    px = jaw.pivot_x
    L = jaw.reach
    box = obj.kind == "box"
    x_face = obj.thickness * 1e3 / 2
    z_top = -L + depth
    z_bot = z_top - obj.height * 1e3
    info: dict[str, Any] = {"insertion_mm": depth}
    if z_top >= -1.0:
        return Grip(obj.id, False, "object top above the pivots", info=info, status="stop")
    if sc.keep_out is not None and box:
        top = sc.keep_out.max_object_top(-x_face, x_face)
        if z_top > top + 1e-6:
            return Grip(obj.id, False, "object top would enter the hub keep-out", info=info, status="stop")
    clr = lim.clearance_box if box else lim.clearance_sheet
    phi_open, z_block = approach_opening(jaw, S, x_face + clr, z_bot, z_top)
    if not math.isfinite(phi_open):
        if box and math.isfinite(z_block) and z_block >= z_top - STOP_TOP:
            return Grip(obj.id, False, "a finger part sits over the box top", info=info, status="stop")
        return Grip(obj.id, False, "does not fit between the open fingers", info=info)
    info["approach_opening_deg"] = math.degrees(phi_open - jaw.phi_meet)
    allp = S.grip_np + S.body_np

    def x_all(phi: float) -> float:
        return _band_min(_view(allp, phi, px), z_bot, z_top)[0]

    lo_lim = jaw.phi_meet
    if x_all(lo_lim) > x_face:
        return Grip(obj.id, False, "fingers close fully without touching the object", info=info, status="nogrip")
    lo, hi = lo_lim, phi_open
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if x_all(mid) <= x_face:
            lo = mid
        else:
            hi = mid
    phi_touch = hi
    info["contact_angle_deg"] = math.degrees(phi_touch - jaw.phi_meet)
    xg, zg = _band_min(_view(S.grip_np, lo, px), z_bot, z_top)
    xb, zb = _band_min(_view(S.body_np, lo, px), z_bot, z_top)
    if xb <= x_face and (xb <= xg or xg > x_face):
        if box and zb >= z_top - STOP_TOP:
            return Grip(obj.id, False, "the body lands on the box top", info=info, status="stop")
        return Grip(obj.id, False, "the body touches the object before the grip faces", info=info, status="nogrip")
    if not box and _band_min(_view(allp, phi_touch, px), z_top + 1e-6, math.inf)[0] <= 0.0:
        return Grip(obj.id, False, "fingertips meet above the sheet", info=info)

    k_obj = obj.material.k_surface
    band_top = z_top - lim.band_stiff_mm if box else math.inf
    bf = lim.band_factor if box else 1.0

    def load(phi: float) -> dict[str, np.ndarray]:
        return _load(S, phi, px, x_face, z_bot, z_top, k_obj, band_top, bf)

    lo, hi = max(phi_touch - math.radians(25.0), jaw.phi_meet - math.radians(25.0)), phi_touch
    if _torque_box(load(lo)) < tau:
        return Grip(obj.id, False, "pads bottom out before the drive torque is reached", info=info,
                    status="mujoco" if not box else "")
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if _torque_box(load(mid)) > tau:
            lo = mid
        else:
            hi = mid
    phi = 0.5 * (lo + hi)
    ld = load(phi)
    if _torque_box(ld) <= 0.0:
        phi = lo
        ld = load(lo)
    ex, ez, tx, tz, Me, Mt, theta = _frames(S, ld, phi, px, x_face, z_top)
    M0 = float(np.sum(ld["N"] * Me))
    if M0 <= 1e-9:
        return Grip(obj.id, False, "the contact does not resist the drive", info=info)
    s = tau / M0
    N_i = ld["N"] * s
    p_i = ld["p"] * s
    strain = ld["strain"] * s
    N = float(N_i.sum())
    if N <= 1e-9:
        return Grip(obj.id, False, "no grip force developed", info=info)
    info["phi_deg"] = math.degrees(phi - jaw.phi_meet)
    xb2, zb2 = _band_min(_view(S.body_np, phi, px), z_bot, z_top)
    if xb2 <= x_face:
        if box and zb2 >= z_top - STOP_TOP:
            return Grip(obj.id, False, "the body lands on the box top while squeezing", info=info, status="stop")
        return Grip(obj.id, False, "the body reaches the object while squeezing", info=info)
    if float(strain.max()) > MAX_PAD_STRAIN:
        return Grip(obj.id, False, "pads bottom out", info=info, status="" if box else "mujoco")
    if box:
        zmin = math.inf
        for ph in (phi, phi_open):
            zmin = min(zmin, min(float(P[:, 1].min()) for P in _view(allp, ph, px)))
        if zmin < z_bot + TABLE_CLEARANCE:
            return Grip(obj.id, False, "fingertips hit the table", info=info)

    loaded = N_i > 1e-9
    w = N_i / N
    m = obj.mass
    a_x = a_z = lim.shake_g * G
    W_half = m * (G + a_z) / 2
    h0 = _hold(w, S.mu, ez, tz, Me, Mt, tau, W_half, sc.friction_sf)
    hr = _hold(w, S.mu * lim.robust_friction, ez, tz, Me, Mt, tau, W_half, sc.friction_sf)
    n_dem = min(h0["n_load"], tau / max(h0["C"] + h0["D"], 1e-12)) if h0["kappa"] > 1 else h0["n_load"]
    scale = n_dem / N
    margins: dict[str, float] = {"hold": h0["margin"]}
    demand: dict[str, float] = {"hold": W_half * sc.friction_sf}

    zc = ld["z"][loaded]
    contact_len = max(float(zc.max() - zc.min()), 0.5) * 1e-3
    cop_z = float(np.sum(ld["z"] * N_i) / N)
    z_com = z_top - obj.com_below_top * 1e3
    h = abs(cop_z - z_com) * 1e-3
    m_dem = m * a_x * h
    mu_bar = float(np.sum(w * S.mu))
    e_max = contact_len / 3
    m_fric = max(0.0, 2 * mu_bar * n_dem - m * (G + a_z)) * x_face * 1e-3
    m_norm = 2 * n_dem * e_max
    short = bool(len(S.face_len)) and float(S.face_len[np.unique(S.face[loaded])].max()) < SHORT_PAD
    if short:
        # A short pad's pressure couple is far softer than the friction couple, so within the rotation limit it
        # gives only its own stiffness times that angle (SIM, round 9).
        k_rot = float(np.sum((_series(S.k_pad, ld["k_o"]) * S.w * S.ds * ((ld["z"] - cop_z) * 1e-3) ** 2)[loaded]))
        m_norm = min(m_norm, 2 * k_rot * sc.max_rotation)
    m_cap = m_norm + m_fric
    margins["pitch"] = m_cap / (sc.pitch_sf * m_dem) if m_dem > 1e-12 else math.inf
    demand["pitch"] = sc.pitch_sf * m_dem
    b = float(S.w[loaded].min())
    # Until a pad slips, the couples share the moment by stiffness, and the friction couple is hundreds of times
    # stiffer than the pressure couple: the bed takes only what friction can't.
    m_pad = max(0.0, m_dem - m_fric) / 2
    p_dem = p_i * scale + 6 * m_pad / (b * contact_len ** 2) * loaded
    if box:
        # Each zone takes its own limit: the top band its force, the empty panel above the fill line
        # its force alone, and only the wall the cereal backs the bed pressure.
        in_band = ld["z"] >= z_top - lim.band_mm
        fill_line = z_top - obj.headspace * 1e3
        bed = loaded & ~in_band & (ld["z"] <= fill_line)
        on_pad = loaded & ~S.rigid
        info["pad_below_fill"] = bool((ld["z"][on_pad if on_pad.any() else loaded] <= fill_line).all())
        p_out = float(p_dem[bed].max()) if bed.any() else 0.0
        if p_out > 0:
            margins["pressure"] = lim.bed_pressure / (sc.pressure_sf * p_out)
            demand["pressure"] = p_out / 1e3
        f_band = float(N_i[in_band].sum()) * scale
        f_panel = float(N_i[(ld["z"] > fill_line) & ~in_band].sum()) * scale
        if f_band > 1e-6:
            margins["band"] = lim.band_force / f_band
            demand["band"] = f_band
        if f_panel > 1e-6:
            margins["panel"] = lim.panel_force / f_panel
            demand["panel"] = f_panel
        info.update(band_force_N=f_band, panel_force_N=f_panel, bed_pressure_kPa=p_out / 1e3,
                    humid_band=lim.band_force * lim.humid / f_band if f_band > 1e-6 else math.inf,
                    humid_panel=lim.panel_force * lim.humid / f_panel if f_panel > 1e-6 else math.inf)
    else:
        p_max = float(p_dem.max())
        margins["pressure"] = obj.material.allow_pressure / (sc.pressure_sf * p_max)
        demand["pressure"] = p_max / 1e3
        exit_i = int(np.argmin(np.where(loaded, ld["z"], np.inf)))
        b_exit = float(S.w[exit_i]) if sc.crease_width == "pad" else obj.width
        cap = obj.material.bending_strength * b_exit
        margins["crease"] = cap / (sc.crease_safety_factor * m_dem) if m_dem > 1e-12 else math.inf
        info["crease_width_mm"] = b_exit * 1e3
        demand["crease"] = sc.crease_safety_factor * m_dem
    used = sorted(set(S.face[loaded].tolist()))
    tilt = float(np.degrees(np.abs(theta[loaded]).max()))
    info.update({
        "grip_N": N, "grip_loaded_N": n_dem, "friction_use": h0["kappa"], "hold_robust": hr["margin"],
        "hold_capacity_N": h0["capacity_N"], "support_demand_N": W_half,
        "contact_length_mm": contact_len * 1e3, "cop_below_top_mm": z_top - cop_z, "faces": ",".join(S.names[i] for i in used),
        "bare_share": float(N_i[S.rigid].sum() / N), "face_tilt_deg": tilt, "max_pad_strain": float(strain.max()),
        "peak_pressure_kPa": float(p_dem.max()) / 1e3, "mu_mean": mu_bar,
        "drive_Nm_per_gripper": 2 * tau, "moment_Nm": m_dem, "friction_couple_Nm": m_fric,
        "pressure_couple_Nm": m_norm, "bed_moment_Nm": m_pad, "short_pads": short,
    })
    return Grip(obj.id, True, "", {k: float(min(v, 99.0)) for k, v in margins.items()}, demand, info)


def insertion_cap(jaw: Jaw, obj: GraspObject, sc: Scenario) -> tuple[float, str]:
    pres = sc.for_kind("box")
    caps = [(pres.max_insertion * 1e3, f"the {pres.max_insertion * 1e3:g} mm insertion cap")]
    caps.append((obj.height * 1e3 - 2 * TABLE_CLEARANCE, "the table"))
    if sc.keep_out is not None:
        top = sc.keep_out.max_object_top(-obj.thickness * 500, obj.thickness * 500)
        if math.isfinite(top):
            caps.append((jaw.reach + top, "the hub keep-out"))
    return min(caps)


@dataclass
class BoxScan:
    obj: str
    nominal: Grip | None
    depth: float  # mm, the chosen depth
    robust: float  # least score over the chosen depth and its +-3 mm neighbours
    corners: dict[str, Grip]
    scanned: tuple[float, float]  # first depth with a grip, last depth scanned
    ended: str
    grid: dict[float, Grip]
    stop_mm: float = math.nan  # the depth stop, where one ended the scan
    cap_mm: float = math.nan
    deep: tuple[float, Grip] | None = None  # the best grip with the pad all below the fill line

    @property
    def passed(self) -> bool:
        return self.robust >= 1.0

    def row(self) -> dict[str, Any]:
        g = self.nominal
        out = {"object": self.obj, "depth_mm": self.depth, "robust": self.robust, "passed": self.passed,
               "scan_from_mm": self.scanned[0], "scan_to_mm": self.scanned[1], "scan_ended": self.ended,
               "stop_mm": self.stop_mm, "cap_mm": self.cap_mm}
        dd, dg = self.deep if self.deep is not None else (math.nan, None)
        out.update(deep_depth_mm=dd, deep_score=dg.score if dg is not None else math.nan,
                   deep_limiting=dg.limiting if dg is not None else "",
                   deep_bed_kPa=dg.info.get("bed_pressure_kPa", math.nan) if dg is not None else math.nan)
        if g is not None:
            out.update(score=g.score, limiting=g.limiting, reason=g.reason)
            out.update({f"m_{k}": v for k, v in g.margins.items()})
            out.update({f"d_{k}": v for k, v in g.demand.items()})
            out.update({k: v for k, v in g.info.items() if not isinstance(v, (list, dict))})
        for k, c in self.corners.items():
            out[f"corner_{k}"] = c.score if c.ok else 0.0
            out[f"corner_{k}_why"] = c.limiting
        return out


def finger_pair(jaw: Jaw, side: int) -> Jaw:
    """One finger of a jaw as a mirrored pair (each side's geometry is drawn in its own mirrored view)."""
    return replace(jaw, sides={1: jaw.sides[side], -1: jaw.sides[side]}, symmetric=True)


def scan_box(jaw: Jaw, obj: GraspObject, sc: Scenario, lim: Limits, tau: float | None = None) -> BoxScan:
    pairs = [jaw] if jaw.symmetric else [finger_pair(jaw, s) for s in (1, -1)]
    sides = [_side(p.right, obj, lim.friction_scale) for p in pairs]

    def solve_at(depth: float) -> Grip:
        gs = [solve(p, obj, sc, lim, depth, S, tau) for p, S in zip(pairs, sides)]
        return next((g for g in gs if g.status == "stop"), None) or min(gs, key=lambda g: (g.ok, g.score))

    cap, cap_why = insertion_cap(jaw, obj, sc)
    step = lim.step
    grid: dict[float, Grip] = {}
    ended = f"reached the cap ({cap_why}, {cap:.0f} mm)"
    d = step
    while d <= cap + 1e-6:
        g = solve_at(d)
        grid[round(d, 3)] = g
        if g.status == "stop":
            ended = f"depth stop at {d:.0f} mm: {g.reason}"
            break
        d += step
    depths = sorted(grid)
    first = next((x for x in depths if grid[x].status not in ("nogrip",) and grid[x].ok), None)
    if first is None:
        first = next((x for x in depths if grid[x].status != "nogrip"), depths[0] if depths else math.nan)
    last = depths[-1] if depths else math.nan
    stop_at = math.nan
    deepest = cap
    if ended.startswith("depth stop") and depths:
        # The stop itself, to 0.2 mm: the nominal sits a placement error short of it, and the deep
        # corner lands on it.
        lo, hi = max(last - step, 0.0), last
        why = grid[round(last, 3)].reason
        for _ in range(4):
            mid = 0.5 * (lo + hi)
            g = solve_at(mid)
            if g.status == "stop":
                hi, why = mid, g.reason
            else:
                lo = mid
        stop_at = round(lo, 3)
        ended = f"depth stop at {stop_at:.1f} mm: {why}"
        if stop_at > 0:
            grid.setdefault(stop_at, solve_at(stop_at))
            x_stop = round(stop_at - lim.placement, 3)
            if x_stop > 0:
                grid.setdefault(x_stop, solve_at(x_stop))
        deepest = min(cap, stop_at - lim.placement)
        depths = sorted(grid)
    elif depths:
        extra = round(last + lim.placement, 3)
        if extra not in grid:
            grid[extra] = solve_at(extra)

    def sc_of(x: float) -> float:
        g = grid.get(round(x, 3))
        if g is None and x > 0:
            g = grid[round(x, 3)] = solve_at(x)
        return g.score if (g is not None and g.ok) else 0.0

    best = None
    for x in depths:
        if grid[x].status == "stop" or x > deepest + 1e-6:
            continue
        rob = min(sc_of(x - lim.placement), sc_of(x), sc_of(x + lim.placement))
        key = (min(rob, 1.0), min(sc_of(x), 1.0), rob, sc_of(x))
        if best is None or key > best[0]:
            best = (key, x, rob)
    if best is None:
        return BoxScan(obj.id, grid[depths[0]] if depths else None, depths[0] if depths else math.nan, 0.0, {},
                       (first, last), ended, grid, stop_at, cap)
    _, x, rob = best
    corners = {}
    for tag, dx in (("deep", lim.placement), ("shallow", -lim.placement)):
        g = grid.get(round(x + dx, 3))
        if g is not None:
            corners[tag] = g
    # The best grip with the pad's contact all below the fill line, on the wall the cereal backs (VAL R9.4).
    bed = [(grid[d].score, d) for d in depths if d <= deepest + 1e-6 and grid[d].ok and grid[d].info.get("pad_below_fill")]
    deep = (max(bed)[1], grid[max(bed)[1]]) if bed else None
    return BoxScan(obj.id, grid[x], x, rob, corners, (first, last), ended, grid, stop_at, cap, deep)


def solve_sheet(jaw: Jaw, obj: GraspObject, sc: Scenario, lim: Limits, pinch: float, tau: float | None = None) -> Grip:
    return solve(jaw, obj, sc, lim, pinch, None, tau)
