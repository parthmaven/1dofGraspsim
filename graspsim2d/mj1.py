"""Batch-1 MuJoCo check (level 1): a jaw on a box or a sheet (T1, T2), to test conditions
revision 2.10. The gripper plane is MuJoCo's x-z plane (y out of plane, gravity -z).

Differences from the first pass's model (mjsim.py, which PR #2's pipeline keeps):

* Both fingers come from the jaw, each with its own geometry, so an asymmetric pair runs as drawn.
* The printed body collides as convex mesh pieces, at the bare friction for the object. Every face
  that can touch an object, pad or bare, is split into sprung elements: at 4 mm, or 1.5 mm on faces
  shorter than 20 mm and on curved faces. A bare face's elements carry the object's own surface
  stiffness over 1.5 mm of travel, at the bare friction; the body is cut back 1.6 mm under them.
* Element force is read from the contact force, which stays right when an element reaches its
  compression stop (85 % of a pad's thickness).
* The drive closes at 1 rad/s, then at 0.3 rad/s over the last 5 deg before the touch, through
  0.015 kg m^2 of reflected inertia per finger axis, then holds its set torque. The closing impact
  is scored against the band, panel and pressure limits.
* After the shake: a stop staircase (each level as from 2 m/s), then the power cut: drive torque to
  zero with the brake on (15 N m per finger motor for the 30 N m reference), held 5 s.
* Box limits per finger: the top band (15 N, x50 stiffer over its top 4 mm), the empty panel
  (10 N, interpolated across the fill line) and the bed (25 kPa, only below the fill line). In T1 - T3 the
  body and the unmarked bare faces may put at most 1 N on the object.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .geometry import to_view
from .jaw import Jaw, JawSide, convex_pieces, element_layout
from .objects import G, GraspObject
from .pads import foundation_stiffness, series, shear_stiffness
from .scenario import Scenario
from .screen1 import K_RIGID, Limits, _band_min, _geometry, _view
from .sensor import layer_depth

GRIP, OBJ, TABLE = 1, 2, 4
PA12_DENSITY = 1010.0


def finger_density(b1: dict) -> float:
    """kg/m^3 of the selected finger material (`finger_material`); printed PA12 where the config names none."""
    mats = b1.get("finger_materials") or {}
    return float((mats.get(b1.get("finger_material", "pa12")) or {}).get("density_kg_m3", PA12_DENSITY))
PAD_DENSITY = 1200.0
TABLE_FRICTION = 0.4
DROP = 0.02  # m of slip counted as a drop
E_H = 0.4  # mm, half-thickness of an element geom
STOP_SPEED = 2.0  # m/s
RETURN_G = 0.25  # g, the gentle return after each stop level
BAD_STATE = tuple(int(w) for w in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                                   mujoco.mjtWarning.mjWARN_BADQACC))


def _fmt(v: Any) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(f"{float(x):.6g}" for x in v)
    return f"{float(v):.6g}"


def _beta(dX: float, dZ: float) -> float:
    return math.atan2(-dZ, dX)


def _p(u: float, v: float, side: int) -> tuple[float, float]:
    return (-side * v * 1e-3, -u * 1e-3)


def _d(du: float, dv: float, side: int) -> tuple[float, float]:
    return (-side * dv, -du)


def side_touch(jaw: Jaw, side: int, x_face: float, z_bot: float, z_top: float, clearance: float) -> tuple[float, float]:
    """(approach opening, touch angle) in rad for one finger against an object face at x_face (mm,
    in that finger's own mirrored view)."""
    grip, body = _geometry(jaw.sides[side])
    allp = grip + body
    px = jaw.pivot_x

    def xmin(phi: float) -> float:
        return _band_min(_view(allp, phi, px), z_bot, z_top)[0]

    def bis(target: float, want_clear: bool) -> float:
        lo, hi = jaw.phi_meet, jaw.phi_max
        if want_clear:
            if xmin(hi) < target:
                return math.nan
            if xmin(lo) >= target:
                return lo
        else:
            if xmin(lo) > target:
                return lo
            if xmin(hi) <= target:
                return hi
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            ok = xmin(mid) >= target if want_clear else xmin(mid) > target
            lo, hi = (lo, mid) if ok else (mid, hi)
        return hi

    return bis(x_face + clearance, True), bis(x_face, False)


@dataclass
class _El:
    geom: str
    joint: str
    side: int
    kind: str  # pad, marked, bare
    face: str
    area: float  # m^2
    seg: float  # mm
    w: float  # m
    ext: float  # mm, vertical extent at the planned grip
    depth: float = 0.0  # mm along the sensor's axis below the top datum, from the element's layer


class _Builder:
    def __init__(self, jaw: Jaw, obj: GraspObject, sc: Scenario, b1: dict, depth: float, phi_plan: dict[int, float],
                 friction_scale: float, band_factor: float, z_top_plan: float):
        self.jaw, self.obj, self.sc, self.b1 = jaw, obj, sc, b1
        sim = sc.sim
        self.dt = float(sim.get("timestep_s", 1e-4))
        # Finer integration runs in substeps of dt; every time constant stays tied to dt.
        self.fine_dt = float(sim.get("fine_timestep_s", self.dt / 2))
        self.substeps = max(1, int(b1.get("substeps", 1)))
        self.pitch = float(sim.get("pad_element_pitch_mm", 4.0))
        self.fine = float(b1.get("fine_pitch_mm", 1.5))
        self.short = float(b1.get("short_face_mm", 20.0))
        self.travel = float(b1.get("bare_travel_mm", 1.5))
        self.f_el = float(sim.get("element_frequency_hz", 250.0))
        self.zeta = float(sim.get("element_damping_ratio", 0.3))
        self.contact_tc = float(sim.get("contact_timeconst_s", 2 * self.dt))
        self.k_obj = obj.material.k_surface
        self.mu_pad = obj.material.friction * friction_scale
        self.mu_bare = obj.material.bare_friction * friction_scale
        self.band_factor = band_factor
        self.band_stiff = float(b1.get("box_limits", {}).get("band_stiff_mm", 4.0))
        self.phi_plan = phi_plan
        self.z_top_plan = z_top_plan
        self.elements: list[_El] = []
        self.assets: list[str] = []
        self.body_geoms: dict[str, int] = {}
        self.density = finger_density(b1)
        self.geom_depth: dict[str, float] = {}  # every finger geom, for the sensor's bending

    def armature(self, k: float, mass: float) -> float:
        w = 2 * math.pi * self.f_el
        return max(k / (w * w) - mass, 0.0)

    def finger_xml(self, side: int) -> tuple[str, float, float]:
        js = self.jaw.sides[side]
        tag = "R" if side > 0 else "L"
        out = []
        mass, mom = 0.0, 0.0
        k = 0
        for poly in js.body_mj:
            for piece in convex_pieces(poly):
                c = piece.centroid
                w = js.width_at(float(c.x))
                pts = np.asarray(piece.exterior.coords)[:-1]
                verts = []
                for u, v in pts:
                    X, Z = _p(u, v, side)
                    verts += [X, -w * 0.5e-3, Z, X, w * 0.5e-3, Z]
                name = f"{tag}m{k}"
                self.assets.append(f'<mesh name="{name}" vertex="{_fmt(verts)}"/>')
                gname = f"{tag}_body{k}"
                out.append(f'<geom name="{gname}" type="mesh" mesh="{name}" mass="0" friction="{_fmt(self.mu_bare)} 0.005 0.0001" '
                           f'priority="1" contype="{GRIP}" conaffinity="{OBJ | TABLE}" rgba="0.55 0.6 0.68 1"/>')
                self.body_geoms[gname] = side
                self.geom_depth[gname] = layer_depth(self.b1, js, float(c.x), float(c.y), w)
                m = self.density * piece.area * 1e-6 * w * 1e-3
                mass += m
                mom += m * ((c.x * 1e-3) ** 2 + (c.y * 1e-3) ** 2)
                k += 1
        for face in js.faces:
            out.append(self.face_xml(face, side, tag))
        return "".join(out), mass, mom

    def face_xml(self, face, side: int, tag: str) -> str:
        pad = face.pad
        lay = element_layout(face, self.pitch, self.fine, self.short)
        n_el = len(lay["c"])
        fname = f"{tag}_{pad.name}"
        parts, close = [], []
        mu = self.mu_bare if face.rigid else self.mu_pad
        w_obj = self.obj.width
        if not face.rigid:
            w = min(face.width * 1e-3, w_obj)
            t_n = pad.normal_thickness
            sh_k = shear_stiffness(pad.shore, t_n * 1e-3) * pad.length * 1e-3 * w
            pad_mass = PAD_DENSITY * pad.length * t_n * 1e-6 * w
            tX, tZ = _d(pad.tangent[0], pad.tangent[1], side)
            sh_arm = self.armature(sh_k, pad_mass)
            sh_damp = 2 * self.zeta * math.sqrt(sh_k * (pad_mass + sh_arm))
            parts.append(f'<body name="{fname}_shear" pos="0 0 0" gravcomp="1">'
                         f'<joint name="{fname}_shear" type="slide" axis="{_fmt((tX, 0, tZ))}" stiffness="{_fmt(sh_k)}" '
                         f'damping="{_fmt(sh_damp)}" armature="{_fmt(sh_arm)}" range="{_fmt(-t_n * 0.5e-3)} {_fmt(t_n * 0.5e-3)}"/>'
                         f'<inertial pos="0 0 0" mass="{_fmt(pad_mass * 0.5)}" diaginertia="1e-7 1e-7 1e-7"/>')
            close.append("</body>")
        phi = self.phi_plan[side]
        for j in range(n_el):
            cu, cv = lay["c"][j]
            nu, nv = lay["n"][j]
            seg = float(lay["seg"][j])
            th = float(lay["th"][j])
            w = min(float(lay["w"][j]) * 1e-3, w_obj)
            area = w * seg * 1e-3
            if face.rigid:
                k_pad = K_RIGID
            else:
                k_pad = foundation_stiffness(pad.shore, pad.length * 1e-3, w, th * 1e-3)
            zv = float(to_view(np.array([[cu, cv]]), phi, self.jaw.pivot_x)[0, 1])
            band = self.obj.kind == "box" and self.z_top_plan - self.band_stiff <= zv <= self.z_top_plan + 2.0
            k_o = self.k_obj * (self.band_factor if band else 1.0)
            kk = series(k_pad, k_o) * area
            m_el = PAD_DENSITY * area * 2 * E_H * 1e-3
            arm = self.armature(kk, m_el)
            damp = 2 * self.zeta * math.sqrt(kk * (m_el + arm))
            gc = np.array([cu, cv]) - E_H * np.array([nu, nv])
            X, Z = _p(gc[0], gc[1], side)
            nX, nZ = _d(nu, nv, side)
            tX_, tZ_ = _d(nv, -nu, side)
            jn = f"{fname}_e{j}"
            lo = -(self.travel if face.rigid else 0.85 * th) * 1e-3
            rgba = "0.15 0.15 0.15 1" if not face.rigid else ("0.75 0.45 0.2 1" if face.kind == "marked" else "0.6 0.6 0.6 1")
            parts.append(
                f'<body name="{jn}" pos="{_fmt((X, 0, Z))}" gravcomp="1">'
                f'<joint name="{jn}" type="slide" axis="{_fmt((nX, 0, nZ))}" stiffness="{_fmt(kk)}" damping="{_fmt(damp)}" '
                f'armature="{_fmt(arm)}" range="{_fmt(lo)} 0.0002"/>'
                f'<inertial pos="0 0 0" mass="{_fmt(m_el)}" diaginertia="1e-9 1e-9 1e-9"/>'
                f'<geom name="{jn}" type="box" euler="0 {_fmt(_beta(tX_, tZ_))} 0" size="{_fmt((seg / 2 * 1e-3, w / 2, E_H * 1e-3))}" '
                f'mass="0" friction="{_fmt(mu)} 0.005 0.0001" priority="2" contype="{GRIP}" conaffinity="{OBJ | TABLE}" '
                f'solref="{_fmt(self.contact_tc)} 1" rgba="{rgba}"/></body>')
            tz_view = abs(float(np.cos(phi) * nv - np.sin(phi) * nu))  # |dz| per unit length along the face
            depth = layer_depth(self.b1, self.jaw.sides[side], float(cu), float(cv), float(lay["w"][j]))
            self.geom_depth[jn] = depth
            self.elements.append(_El(jn, jn, side, face.kind, pad.name, area, seg, w, seg * max(tz_view, 0.05), depth))
        return "".join(parts) + "".join(close)


def _min_jerk(s: float) -> tuple[float, float]:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4)


def _envelope(t: float, T: float, ramp: float) -> tuple[float, float]:
    if t <= 0 or t >= T:
        return 0.0, 0.0
    if t < ramp:
        a = math.pi * t / ramp
        return 0.5 * (1 - math.cos(a)), 0.5 * math.pi / ramp * math.sin(a)
    if t > T - ramp:
        a = math.pi * (T - t) / ramp
        return 0.5 * (1 - math.cos(a)), -0.5 * math.pi / ramp * math.sin(a)
    return 1.0, 0.0


class Timeline:
    """Base motion relative to its start: close, settle, lift, hold, shake, hold, the stop levels,
    then the power cut."""

    def __init__(self, sc: Scenario, t_close: float, shake_g: float, shake_hz: tuple[float, float], shake_s: float,
                 stops: list[float], power_s: float):
        self.sc = sc
        self.t_close = t_close
        self.t_settle = t_close + sc.settle_time
        self.t_lift = self.t_settle + sc.lift_time
        self.t_hold = self.t_lift + sc.hold_time
        self.t_shake = self.t_hold + shake_s
        self.t_final = self.t_shake + sc.final_hold_time
        self.shake_s = shake_s
        self.ramp = float(sc.raw.get("sequence", {}).get("shake", {}).get("ramp_s", 0.5))
        self.wx, self.wz = 2 * math.pi * shake_hz[0], 2 * math.pi * shake_hz[1]
        a = shake_g * G
        self.ax = a / (self.wx ** 2) if self.wx > 0 else 0.0
        self.az = a / (self.wz ** 2) if self.wz > 0 else 0.0
        self.stops = []
        t = self.t_final
        for lvl in stops:
            a1 = abs(lvl) * G
            T1 = STOP_SPEED / a1
            a2 = RETURN_G * G
            T2 = STOP_SPEED / a2
            self.stops.append((t, T1, T2, math.copysign(1.0, lvl), a1, a2, lvl))
            t += T1 + T2 + 0.2
        self.t_power = t
        self.t_end = t + power_s

    def stop_state(self, t: float) -> tuple[float, float]:
        x = vx = 0.0
        for t0, T1, T2, sgn, a1, a2, _ in self.stops:
            if t <= t0:
                break
            s = t - t0
            d1 = -0.5 * a1 * T1 * T1
            if s < T1:
                x += sgn * (-0.5 * a1 * s * s)
                vx += sgn * (-a1 * s)
            elif s < T1 + T2:
                r = s - T1
                x += sgn * (d1 - STOP_SPEED * r + 0.5 * a2 * r * r)
                vx += sgn * (-STOP_SPEED + a2 * r)
            else:
                x += sgn * (d1 - STOP_SPEED * T2 + 0.5 * a2 * T2 * T2)
        return x, vx

    def __call__(self, t: float) -> tuple[float, float, float, float]:
        sc = self.sc
        x = z = vx = vz = 0.0
        if t > self.t_settle:
            s = (t - self.t_settle) / sc.lift_time
            p, dp = _min_jerk(s)
            z = sc.lift_height * p
            vz = sc.lift_height * dp / sc.lift_time if s < 1 else 0.0
        if self.t_hold < t < self.t_shake:
            ts = t - self.t_hold
            e, de = _envelope(ts, self.shake_s, self.ramp)
            x += self.ax * e * math.sin(self.wx * ts)
            vx += self.ax * (de * math.sin(self.wx * ts) + e * self.wx * math.cos(self.wx * ts))
            z += self.az * e * math.sin(self.wz * ts)
            vz += self.az * (de * math.sin(self.wz * ts) + e * self.wz * math.cos(self.wz * ts))
        xs, vxs = self.stop_state(t)
        return x + xs, z, vx + vxs, vz

    def phase(self, t: float) -> str:
        if t < self.t_settle:
            return "close"
        if t < self.t_final:
            return "main"
        if t < self.t_power:
            for i, (t0, T1, T2, *_rest) in enumerate(self.stops):
                if t0 <= t < t0 + T1 + T2 + 0.2:
                    return f"stop{i}"
            return "stop_end"
        return "power"


def simulate(jaw: Jaw, obj: GraspObject, sc: Scenario, b1: dict, task: str, depth: float, lateral: float = 0.0,
             shake_hz: tuple[float, float] | None = None, friction_scale: float = 1.0, brake: float = 30.0,
             band_factor: float | None = None, tau: float | None = None, stops: list[float] | None = None,
             power_s: float | None = None, record: bool = False, lim: Limits | None = None) -> dict[str, Any]:
    """One run: close, settle, lift, hold, shake, hold, stops, power cut. `depth` (mm) puts a box's
    top, or a sheet's free edge, that far above the fingertip line; `lateral` (mm) moves the object
    off the jaw centreline. `brake` is N m per gripper."""
    lim = lim or Limits.from_config(b1, task, friction_scale=friction_scale)
    if band_factor is not None:
        lim.band_factor = band_factor
    tau = jaw.torque if tau is None else tau
    box = obj.kind == "box"
    L = jaw.reach
    x_face = obj.thickness * 1e3 / 2
    z_top = -L + depth
    z_bot = z_top - obj.height * 1e3
    clr = lim.clearance_box if box else lim.clearance_sheet
    plan = {}
    phi0 = jaw.phi_meet
    for side in (1, -1):
        xf = x_face + side * lateral
        app, touch = side_touch(jaw, side, xf, z_bot, z_top, clr)
        if not math.isfinite(app):
            return {"ok": False, "reason": "does not fit between the open fingers", "depth_mm": depth, "lateral_mm": lateral}
        plan[side] = (app, touch)
        phi0 = max(phi0, app)
    phi_touch = max(plan[1][1], plan[-1][1])
    b = _Builder(jaw, obj, sc, b1, depth, {1: plan[1][1], -1: plan[-1][1]}, lim.friction_scale, lim.band_factor, z_top)
    fingers = []
    for side, nm in ((1, "R"), (-1, "L")):
        xml, m_f, mom = b.finger_xml(side)
        m_f = max(m_f, 1e-3)
        I_f = max(mom, 1e-6)
        fingers.append(
            f'<body name="finger{nm}" pos="{_fmt((side * jaw.pivot_x * 1e-3, 0, 0))}" gravcomp="1">'
            f'<joint name="j{nm}" type="hinge" axis="0 {-side} 0" range="{_fmt(jaw.phi_meet)} {_fmt(jaw.phi_max + 1e-3)}" '
            f'armature="{_fmt(float(b1.get("closing", {}).get("armature_kgm2", 0.015)))}" damping="0" '
            f'solreffriction="{_fmt(2 * b.dt)} 1" solimpfriction="0.999 0.9999 0.001 0.5 2"/>'
            f'<inertial pos="0 0 {_fmt(-L * 0.5e-3)}" mass="{_fmt(m_f)}" diaginertia="{_fmt((I_f, I_f, I_f))}"/>'
            f"{xml}</body>")
    if not box or any(e.seg <= 2.0 + 1e-9 for e in b.elements):
        # Test conditions 2.16: pressure peaks converge at the fine timestep for sheets and 1 - 2 mm elements.
        b.substeps = max(b.substeps, int(round(b.dt / b.fine_dt)))
    k_servo = float(sc.sim.get("servo_kp", 2.0e6))
    m_total = 1.0 + 0.6 + obj.mass
    kv_servo = 2 * math.sqrt(k_servo * m_total)
    tc = b.contact_tc
    hinge_names: list[str] = []
    hinge_k: list[float] = []
    if box:
        H = obj.height
        z_piv = H - depth * 1e-3 + L * 1e-3
        com_z = H / 2 - obj.com_below_top
        I = max(obj.inertia, 1e-7)
        obj_xml = (
            f'<body name="obj" pos="{_fmt((lateral * 1e-3, 0, H / 2))}">'
            '<joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/>'
            '<joint name="oth" type="hinge" axis="0 1 0"/>'
            f'<inertial pos="0 0 {_fmt(com_z)}" mass="{_fmt(obj.mass)}" diaginertia="{_fmt((I, I, I))}"/>'
            f'<geom name="obj" type="box" size="{_fmt((obj.thickness / 2, obj.width / 2, H / 2))}" mass="0" '
            f'friction="{_fmt(obj.material.friction)} 0.005 0.0001" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.9 0.75 0.35 1"/>'
            "</body>")
        table = (f'<geom name="table" type="box" pos="0 0 -0.01" size="0.4 0.4 0.01" friction="{TABLE_FRICTION} 0.005 0.0001" '
                 f'priority="1" contype="{TABLE}" conaffinity="{GRIP | OBJ}" rgba="0.8 0.8 0.8 1"/>')
        equality_obj = ""
        top_body = "obj"
    else:
        z_top_w = 1.0
        z_piv = z_top_w - depth * 1e-3 + L * 1e-3
        seg = float(sc.sim.get("sheet_segment_mm", 50.0)) * 1e-3
        t_ = obj.thickness
        W = obj.width
        H = obj.height
        gsm = obj.mass / (H * W)
        first = min(depth * 1e-3 + 0.01, H)
        lengths = [first]
        rest = H - first
        while rest > 1e-6:
            s_ = min(seg, rest)
            if rest - s_ < 0.3 * seg:
                s_ = rest
            lengths.append(s_)
            rest -= s_
        D = obj.material.bending_stiffness
        xml, closes = "", ""
        for i, ln in enumerate(lengths):
            m_s = gsm * W * ln
            I_s = m_s * (ln * ln + t_ * t_) / 12
            below = sum(lengths[i:])
            if i == 0:
                xml += (f'<body name="s0" pos="{_fmt((lateral * 1e-3, 0, z_top_w))}">'
                        '<joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/>'
                        '<joint name="oth" type="hinge" axis="0 1 0"/>')
            else:
                k_h = D * W / (0.5 * (lengths[i - 1] + ln))
                m_below = gsm * W * below
                I_below = m_below * below * below / 3
                c_h = 2 * 0.1 * math.sqrt(k_h * I_below)
                xml += (f'<body name="s{i}" pos="0 0 {_fmt(-lengths[i - 1])}">'
                        f'<joint name="h{i}" type="hinge" axis="0 1 0" stiffness="{_fmt(k_h)}" damping="{_fmt(c_h)}"/>')
                hinge_names.append(f"h{i}")
                hinge_k.append(k_h)
            xml += (f'<inertial pos="0 0 {_fmt(-ln / 2)}" mass="{_fmt(m_s)}" diaginertia="{_fmt((I_s, I_s, I_s))}"/>'
                    f'<geom name="s{i}" type="box" pos="0 0 {_fmt(-ln / 2)}" size="{_fmt((t_ / 2, W / 2, ln / 2))}" mass="0" '
                    f'friction="{_fmt(obj.material.friction)} 0.005 0.0001" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.72 0.55 0.35 1"/>')
            closes += "</body>"
        obj_xml = xml + closes
        table = ""
        equality_obj = f'<weld name="holder" body1="s0" solref="{_fmt(4 * b.dt)} 1"/>'
        top_body = "s0"

    xml = f"""<mujoco model="grasp2d_batch1">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{_fmt(b.dt / b.substeps)}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10" iterations="60" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default><geom condim="3" solref="{_fmt(tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"/></default>
  <asset>{''.join(b.assets)}</asset>
  <worldbody>
    {table}
    <body name="base" pos="0 0 {_fmt(z_piv)}" gravcomp="1">
      <joint name="base_x" type="slide" axis="1 0 0"/><joint name="base_z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="1e-3 1e-3 1e-3"/>
      {''.join(fingers)}
    </body>
    {obj_xml}
  </worldbody>
  <equality>
    <joint name="gear" joint1="jL" joint2="jR" polycoef="0 1 0 0 0" solref="{_fmt(4 * b.dt)} 1"/>
    {equality_obj}
  </equality>
  <actuator>
    <position name="servo_x" joint="base_x" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <position name="servo_z" joint="base_z" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <motor name="driveR" joint="jR" gear="1"/>
    <motor name="driveL" joint="jL" gear="1"/>
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    # Otherwise MuJoCo resets a blown-up run to t = 0 from qpos0 and carries on, with finite
    # accelerations, so a run can loop forever or finish with a meaningless verdict.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    data = mujoco.MjData(model)
    return _run(model, data, jaw, obj, sc, b1, lim, b, task, depth, lateral, phi0, phi_touch, tau, brake, shake_hz, stops,
                power_s, record, hinge_names, hinge_k, top_body, box, plan)


def _run(model, data, jaw: Jaw, obj: GraspObject, sc: Scenario, b1: dict, lim: Limits, b: _Builder, task: str, depth: float,
         lateral: float, phi0: float, phi_touch: float, tau: float, brake: float, shake_hz, stops, power_s, record: bool,
         hinge_names: list[str], hinge_k: list[float], top_body: str, box: bool, plan) -> dict[str, Any]:
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    vadr = lambda n: model.jnt_dofadr[jid(n)]
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    data.qpos[qadr("jR")] = phi0
    data.qpos[qadr("jL")] = phi0
    mujoco.mj_forward(model, data)
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "holder") if not box else -1
    els = b.elements
    n_el = len(els)
    el_geom = np.array([gid(e.geom) for e in els], dtype=int)
    el_of_geom = np.full(model.ngeom, -1, dtype=int)
    el_of_geom[el_geom] = np.arange(n_el)
    el_side = np.array([e.side for e in els])
    el_kind = np.array([e.kind for e in els])
    el_face = np.array([e.face for e in els])
    el_grip = el_kind != "bare"
    el_area = np.array([e.area for e in els])
    el_ext = np.array([e.ext for e in els]) * 1e-3
    el_w = np.array([e.w for e in els])
    el_qadr = np.array([qadr(e.joint) for e in els], dtype=int)
    el_lo = np.array([model.jnt_range[jid(e.joint), 0] for e in els])
    el_k = np.array([model.jnt_stiffness[jid(e.joint)] for e in els])
    el_q0 = np.array([model.qpos_spring[qadr(e.joint)] for e in els])
    el_pad = el_kind == "pad"
    body_side = np.zeros(model.ngeom, dtype=int)
    for gname, s in b.body_geoms.items():
        body_side[gid(gname)] = s
    unmarked_side = np.zeros(model.ngeom, dtype=int)
    unmarked_side[el_geom[~el_grip]] = el_side[~el_grip]
    is_obj = (model.geom_contype & OBJ) > 0
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, top_body)
    hinge_adr = np.array([qadr(h) for h in hinge_names], dtype=int)
    hk = np.array(hinge_k)
    a_oth, a_bx, a_bz, a_jR = qadr("oth"), qadr("base_x"), qadr("base_z"), qadr("jR")
    d_jR, d_jL = vadr("jR"), vadr("jL")
    cl = b1.get("closing", {})
    w_fast, w_slow = float(cl.get("fast_rad_s", 1.0)), float(cl.get("slow_rad_s", 0.3))
    slow_from = phi_touch + math.radians(float(cl.get("slow_deg", 5.0)))
    t_close = max(phi0 - slow_from, 0.0) / w_fast + (min(phi0, slow_from) - phi_touch) / w_slow + sc.squeeze_time
    t_close = max(t_close, 0.15)
    shake_g = lim.shake_g
    hz = shake_hz or (sc.shake_lateral_hz, sc.shake_vertical_hz)
    shake_s = float(b1.get("shake_s", 3.0))
    if stops is None:
        stops = list(b1.get("stop", {}).get("decel_g", [0.5, 1.0]))
        if not jaw.symmetric:
            stops = [s for lvl in stops for s in (lvl, -lvl)]
    pc = b1.get("power_cut", {})
    power_s = float(pc.get("hold_s", 5.0)) if power_s is None else power_s
    tl = Timeline(sc, t_close, shake_g, hz, shake_s, stops, power_s)
    kp, kv = float(model.actuator_gainprm[0, 0]), float(-model.actuator_biasprm[0, 2])
    kv_drive = tau / 0.2
    # SIM round 19: the close is a speed servo that drives and brakes within its torque, as a geared drive's is;
    # `closing: {brakes: false}` keeps the push-only governor the rows before block C ran.
    brake_floor = -tau if cl.get("brakes", True) else 0.0
    dt = model.opt.timestep
    box_H = obj.height
    fill_rel = -obj.headspace if box else -math.inf
    band = lim.band_mm * 1e-3
    every = 4 * b.substeps
    n_mean = max(1, int(round(float(b1.get("friction_mean_ms", 10.0)) * 1e-3 / (every * dt))))
    fric_win: deque = deque(maxlen=n_mean)
    cone_run = {1: 0, -1: 0}

    def contacts():
        n = data.ncon
        if n == 0:
            return None
        con = data.contact
        g1, g2 = con.geom1[:n], con.geom2[:n]
        adr = con.efc_address[:n]
        ok = adr >= 0
        fn = np.where(ok, data.efc_force[np.maximum(adr, 0)], 0.0)
        return con, n, g1, g2, adr, ok, fn

    def measure():
        """Per element: load from its spring, or from the contact force at its stop (pads and marked faces;
        test conditions 2.16) and always for unmarked bare faces, which count as body contact. Per finger: body
        load, friction use, and the tangential force and friction capacity behind it."""
        c = contacts()
        Fc = np.zeros(n_el)
        body = {1: 0.0, -1: 0.0}
        fric = {1: 0.0, -1: 0.0}
        tang = {1: 0.0, -1: 0.0}
        caps = {1: 0.0, -1: 0.0}
        q = data.qpos[el_qadr]
        spring = np.where(el_grip & (q > el_lo + 2e-5), el_k * np.maximum(el_q0 - q, 0.0), np.nan)
        if c is None:
            return np.where(np.isnan(spring), 0.0, spring), Fc, body, fric, tang, caps
        con, n, g1, g2, adr, ok, fn = c
        e1, e2 = el_of_geom[g1], el_of_geom[g2]
        o1, o2 = is_obj[g1], is_obj[g2]
        s1 = (e1 >= 0) & o2 & ok
        s2 = (e2 >= 0) & o1 & ok
        np.add.at(Fc, e1[s1], fn[s1])
        np.add.at(Fc, e2[s2], fn[s2])
        F = np.where(np.isnan(spring), Fc, spring)
        for sd in (1, -1):
            bsel = (((body_side[g1] == sd) | (unmarked_side[g1] == sd)) & o2 & ok) | \
                   (((body_side[g2] == sd) | (unmarked_side[g2] == sd)) & o1 & ok)
            body[sd] = float(fn[bsel].sum())
            first = ((el_of_geom[g1] >= 0) & (np.where(el_of_geom[g1] >= 0, el_side[np.maximum(el_of_geom[g1], 0)], 0) == sd) |
                     (body_side[g1] == sd)) & o2 & ok
            second = ((el_of_geom[g2] >= 0) & (np.where(el_of_geom[g2] >= 0, el_side[np.maximum(el_of_geom[g2], 0)], 0) == sd) |
                      (body_side[g2] == sd)) & o1 & ok
            sel = first | second
            if sel.any():
                a = adr[sel]
                fr = con.frame[:n][sel]
                sign = np.where(first[sel], 1.0, -1.0)[:, None]
                tw = sign * (data.efc_force[a + 1][:, None] * fr[:, 3:6] + data.efc_force[a + 2][:, None] * fr[:, 6:9])
                mu = con.friction[:n][sel, 0]
                cap = float(np.sum(mu * fn[sel]))
                tang[sd], caps[sd] = float(np.linalg.norm(tw.sum(axis=0))), cap
                if cap > 1e-6:
                    fric[sd] = tang[sd] / cap
        return F, Fc, body, fric, tang, caps

    def zones(F: np.ndarray):
        """Per finger: band force, panel force, bed pressure (below the fill line only), all-grip pressure."""
        pos = data.geom_xpos[el_geom]
        R = data.xmat[obj_body].reshape(3, 3)
        rel = (pos - data.xpos[obj_body]) @ R
        out = {}
        if box:
            z_rel = rel[:, 2] - box_H / 2
            lo, hi = z_rel - el_ext / 2, z_rel + el_ext / 2
            span = np.maximum(hi - lo, 1e-6)
            a_band = np.clip((hi - (-band)) / span, 0.0, 1.0)
            a_fill = np.clip((hi - fill_rel) / span, 0.0, 1.0)
            p = F / el_area
            for sd in (1, -1):
                m = (el_side == sd) & el_grip
                fb = float(np.sum(F[m] * a_band[m]))
                fp = float(np.sum(F[m] * np.maximum(a_fill[m] - a_band[m], 0.0)))
                bed = m & (a_fill < 0.5)
                pb = float(p[bed].max()) if bed.any() else 0.0
                out[sd] = (fb, fp, pb, float(p[m].max()) if m.any() else 0.0)
        else:
            p = F / el_area
            for sd in (1, -1):
                m = (el_side == sd) & el_grip
                pm = float(p[m].max()) if m.any() else 0.0
                out[sd] = (0.0, 0.0, pm, pm)
        return out

    ref = None
    p_grip_local = None
    peaks = {k: 0.0 for k in ("band", "panel", "bed", "press", "body", "fric", "slip", "rot", "crease",
                              "i_band", "i_panel", "i_bed", "i_body")}
    diag = {"fric_sample": 0.0, "cone_n": 0, "press_contact": 0.0}
    stop_rec: list[dict[str, float]] = []
    stop_ref = None
    stop_peak = None
    power_ref = None
    power_slip = 0.0
    power_rot = 0.0
    brake_use = 0.0
    brake_net: list[float] = []
    brake_on = None
    bottomed = 0
    surf_load = {}
    released = False
    unstable, unstable_s = False, math.nan
    first_contact = contact_w = math.nan
    grip_settle = 0.0
    torque_hold: list[float] = []
    torque_peak = 0.0
    b_exit = None
    dropped_in = ""
    trace: dict[str, list[float]] = {k: [] for k in ("t", "phi", "slip", "grip", "band", "panel", "body", "fric", "press")} if record else {}

    def grip_rel() -> np.ndarray:
        R = data.xmat[obj_body].reshape(3, 3)
        pw = data.xpos[obj_body] + R @ p_grip_local
        return np.array([pw[0] - data.qpos[a_bx], pw[2] - (data.qpos[a_bz])])

    brake_motor = brake / 2.0
    phase = "close"
    snap = None
    stop_dropped = ""
    i = -1
    while data.time < tl.t_end:
        i += 1
        t = data.time
        if snap is None and tl.stops and t >= tl.t_final:
            snap = copy.copy(data)
        phi = float(data.qpos[a_jR])
        if t < tl.t_close:
            om = -float(data.qvel[d_jR])
            target = w_fast if phi > slow_from else w_slow
            u = min(max(kv_drive * (target - om), brake_floor), tau)
        elif t < tl.t_power:
            u = tau
        else:
            u = 0.0
        data.ctrl[2] = -u
        data.ctrl[3] = -u
        x, z, vx, vz = tl(t)
        data.ctrl[0] = x + kv / kp * vx
        data.ctrl[1] = z + kv / kp * vz
        if eq >= 0 and not released and t >= tl.t_close:
            data.eq_active[eq] = 0
            released = True
        # On the drive's clock, not the sampled phase, which reads "power" one step early.
        if t >= tl.t_power and brake_on is None:
            model.dof_frictionloss[d_jR] = brake_motor
            model.dof_frictionloss[d_jL] = brake_motor
            brake_on = t
        t_step = data.time
        mujoco.mj_step(model, data)
        if (data.time < t_step or any(data.warning[w].number for w in BAD_STATE)
                or not np.isfinite(data.qacc).all() or abs(data.qpos[a_jR]) > 3.0):
            unstable = True
            unstable_s = t_step
            break
        if i % every:
            continue
        t = data.time
        phase = tl.phase(t)
        F, Fc, body, fric, tang, caps = measure()
        if not math.isfinite(first_contact) and Fc.sum() + body[1] + body[-1] > 1e-3:
            first_contact, contact_w = float(data.qpos[a_jR]), -float(data.qvel[d_jR])
        zn = zones(F)
        if phase == "close":
            for sd in (1, -1):
                peaks["i_band"] = max(peaks["i_band"], zn[sd][0])
                peaks["i_panel"] = max(peaks["i_panel"], zn[sd][1])
                peaks["i_bed"] = max(peaks["i_bed"], zn[sd][2])
                peaks["i_body"] = max(peaks["i_body"], body[sd])
            if t >= tl.t_settle - 0.02 - every * dt and ref is None:
                pass
            continue
        if ref is None:
            loaded = F > 1e-3
            zs = data.geom_xpos[el_geom[loaded], 2] if loaded.any() else np.array([data.xpos[obj_body][2]])
            zg = float(np.average(zs, weights=F[loaded])) if loaded.any() else float(zs[0])
            xg = float(data.xpos[obj_body][0])
            R = data.xmat[obj_body].reshape(3, 3)
            p_grip_local = R.T @ (np.array([xg, 0.0, zg]) - data.xpos[obj_body])
            ref = grip_rel()
            ref_th = float(data.qpos[a_oth])
            grip_settle = float(F[(el_side == 1) & el_grip].sum())
            surf_load = {k: float(F[(el_side == 1) & (el_kind == k)].sum()) for k in ("pad", "marked", "bare")}
            surf_load["body"] = body[1] - surf_load["bare"]
            if loaded.any():
                lowest = int(np.argmin(np.where(loaded & el_grip, data.geom_xpos[el_geom, 2], np.inf)))
                b_exit = float(el_w[lowest])
        rel = grip_rel()
        slip = float(np.hypot(*(rel - ref)))
        rot = abs(float(data.qpos[a_oth]) - ref_th)
        act = abs(float(data.actuator_force[2])) + abs(float(data.actuator_force[3]))
        torque_peak = max(torque_peak, act)
        if record:
            trace["t"].append(t), trace["phi"].append(phi), trace["slip"].append(slip)
            trace["grip"].append(float(F[(el_side == 1) & el_grip].sum())), trace["band"].append(zn[1][0])
            trace["panel"].append(zn[1][1]), trace["body"].append(max(body.values())), trace["fric"].append(max(fric.values()))
            trace["press"].append(max(zn[1][3], zn[-1][3]))
        if phase == "main":
            if t > tl.t_lift:
                torque_hold.append(act)
            for sd in (1, -1):
                peaks["band"] = max(peaks["band"], zn[sd][0])
                peaks["panel"] = max(peaks["panel"], zn[sd][1])
                peaks["bed"] = max(peaks["bed"], zn[sd][2])
                peaks["press"] = max(peaks["press"], zn[sd][3])
                peaks["body"] = max(peaks["body"], body[sd])
                diag["fric_sample"] = max(diag["fric_sample"], fric[sd])
                cone_run[sd] = cone_run[sd] + 1 if fric[sd] >= 0.99 else 0
                diag["cone_n"] = max(diag["cone_n"], cone_run[sd])
            cap_both = caps[1] + caps[-1]
            fric_win.append((tang[1] + tang[-1]) / cap_both if cap_both > 1e-6 else 0.0)
            if len(fric_win) == n_mean:
                peaks["fric"] = max(peaks["fric"], sum(fric_win) / n_mean)
            gc = el_grip & (Fc > 0.0)
            if gc.any():
                diag["press_contact"] = max(diag["press_contact"], float((Fc[gc] / el_area[gc]).max()))
            peaks["slip"] = max(peaks["slip"], slip)
            peaks["rot"] = max(peaks["rot"], rot)
            at_stop = el_pad & (F > 0.01) & (data.qpos[el_qadr] <= el_lo + 2e-5)
            if at_stop.any():
                bottomed = max(bottomed, int(at_stop.sum()))
            if len(hinge_adr):
                peaks["crease"] = max(peaks["crease"], float(np.max(np.abs(hk * data.qpos[hinge_adr]))))
            if slip > DROP:
                dropped_in = "main"
                break
        elif phase.startswith("stop"):
            if phase != "stop_end":
                k = int(phase[4:])
                if stop_ref is None or stop_ref[0] != k:
                    stop_ref = (k, slip, rot)
                    stop_peak = {"level_g": tl.stops[k][6], "slip_mm": 0.0, "rot_deg": 0.0, "band": 0.0, "panel": 0.0, "bed": 0.0}
                    stop_rec.append(stop_peak)
                stop_peak["slip_mm"] = max(stop_peak["slip_mm"], (slip - stop_ref[1]) * 1e3)
                stop_peak["rot_deg"] = max(stop_peak["rot_deg"], math.degrees(abs(rot - stop_ref[2])))
                for sd in (1, -1):
                    fb, fp, pb, _ = zn[sd]
                    stop_peak["band"] = max(stop_peak["band"], fb)
                    stop_peak["panel"] = max(stop_peak["panel"], fp)
                    stop_peak["bed"] = max(stop_peak["bed"], pb / 1e3)
            if slip > DROP:
                # The stop dropped it: report that level, then run the power cut from the end of the
                # shake, as if the stops had not been run.
                stop_dropped = phase
                if snap is None:
                    dropped_in = phase
                    break
                mujoco.mj_copyData(data, model, snap)
                tl = Timeline(sc, t_close, shake_g, hz, shake_s, [], power_s)
                continue
        elif phase == "power":
            if power_ref is None:
                power_ref = (slip, rot)
            ne = data.nefc
            if ne:
                fr = data.efc_type[:ne] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
                f = data.efc_force[:ne]
                # Net over both motors: the gear couples the fingers, so the two frictions can
                # fight each other and their magnitudes overstate what the brake holds.
                net = abs(float(f[fr & (data.efc_id[:ne] == d_jR)].sum() + f[fr & (data.efc_id[:ne] == d_jL)].sum()))
                brake_net.append(net)
                brake_use = max(brake_use, net)
            power_slip = max(power_slip, slip - power_ref[0])
            power_rot = max(power_rot, abs(rot - power_ref[1]))
            if slip > DROP:
                dropped_in = "power"
                break

    row: dict[str, Any] = {"ok": not unstable and ref is not None, "depth_mm": depth, "lateral_mm": lateral, "task": task,
                           "friction_scale": lim.friction_scale, "band_factor": lim.band_factor, "brake_Nm": brake,
                           "shake_hz": hz[0], "elements": n_el, "elements_R": int(np.sum(el_side == 1)),
                           "approach_opening_deg": math.degrees(phi0 - jaw.phi_meet),
                           "touch_plan_deg": math.degrees(phi_touch - jaw.phi_meet),
                           "contact_angle_deg": math.degrees(first_contact - jaw.phi_meet) if math.isfinite(first_contact) else math.nan,
                           "contact_rad_s": contact_w, "close_brakes": brake_floor < 0.0,
                           "close_s": tl.t_close, "finger_torque_Nm": tau,
                           "mount_fatigue_flag": bool(tau >= float(b1.get("mount_fatigue_Nm_per_finger", 4.0)) - 1e-9),
                           "substeps": b.substeps, "reset_flag": bool(unstable)}
    if unstable:
        # Test conditions 2.15: a run that blows up is flagged and rerun, never scored as a fail.
        row.update(reason=f"simulation went unstable at {unstable_s:.4f} s in {phase}; flagged, not scored",
                   unstable_s=unstable_s, unstable_phase=phase, passed=None, score=None, limiting=None)
        return row
    if ref is None:
        row.update(reason="never reached the lift")
        return row
    held_main = dropped_in != "main"
    body_lim = float(b1.get("body_contact_limit_N", 1.0))

    def score_peaks(pk: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        margins = {
            "slip": (sc.max_slip / max(pk["slip"], 1e-9)) if held_main else 0.0,
            "rotation": sc.max_rotation / max(pk["rot"], 1e-9),
        }
        demand = {"slip": pk["slip"] * 1e3, "rotation": math.degrees(pk["rot"]), "friction": pk["fric"]}
        if task in ("T1", "T2", "T3") and pk["body"] > 0.1 * body_lim:
            margins["body"] = body_lim / pk["body"]
            demand["body"] = pk["body"]
        if box:
            for k, lim_v, key in (("band", lim.band_force, "band"), ("panel", lim.panel_force, "panel"), ("bed", lim.bed_pressure, "pressure")):
                if pk[k] > 1e-6:
                    margins[key] = lim_v / (pk[k] * (sc.pressure_sf if key == "pressure" else 1.0))
                    demand[key] = pk[k] / (1e3 if key == "pressure" else 1.0)
                ik = "i_" + k
                if pk[ik] > 1e-6:
                    margins["impact_" + key] = lim_v / (pk[ik] * (sc.pressure_sf if key == "pressure" else 1.0))
                    demand["impact_" + key] = pk[ik] / (1e3 if key == "pressure" else 1.0)
            if task in ("T1", "T2", "T3") and pk["i_body"] > 0.1 * body_lim:
                margins["impact_body"] = body_lim / pk["i_body"]
                demand["impact_body"] = pk["i_body"]
        else:
            margins["pressure"] = obj.material.allow_pressure / (sc.pressure_sf * max(pk["press"], 1e-9))
            demand["pressure"] = pk["press"] / 1e3
            if len(hinge_adr):
                bw = b_exit if (b_exit and sc.crease_width == "pad") else obj.width
                cap = obj.material.bending_strength * bw
                margins["crease"] = cap / (sc.crease_safety_factor * max(pk["crease"], 1e-9))
                demand["crease"] = pk["crease"]
                row["crease_width_mm"] = bw * 1e3
        return {k: float(min(v, 99.0)) for k, v in margins.items()}, demand

    margins, demand = score_peaks(peaks)
    # Test conditions 2.16: the friction x0.8 run is the hold safety factor, so it passes on holding alone.
    x08 = lim.friction_scale < 1.0 - 1e-9
    rule = {k: margins[k] for k in ("slip", "rotation")} if x08 else margins
    worst = min(rule, key=rule.get)
    passed = rule[worst] >= 1.0 and held_main
    sample_ms = every * dt * 1e3
    held_levels = []
    for rec in stop_rec:
        ok_lvl = rec["slip_mm"] <= sc.max_slip * 1e3 and math.radians(rec["rot_deg"]) <= sc.max_rotation
        if box:
            ok_lvl = ok_lvl and rec["band"] <= lim.band_force and rec["panel"] <= lim.panel_force and rec["bed"] * 1e3 <= lim.bed_pressure
        rec["held"] = bool(ok_lvl)
        if ok_lvl:
            held_levels.append(abs(rec["level_g"]))
    stop_max = 0.0
    for rec in stop_rec:
        if rec["held"]:
            stop_max = max(stop_max, abs(rec["level_g"]))
        else:
            break
    hold_torque = float(np.mean(torque_hold)) if torque_hold else 2 * tau
    drive = b1.get("drive_Nm_per_gripper", {"continuous": [7.0, 10.0], "peak": 25.0})
    row.update({
        "passed": bool(passed), "reason": "" if passed else ("dropped" if not held_main else f"{worst} margin {margins[worst]:.2f}"),
        "score": margins[worst] if held_main else 0.0, "limiting": worst if held_main else "slip",
        "margins": margins, "demand": demand,
        "slip_mm": peaks["slip"] * 1e3, "rotation_deg": math.degrees(peaks["rot"]), "friction_use": peaks["fric"],
        "band_N": peaks["band"], "panel_N": peaks["panel"], "bed_kPa": peaks["bed"] / 1e3, "pressure_kPa": peaks["press"] / 1e3,
        "body_N": peaks["body"], "impact_band_N": peaks["i_band"], "impact_panel_N": peaks["i_panel"],
        "impact_bed_kPa": peaks["i_bed"] / 1e3, "impact_body_N": peaks["i_body"], "grip_N": grip_settle,
        "humid_band": lim.band_force * lim.humid / peaks["band"] if peaks["band"] > 1e-6 else 99.0,
        "humid_panel": lim.panel_force * lim.humid / peaks["panel"] if peaks["panel"] > 1e-6 else 99.0,
        "stops": stop_rec, "stop_max_g": stop_max, "stop_dropped_in": stop_dropped, "dropped_in": dropped_in,
        "power_cut_held": bool(dropped_in == "" and power_ref is not None and power_slip <= sc.max_slip),
        "power_cut_slip_mm": power_slip * 1e3,
        "drive_hold_Nm": hold_torque, "drive_peak_Nm": torque_peak,
        "drive_cont_margin": drive["continuous"][1] / max(hold_torque, 1e-9),
        "drive_over_flag": bool(hold_torque > drive["continuous"][0]),
        "drive_peak_margin": drive["peak"] / max(torque_peak, 1e-9),
        "brake_use_Nm": float(np.median(brake_net[len(brake_net) // 2:])) if brake_net else 0.0,
        "brake_peak_Nm": brake_use,
        "brake_on_s": brake_on if brake_on is not None else math.nan, "power_cut_s": tl.t_power,
        "pass_rule": "holds: slip and rotation" if x08 else "all margins",
        "friction_use_sample_peak": diag["fric_sample"], "friction_cone_ms": diag["cone_n"] * sample_ms,
        "pressure_kPa_contact_sample": diag["press_contact"] / 1e3,
        "pad_elements_at_stop": bottomed,
        "load_by_surface_N": surf_load,
        "mu_pad": obj.material.friction * lim.friction_scale,
        "mu_bare": obj.material.bare_friction * lim.friction_scale,
        "element_faces": {f: int(np.sum((el_side == 1) & (el_face == f))) for f in dict.fromkeys(el_face[el_side == 1].tolist())},
        "element_pitch_mm": {"long": b.pitch, "short_or_curved": b.fine},
    })
    if record:
        row["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    return row
