"""Level 1: planar MuJoCo check of one design on one object.

The gripper plane is the MuJoCo x-z plane (y out of plane, gravity -z).

* Gripper base: x and z slide joints driven by stiff position servos along
  the lift-and-shake trajectory; gripper bodies are gravity-compensated so the
  servos only carry the object.
* Fingers: hinges at the pivots, synced by a joint equality, each driven by a
  constant torque (the 1-DoF drive) with damping that sets the free closing
  speed. The drive is backdrivable: nothing but the torque holds the jaw shut.
* Pads: each face is split into short box elements on spring slide joints
  along the face normal (a discretised Winkler foundation, pad and object
  surface in series), all riding on one shear slide joint per pad (elastomer
  shear compliance). Self-aligning pads add a sprung, range-limited hinge.
  Element armature sets the element frequency; contacts are made several
  times stiffer than the element springs so the springs carry the compliance.
* Box: rigid body (x, z, pitch) standing on a table.
* Sheet: hinged chain of rigid segments with hinge stiffness D*W/l, held in a
  holder by a weld that is released after the jaw closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .analytic import solve_grip
from .geometry import Finger, Pad, build_finger
from .objects import G, GraspObject
from .pads import foundation_stiffness, series, shear_stiffness
from .scenario import Scenario
from .schema import GripperDesign

GRIP, OBJ, TABLE = 1, 2, 4
ALU_DENSITY = 2700.0
PAD_DENSITY = 1200.0
FINGER_FRICTION = 0.3
TABLE_FRICTION = 0.4
DROP_DISTANCE = 0.02  # m of slip counted as a drop


@dataclass
class _Element:
    joint: str
    k: float  # N/m
    area: float  # m^2
    finger: int  # +1 right, -1 left
    pad: str


@dataclass
class SimResult:
    design: GripperDesign
    object_id: str
    passed: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    insertion_mm: float = 0.0

    @property
    def score(self) -> float:
        """Smallest pass-criterion margin (>= 1 passes)."""
        m = self.metrics.get("margins", {})
        return float(min(m.values())) if m else 0.0


def _fmt(v: Any) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(f"{float(x):.6g}" for x in v)
    return f"{float(v):.6g}"


def _beta(dX: float, dZ: float) -> float:
    """Rotation about +y that takes a geom's local x-axis to (dX, dZ)."""
    return math.atan2(-dZ, dX)


class _Builder:
    def __init__(self, f: Finger, obj: GraspObject, sc: Scenario, depth: float):
        self.f, self.obj, self.sc, self.depth = f, obj, sc, depth
        sim = sc.sim
        self.dt = float(sim.get("timestep_s", 1e-4))
        self.pitch = float(sim.get("pad_element_pitch_mm", 4.0))
        self.f_el = float(sim.get("element_frequency_hz", 250.0))
        self.zeta = float(sim.get("element_damping_ratio", 0.3))
        self.contact_tc = float(sim.get("contact_timeconst_s", 2 * self.dt))
        self.elements: list[_Element] = []
        self.k_obj = obj.material.k_surface
        self.mu = obj.material.friction * sc.friction_scale
        self.b = f.design.pad_width * 1e-3

    # finger-local (u, v) in mm -> finger body frame (X, Z) in m
    def p(self, u: float, v: float, side: int) -> tuple[float, float]:
        return (-side * v * 1e-3, -u * 1e-3)

    def d(self, du: float, dv: float, side: int) -> tuple[float, float]:
        return (-side * dv, -du)

    def armature(self, k: float, mass: float) -> float:
        w = 2 * math.pi * self.f_el
        return max(k / (w * w) - mass, 0.0)

    def finger_xml(self, side: int) -> str:
        f, d = self.f, self.f.design
        name = "R" if side > 0 else "L"
        out = []
        for piece in f.body:
            if piece.rect is not None:
                cu, cv, ang, hl, ht = piece.rect
                X, Z = self.p(cu, cv, side)
                dX, dZ = self.d(math.cos(ang), math.sin(ang), side)
                out.append(
                    f'<geom name="{name}_{piece.name}" type="box" pos="{_fmt((X, 0, Z))}" euler="0 {_fmt(_beta(dX, dZ))} 0" '
                    f'size="{_fmt((hl * 1e-3, self.b / 2, ht * 1e-3))}" density="{ALU_DENSITY}" friction="{FINGER_FRICTION} 0.005 0.0001" '
                    f'priority="1" contype="{GRIP}" conaffinity="{OBJ | TABLE}" rgba="0.55 0.6 0.68 1"/>'
                )
        for pad in f.pads:
            out.append(self.mount_xml(pad, side, name))
            out.append(self.pad_xml(pad, side, name))
        return "\n".join(out)

    def mount_xml(self, pad: Pad, side: int, name: str) -> str:
        f = self.f
        v_f = max(f.design.crank_offset, 0.0) + f.design.finger_thickness / 2
        if pad.pivot_depth is not None:
            pv = pad.pivot
            w = max(0.3 * pad.length, 6.0)
            lo, hi = v_f - 1.0, pv[1] + 2.0
            if hi - lo < 1.0:
                return ""  # the pin sits in the finger beam, which carries it
            X, Z = self.p(pv[0], (lo + hi) / 2, side)
            dX, dZ = self.d(1.0, 0.0, side)
            size = (w / 2 * 1e-3, self.b / 2, (hi - lo) / 2 * 1e-3)
        else:
            t, n = pad.tangent, pad.normal
            back = pad.centre - pad.thickness * n
            # Mount block aligned with the pad: from the finger face up to the pad back.
            depth = max(float(np.dot(back - np.array([pad.face_u, v_f - 1.0]), n)), 1.0)
            c = back - (depth / 2) * n
            X, Z = self.p(c[0], c[1], side)
            dX, dZ = self.d(t[0], t[1], side)
            size = (pad.length / 2 * 1e-3, self.b / 2, depth / 2 * 1e-3)
        return (
            f'<geom name="{name}_mount_{pad.name}" type="box" pos="{_fmt((X, 0, Z))}" euler="0 {_fmt(_beta(dX, dZ))} 0" '
            f'size="{_fmt(size)}" density="{ALU_DENSITY}" friction="{FINGER_FRICTION} 0.005 0.0001" priority="1" '
            f'contype="{GRIP}" conaffinity="{OBJ | TABLE}" rgba="0.45 0.5 0.58 1"/>'
        )

    def pad_xml(self, pad: Pad, side: int, name: str) -> str:
        d = self.f.design
        n_el = max(int(round(pad.length / self.pitch)), 3)
        pts, normals, ds, thick = pad.face_samples(2 * n_el + 1)
        centres = pts[1::2]
        nrm = normals[1::2]
        th = thick[1::2]
        seg = np.full(n_el, pad.length / n_el)
        if pad.crowned:
            seg = np.full(n_el, float(np.sum(ds)) / n_el)
        e_h = 0.4  # mm, half-thickness of an element geom
        pname = f"{name}_{pad.name}"
        # Shear joint (and pivot) frame: rotation-free child of the finger.
        sh_k = shear_stiffness(pad.shore, pad.thickness * 1e-3) * pad.length * 1e-3 * self.b
        pad_mass = PAD_DENSITY * pad.length * pad.thickness * 1e-6 * self.b
        tX, tZ = self.d(pad.tangent[0], pad.tangent[1], side)
        parts = []
        open_tags = []
        if pad.pivot_depth is not None:
            pv = pad.pivot
            X, Z = self.p(pv[0], pv[1], side)
            r = pad.pivot_range
            I = pad_mass * ((pad.length * 1e-3) ** 2 / 12 + (pad.pivot_depth * 1e-3) ** 2)
            arm = max(pad.pivot_stiffness / (2 * math.pi * self.f_el) ** 2 - I, 0.0)
            damp = 2 * 0.3 * math.sqrt(max(pad.pivot_stiffness, 1e-4) * (I + arm))
            parts.append(
                f'<body name="{pname}_pivot" pos="{_fmt((X, 0, Z))}" gravcomp="1">'
                f'<joint name="{pname}_pivot" type="hinge" axis="0 {-side} 0" stiffness="{_fmt(pad.pivot_stiffness)}" '
                f'damping="{_fmt(damp)}" armature="{_fmt(arm)}" range="{_fmt(-r)} {_fmt(r)}"/>'
                f'<inertial pos="0 0 0" mass="{_fmt(pad_mass)}" diaginertia="{_fmt((I, I, I))}"/>'
            )
            open_tags.append("</body>")
            origin = (pv[0], pv[1])
        else:
            origin = (0.0, 0.0)
        oX, oZ = self.p(origin[0], origin[1], side)
        sh_arm = self.armature(sh_k, pad_mass)
        sh_damp = 2 * self.zeta * math.sqrt(sh_k * (pad_mass + sh_arm))
        parts.append(
            f'<body name="{pname}_shear" pos="0 0 0" gravcomp="1">'
            f'<joint name="{pname}_shear" type="slide" axis="{_fmt((tX, 0, tZ))}" stiffness="{_fmt(sh_k)}" '
            f'damping="{_fmt(sh_damp)}" armature="{_fmt(sh_arm)}" range="{_fmt(-pad.thickness * 0.5e-3)} {_fmt(pad.thickness * 0.5e-3)}"/>'
            f'<inertial pos="0 0 0" mass="{_fmt(pad_mass * 0.5)}" diaginertia="1e-7 1e-7 1e-7"/>'
        )
        open_tags.append("</body>")
        for j in range(n_el):
            cu, cv = centres[j]
            nu, nv = nrm[j]
            k_area = series(foundation_stiffness(pad.shore, pad.length * 1e-3, self.b, th[j] * 1e-3), self.k_obj)
            area = self.b * seg[j] * 1e-3
            k = k_area * area
            m_el = PAD_DENSITY * area * 2 * e_h * 1e-3
            arm = self.armature(k, m_el)
            damp = 2 * self.zeta * math.sqrt(k * (m_el + arm))
            gc = np.array([cu, cv]) - e_h * np.array([nu, nv])
            X, Z = self.p(gc[0], gc[1], side)
            X, Z = X - oX, Z - oZ
            nX, nZ = self.d(nu, nv, side)
            tX_, tZ_ = self.d(nv, -nu, side)
            jn = f"{pname}_e{j}"
            lo = -0.85 * th[j] * 1e-3
            parts.append(
                f'<body name="{jn}" pos="{_fmt((X, 0, Z))}" gravcomp="1">'
                f'<joint name="{jn}" type="slide" axis="{_fmt((nX, 0, nZ))}" stiffness="{_fmt(k)}" damping="{_fmt(damp)}" '
                f'armature="{_fmt(arm)}" range="{_fmt(lo)} 0.0002"/>'
                f'<geom name="{jn}" type="box" euler="0 {_fmt(_beta(tX_, tZ_))} 0" size="{_fmt((seg[j] / 2 * 1e-3, self.b / 2, e_h * 1e-3))}" '
                f'mass="{_fmt(m_el)}" friction="{_fmt(self.mu)} 0.005 0.0001" priority="2" contype="{GRIP}" conaffinity="{OBJ | TABLE}" '
                f'solref="{_fmt(self.contact_tc)} 1" rgba="0.15 0.15 0.15 1"/>'
                f"</body>"
            )
            self.elements.append(_Element(jn, k, area, side, pad.name))
        return "".join(parts) + "".join(reversed(open_tags))


def build_model(design: GripperDesign, obj: GraspObject, sc: Scenario, depth: float, finger: Finger | None = None) -> tuple[str, dict[str, Any]]:
    """MJCF text for one design/object pair, plus bookkeeping for the run."""
    f = finger or build_finger(design)
    b = _Builder(f, obj, sc, depth)
    d = design
    L = d.finger_reach * 1e-3
    phi0 = math.radians(d.max_opening)
    tau = d.finger_torque
    omega = sc.closing_speed
    damping = tau / omega
    lo, hi = sc.closed_limit, phi0 + 1e-3

    if obj.kind == "box":
        H = obj.height
        z_top = H
    else:
        H = obj.height
        z_top = 1.0
    z_piv = z_top - depth + L

    fingers = []
    for side, nm in ((1, "R"), (-1, "L")):
        body = b.finger_xml(side)
        fingers.append(
            f'<body name="finger{nm}" pos="{_fmt((side * d.pivot_spacing / 2e3, 0, 0))}" gravcomp="1">'
            f'<joint name="j{nm}" type="hinge" axis="0 {-side} 0" range="{_fmt(lo)} {_fmt(hi)}" damping="{_fmt(damping)}"/>'
            f"{body}</body>"
        )

    k_servo = float(sc.sim.get("servo_kp", 2.0e6))
    m_total = 1.0 + 0.6 + obj.mass
    kv_servo = 2 * math.sqrt(k_servo * m_total)

    tc = b.contact_tc
    if obj.kind == "box":
        com_z = H / 2 - obj.com_below_top
        I = max(obj.inertia, 1e-7)
        obj_xml = (
            f'<body name="obj" pos="0 0 {_fmt(H / 2)}">'
            '<joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/>'
            '<joint name="oth" type="hinge" axis="0 1 0"/>'
            f'<inertial pos="0 0 {_fmt(com_z)}" mass="{_fmt(obj.mass)}" diaginertia="{_fmt((I, I, I))}"/>'
            f'<geom name="obj" type="box" size="{_fmt((obj.thickness / 2, obj.width / 2, H / 2))}" mass="0" '
            f'friction="{_fmt(obj.material.friction)} 0.005 0.0001" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.9 0.75 0.35 1"/>'
            "</body>"
        )
        table = (
            f'<geom name="table" type="box" pos="0 0 -0.01" size="0.4 0.4 0.01" friction="{TABLE_FRICTION} 0.005 0.0001" '
            f'priority="1" contype="{TABLE}" conaffinity="{GRIP | OBJ}" rgba="0.8 0.8 0.8 1"/>'
        )
        equality_obj = ""
        hinge_names: list[str] = []
        hinge_k: list[float] = []
        top_body = "obj"
    else:
        seg = float(sc.sim.get("sheet_segment_mm", 50.0)) * 1e-3
        t = obj.thickness
        W = obj.width
        gsm = obj.mass / (obj.height * W)
        first = min(depth + 0.01, H)
        lengths = [first]
        rest = H - first
        while rest > 1e-6:
            s_ = min(seg, rest)
            if rest - s_ < 0.3 * seg:
                s_ = rest
            lengths.append(s_)
            rest -= s_
        D = obj.material.bending_stiffness
        xml = ""
        closes = ""
        hinge_names, hinge_k = [], []
        z = z_top
        for i, ln in enumerate(lengths):
            m_s = gsm * W * ln
            I_s = m_s * (ln * ln + t * t) / 12
            below = sum(lengths[i:])
            if i == 0:
                xml += (
                    f'<body name="s0" pos="0 0 {_fmt(z)}">'
                    '<joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/>'
                    '<joint name="oth" type="hinge" axis="0 1 0"/>'
                )
            else:
                k_h = D * W / (0.5 * (lengths[i - 1] + ln))
                m_below = gsm * W * below
                I_below = m_below * below * below / 3
                c_h = 2 * 0.1 * math.sqrt(k_h * I_below)
                xml += (
                    f'<body name="s{i}" pos="0 0 {_fmt(-lengths[i - 1])}">'
                    f'<joint name="h{i}" type="hinge" axis="0 1 0" stiffness="{_fmt(k_h)}" damping="{_fmt(c_h)}"/>'
                )
                hinge_names.append(f"h{i}")
                hinge_k.append(k_h)
            xml += (
                f'<inertial pos="0 0 {_fmt(-ln / 2)}" mass="{_fmt(m_s)}" diaginertia="{_fmt((I_s, I_s, I_s))}"/>'
                f'<geom name="s{i}" type="box" pos="0 0 {_fmt(-ln / 2)}" size="{_fmt((t / 2, W / 2, ln / 2))}" mass="0" '
                f'friction="{_fmt(obj.material.friction)} 0.005 0.0001" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.72 0.55 0.35 1"/>'
            )
            closes += "</body>"
        obj_xml = xml + closes
        table = ""
        equality_obj = f'<weld name="holder" body1="s0" solref="{_fmt(4 * b.dt)} 1"/>'
        top_body = "s0"

    xml = f"""<mujoco model="grasp2d">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_fmt(b.dt)}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10" iterations="60" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default>
    <geom condim="3" solref="{_fmt(tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"/>
  </default>
  <worldbody>
    {table}
    <body name="base" pos="0 0 {_fmt(z_piv)}" gravcomp="1">
      <joint name="base_x" type="slide" axis="1 0 0"/>
      <joint name="base_z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="1e-3 1e-3 1e-3"/>
      {''.join(fingers)}
    </body>
    {obj_xml}
  </worldbody>
  <equality>
    <joint name="sync" joint1="jL" joint2="jR" polycoef="0 1 0 0 0" solref="{_fmt(4 * b.dt)} 1"/>
    {equality_obj}
  </equality>
  <actuator>
    <position name="servo_x" joint="base_x" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <position name="servo_z" joint="base_z" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <motor name="driveR" joint="jR" gear="1"/>
    <motor name="driveL" joint="jL" gear="1"/>
  </actuator>
</mujoco>"""
    meta = {
        "elements": b.elements,
        "z_piv": z_piv,
        "z_top": z_top,
        "phi0": phi0,
        "tau": tau,
        "kv_servo": kv_servo,
        "k_servo": k_servo,
        "hinges": hinge_names,
        "hinge_k": hinge_k,
        "top_body": top_body,
    }
    return xml, meta


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


class Trajectory:
    """Gripper base offset (x, z) and velocity relative to its start pose."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.t_close = sc.close_time
        self.t_settle = self.t_close + sc.settle_time
        self.t_lift = self.t_settle + sc.lift_time
        self.t_hold = self.t_lift + sc.hold_time
        self.t_shake = self.t_hold + sc.shake_time
        self.t_end = self.t_shake + sc.final_hold_time
        self.ramp = float(sc.raw.get("sequence", {}).get("shake", {}).get("ramp_s", 0.5))
        wx = 2 * math.pi * sc.shake_lateral_hz
        wz = 2 * math.pi * sc.shake_vertical_hz
        self.wx, self.wz = wx, wz
        self.ax = sc.lateral_accel / (wx * wx) if wx > 0 else 0.0
        self.az = sc.vertical_accel / (wz * wz) if wz > 0 else 0.0

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
            e, de = _envelope(ts, sc.shake_time, self.ramp)
            x += self.ax * e * math.sin(self.wx * ts)
            vx += self.ax * (de * math.sin(self.wx * ts) + e * self.wx * math.cos(self.wx * ts))
            z += self.az * e * math.sin(self.wz * ts)
            vz += self.az * (de * math.sin(self.wz * ts) + e * self.wz * math.cos(self.wz * ts))
        return x, z, vx, vz


def _friction_use(data: mujoco.MjData, geom_side: np.ndarray, is_obj: np.ndarray) -> float:
    """Largest share of the available friction either pad uses: |sum of the
    tangential contact forces| / (mu * sum of the normal forces), over the pad
    element contacts with the object. 1 means the pad is sliding."""
    n = data.ncon
    if n == 0:
        return 0.0
    con = data.contact
    g1, g2 = con.geom1[:n], con.geom2[:n]
    s1, s2 = geom_side[g1], geom_side[g2]
    pad_first = (s1 != 0) & is_obj[g2]
    sel = (pad_first | ((s2 != 0) & is_obj[g1])) & (con.efc_address[:n] >= 0)
    if not sel.any():
        return 0.0
    adr = con.efc_address[:n][sel]
    f = data.efc_force
    fn = f[adr]
    fr = con.frame[:n][sel]
    sign = np.where(pad_first[sel], 1.0, -1.0)[:, None]
    tw = sign * (f[adr + 1][:, None] * fr[:, 3:6] + f[adr + 2][:, None] * fr[:, 6:9])
    side = np.where(pad_first[sel], s1[sel], s2[sel])
    mu = con.friction[:n][sel, 0]
    worst = 0.0
    for sd in (1, -1):
        m = side == sd
        N = float(fn[m].sum())
        if N > 1e-6:
            worst = max(worst, float(np.linalg.norm(tw[m].sum(axis=0))) / (float(mu[m].mean()) * N))
    return worst


def simulate(design: GripperDesign, obj: GraspObject, sc: Scenario, depth: float | None = None,
             finger: Finger | None = None, record: bool = True) -> SimResult:
    """Close, settle, release (sheets), lift, hold, shake, hold; then score."""
    f = finger or build_finger(design)
    if not f.feasible:
        return SimResult(design, obj.id, False, "infeasible geometry")
    if depth is None:
        an = solve_grip(design, obj, sc, finger=f)
        depth = an.info.get("insertion_mm", sc.for_kind(obj.kind).insertion_depth * 1e3) / 1e3
    xml, meta = build_model(design, obj, sc, depth, finger=f)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    phi0 = meta["phi0"]
    data.qpos[qadr("jR")] = phi0
    data.qpos[qadr("jL")] = phi0
    mujoco.mj_forward(model, data)
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "holder") if obj.kind == "sheet" else -1

    els = meta["elements"]
    el_adr = np.array([qadr(e.joint) for e in els])
    el_k = np.array([e.k for e in els])
    el_area = np.array([e.area for e in els])
    el_right = np.array([e.finger > 0 for e in els])
    el_body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, e.joint) for e in els])
    body_side = dict(zip(el_body.tolist(), (1 if e.finger > 0 else -1 for e in els)))
    geom_side = np.array([body_side.get(int(b), 0) for b in model.geom_bodyid], dtype=int)
    is_obj = (model.geom_contype & OBJ) > 0
    hinge_adr = np.array([qadr(h) for h in meta["hinges"]], dtype=int)
    hinge_k = np.array(meta["hinge_k"])
    a_ox, a_oz, a_oth = qadr("ox"), qadr("oz"), qadr("oth")
    a_bx, a_bz, a_jR = qadr("base_x"), qadr("base_z"), qadr("jR")
    traj = Trajectory(sc)
    tau = meta["tau"]
    kv, kp = meta["kv_servo"], meta["k_servo"]
    top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta["top_body"])

    dt = model.opt.timestep
    n_steps = int(math.ceil(traj.t_end / dt))
    rec_every = max(int(round(1.0 / (float(sc.sim.get("record_hz", 200)) * dt))), 1)
    tr: dict[str, list[float]] = {k: [] for k in ("t", "base_x", "base_z", "phi", "rel_x", "rel_z", "rel_th", "grip_N", "p_max", "hinge_M")}

    z_fill = meta["z_top"] - obj.headspace if obj.kind == "box" else -math.inf
    ref = None
    p_peak = 0.0
    p_impact = 0.0
    fric_peak = 0.0
    panel_peak = 0.0
    crease_peak = 0.0
    slip_peak = 0.0
    rot_peak = 0.0
    grip_settle = 0.0
    released = False
    unstable = False
    p_grip_local = None

    def grip_point_rel() -> np.ndarray:
        # object point at the grip, in the base frame
        R = data.xmat[top].reshape(3, 3)
        pw = data.xpos[top] + R @ p_grip_local
        return np.array([pw[0] - data.qpos[a_bx], pw[2] - (meta["z_piv"] + data.qpos[a_bz])])

    for i in range(n_steps):
        t = data.time
        ramp = min(1.0, t / 0.05)
        data.ctrl[2] = -tau * ramp
        data.ctrl[3] = -tau * ramp
        x, z, vx, vz = traj(t)
        data.ctrl[0] = x + kv / kp * vx
        data.ctrl[1] = z + kv / kp * vz
        if eq >= 0 and not released and t >= traj.t_close:
            data.eq_active[eq] = 0
            released = True
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qacc).all() or abs(data.qpos[a_jR]) > 3.0:
            unstable = True
            break
        q = data.qpos[el_adr]
        force = el_k * np.maximum(-q, 0.0)
        if ref is None:
            p_impact = max(p_impact, float((force / el_area).max()))
        if ref is None and data.time >= traj.t_settle - 0.02:
            loaded = force > 1e-3
            zs = data.xpos[el_body[loaded], 2] if loaded.any() else np.array([meta["z_top"] - 0.01])
            zg = float(np.average(zs, weights=force[loaded])) if loaded.any() else float(zs[0])
            R = data.xmat[top].reshape(3, 3)
            p_grip_local = R.T @ (np.array([0.0, 0.0, zg]) - data.xpos[top])
            ref = grip_point_rel()
            ref_th = data.qpos[a_oth]
            grip_settle = float(force[el_right].sum())
        if ref is not None:
            p_el = force / el_area
            p_peak = max(p_peak, float(p_el.max()))
            if obj.kind == "box" and math.isfinite(obj.unsupported_force):
                above = data.xpos[el_body, 2] > z_fill + data.qpos[a_oz]
                panel_peak = max(panel_peak, float(force[above & el_right].sum()), float(force[above & ~el_right].sum()))
            rel = grip_point_rel()
            slip_peak = max(slip_peak, float(np.hypot(*(rel - ref))))
            rot_peak = max(rot_peak, abs(float(data.qpos[a_oth] - ref_th)))
            if len(hinge_adr):
                crease_peak = max(crease_peak, float(np.max(np.abs(hinge_k * data.qpos[hinge_adr]))))
            if i % 10 == 0:
                fric_peak = max(fric_peak, _friction_use(data, geom_side, is_obj))
            if slip_peak > DROP_DISTANCE:
                break
        if record and i % rec_every == 0:
            tr["t"].append(data.time)
            tr["base_x"].append(float(data.qpos[a_bx]))
            tr["base_z"].append(float(data.qpos[a_bz]))
            tr["phi"].append(float(data.qpos[a_jR]))
            if ref is not None:
                rel = grip_point_rel() - ref
                tr["rel_x"].append(float(rel[0]))
                tr["rel_z"].append(float(rel[1]))
                tr["rel_th"].append(float(data.qpos[a_oth] - ref_th))
            else:
                tr["rel_x"].append(0.0)
                tr["rel_z"].append(0.0)
                tr["rel_th"].append(0.0)
            tr["grip_N"].append(float(force[el_right].sum()))
            tr["p_max"].append(float((force / el_area).max()))
            tr["hinge_M"].append(float(np.max(np.abs(hinge_k * data.qpos[hinge_adr]))) if len(hinge_adr) else 0.0)

    trace = {k: np.asarray(v) for k, v in tr.items()}
    if unstable:
        return SimResult(design, obj.id, False, "simulation went unstable", {}, trace, depth * 1e3)
    if ref is None:
        return SimResult(design, obj.id, False, "never reached the lift", {}, trace, depth * 1e3)

    margins = {
        "slip": sc.max_slip / max(slip_peak, 1e-9),
        "rotation": sc.max_rotation / max(rot_peak, 1e-9),
        "pressure": obj.material.allow_pressure / (sc.pressure_sf * max(p_peak, 1e-9)),
    }
    if obj.kind == "box" and math.isfinite(obj.unsupported_force) and panel_peak > 0:
        margins["panel"] = obj.unsupported_force / panel_peak
    if obj.kind == "sheet":
        b_eff = design.pad_width * 1e-3 if sc.crease_width == "pad" else obj.width
        cap = obj.material.bending_strength * b_eff
        margins["crease"] = cap / (sc.crease_safety_factor * max(crease_peak, 1e-9))
    dropped = slip_peak > DROP_DISTANCE
    metrics = {
        "slip_mm": slip_peak * 1e3,
        "rotation_deg": math.degrees(rot_peak),
        "peak_pressure_kPa": p_peak / 1e3,
        "closing_impact_pressure_kPa": p_impact / 1e3,
        "panel_force_N": panel_peak,
        "crease_moment_Nm": crease_peak,
        "grip_N": grip_settle,
        "friction_use": fric_peak,
        "friction_margin": float(min(1.0 / max(fric_peak, 1e-9), 99.0)),
        "dropped": float(dropped),
        "margins": {k: float(min(v, 99.0)) for k, v in margins.items()},
    }
    if dropped:
        metrics["margins"] = {"slip": 0.0}
        return SimResult(design, obj.id, False, "dropped", metrics, trace, depth * 1e3)
    worst = min(margins, key=margins.get)
    passed = margins[worst] >= 1.0
    reason = "" if passed else f"{worst} margin {margins[worst]:.2f}"
    return SimResult(design, obj.id, passed, reason, metrics, trace, depth * 1e3)
