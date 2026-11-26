"""T5, the two-arm lift (test conditions revision 2.18), in MuJoCo's x-z plane: W along x, gravity -z.

A gripper at each end of the case's W opens into the L, its fingers 90 deg apart: the spec's outside finger
lies level under the bottom edge and the thumb faces the side wall. Seen in the jaw plane, the left gripper is
the right one turned about the vertical, so it is the right one mirrored, and an asymmetric pair keeps its
outside finger under the case at both ends. Each arm moves in until the case meets the hub or a finger; the jaw
closes until both fingers touch, and a stiff position servo holds it there. The arms squeeze, lift together,
hold, shake and make the stop along W; the power cut then sets both drives to zero with the brake on.

The hub is a contact body: the brackets' plan footprints, treated as full height. The case is rigid, on a deck
cut away under each gripped edge. Pad and bare-face elements come from the batch-1 builder, on the case's
C-flute; their load is read from their springs (test conditions 2.16).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import mujoco
import numpy as np
from shapely.ops import unary_union

from .jaw import Jaw, convex_pieces
from .mj1 import BAD_STATE, DROP, GRIP, OBJ, PA12_DENSITY, RETURN_G, STOP_SPEED, TABLE, _Builder, _envelope, _fmt, _min_jerk, _p
from .objects import G, GraspObject, Material
from .scenario import Scenario

UNSTABLE = "simulation went unstable"
HUB, DECK = 8, 16
HUB_WIDTH_MM = 60.0  # out of plane; the contact is planar, so it only sizes the mesh
TOUCH_N = 0.5  # N on a finger that counts as touching the case
APPROACH_N = 1.0  # N on a gripper that ends its arm's approach


def case_object(cfg: dict[str, Any], key: str, materials: dict[str, Material], mass: float | None = None) -> GraspObject:
    """One T5 case as a rigid box: W across the arms, D out of plane, H tall."""
    c = cfg["cases"][key]
    W, D, H = (float(c[k]) * 1e-3 for k in ("W_mm", "D_mm", "H_mm"))
    m = float(c["mass_kg"] if mass is None else mass)
    base = materials[str(cfg.get("material", "c_flute"))]
    # The sheet reading compresses half the caliper from each side; a case wall takes the thumb on one face.
    mat = replace(base, k_surface=base.k_surface / 2.0,
                  allow_pressure=float(c.get("thumb_limit_kPa", cfg.get("thumb_limit_kPa", 105.0))) * 1e3)
    return GraspObject(id=f"t5_{key}", label=str(c.get("label", key)), kind="case", thickness=W, height=H, width=D,
                       mass=m, com_below_top=H / 2, inertia=m * (W * W + H * H) / 12.0, material=mat)


def mirrored(jaw: Jaw) -> Jaw:
    """The jaw seen from the other side: each finger's geometry on the other side of the centreline."""
    sides = {s: replace(jaw.sides[-s], side=s, faces=[replace(f, side=s) for f in jaw.sides[-s].faces]) for s in (1, -1)}
    return replace(jaw, sides=sides)


class _GripperBuilder(_Builder):
    """The batch-1 finger builder with every name prefixed by its gripper."""

    def __init__(self, prefix: str, *args: Any, **kw: Any):
        super().__init__(*args, **kw)
        self.prefix = prefix

    def finger_xml(self, side: int) -> tuple[str, float, float]:
        js = self.jaw.sides[side]
        tag = self.prefix + ("R" if side > 0 else "L")
        out = []
        mass, mom = 0.0, 0.0
        k = 0
        for poly in js.body_mj:
            for piece in convex_pieces(poly):
                c = piece.centroid
                w = js.width_at(float(c.x))
                verts = []
                for u, v in np.asarray(piece.exterior.coords)[:-1]:
                    X, Z = _p(u, v, side)
                    verts += [X, -w * 0.5e-3, Z, X, w * 0.5e-3, Z]
                name = f"{tag}m{k}"
                self.assets.append(f'<mesh name="{name}" vertex="{_fmt(verts)}"/>')
                gname = f"{tag}_body{k}"
                out.append(f'<geom name="{gname}" type="mesh" mesh="{name}" mass="0" friction="{_fmt(self.mu_bare)} 0.005 0.0001" '
                           f'priority="1" contype="{GRIP}" conaffinity="{OBJ | TABLE}" rgba="0.55 0.6 0.68 1"/>')
                self.body_geoms[gname] = side
                m = PA12_DENSITY * piece.area * 1e-6 * w * 1e-3
                mass += m
                mom += m * ((c.x * 1e-3) ** 2 + (c.y * 1e-3) ** 2)
                k += 1
        for face in js.faces:
            out.append(self.face_xml(face, side, tag))
        return "".join(out), mass, mom


def hub_region(sc: Scenario):
    """The brackets' plan footprints in gripper coordinates (mm): x across, z along the centreline, tips at -z."""
    outlines = [b.outline for b in sc.keep_out.brackets if b.outline is not None]
    if not outlines:
        from shapely.geometry import Point
        return unary_union([Point(x, z).buffer(r, 32) for x, z, r in sc.keep_out.circles])
    return unary_union(outlines)


def _hub_xml(region, prefix: str, mirror: bool, mu: float, width_mm: float, assets: list[str]) -> tuple[str, list[str]]:
    geoms, names = [], []
    sx = -1.0 if mirror else 1.0
    for k, piece in enumerate(convex_pieces(region)):
        verts = []
        for x, z in np.asarray(piece.exterior.coords)[:-1]:
            verts += [sx * x * 1e-3, -width_mm * 0.5e-3, z * 1e-3, sx * x * 1e-3, width_mm * 0.5e-3, z * 1e-3]
        assets.append(f'<mesh name="{prefix}hubm{k}" vertex="{_fmt(verts)}"/>')
        nm = f"{prefix}_hub{k}"
        names.append(nm)
        geoms.append(f'<geom name="{nm}" type="mesh" mesh="{prefix}hubm{k}" mass="0" friction="{_fmt(mu)} 0.005 0.0001" '
                     f'priority="1" contype="{HUB}" conaffinity="{OBJ}" rgba="0.35 0.35 0.4 1"/>')
    return "".join(geoms), names


class T5Timeline:
    """Carriage motion from the end of the squeeze: settle, lift, hold, shake, final hold, the stop along W, and
    the power cut, which comes after the stop or, for the mid-move cut, before it."""

    def __init__(self, sc: Scenario, t0: float, shake_g: float, shake_hz: tuple[float, float], shake_s: float,
                 stop_g: float | None, power_s: float, cut_before_stop: bool = False, stop_sign: float = 1.0,
                 final_s: float | None = None):
        self.sc = sc
        self.t_settle = t0 + sc.settle_time
        self.t_lift = self.t_settle + sc.lift_time
        self.t_hold = self.t_lift + max(sc.hold_time, 0.5)
        self.t_shake = self.t_hold + shake_s
        self.t_final = self.t_shake + (sc.final_hold_time if final_s is None else final_s)
        self.shake_s = shake_s
        self.ramp = float(sc.raw.get("sequence", {}).get("shake", {}).get("ramp_s", 0.5))
        self.wx, self.wz = 2 * math.pi * shake_hz[0], 2 * math.pi * shake_hz[1]
        a = shake_g * G
        self.ax = a / self.wx ** 2 if self.wx > 0 else 0.0
        self.az = a / self.wz ** 2 if self.wz > 0 else 0.0
        self.stop = None
        t = self.t_final + (0.2 if cut_before_stop else 0.0)
        if stop_g:
            a1, a2 = stop_g * G, RETURN_G * G
            self.stop = (t, STOP_SPEED / a1, STOP_SPEED / a2, stop_sign, a1, a2)
            t_stop_end = t + STOP_SPEED / a1 + STOP_SPEED / a2 + 0.2
        else:
            t_stop_end = t
        self.t_stop_end = t_stop_end
        self.t_power = self.t_final if cut_before_stop else t_stop_end
        self.t_end = (t_stop_end if cut_before_stop else self.t_power) + power_s

    def stop_state(self, t: float) -> tuple[float, float]:
        if self.stop is None or t <= self.stop[0]:
            return 0.0, 0.0
        t0, T1, T2, sgn, a1, a2 = self.stop
        s = t - t0
        if s < T1:
            return sgn * (-0.5 * a1 * s * s), sgn * (-a1 * s)
        d1 = -0.5 * a1 * T1 * T1
        if s < T1 + T2:
            r = s - T1
            return sgn * (d1 - STOP_SPEED * r + 0.5 * a2 * r * r), sgn * (-STOP_SPEED + a2 * r)
        return sgn * (d1 - STOP_SPEED * T2 + 0.5 * a2 * T2 * T2), 0.0

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
            return "settle"
        if t < self.t_lift:
            return "lift"
        if t < self.t_hold:
            return "hold"
        if t < self.t_shake:
            return "shake"
        if t < self.t_final:
            return "final"
        if self.stop is not None and self.stop[0] <= t < self.t_stop_end:
            return "stop"
        return "power" if t >= self.t_power else "final"


def simulate_t5(jaw: Jaw, case: GraspObject, sc: Scenario, b1: dict[str, Any], *, com_offset: float = 0.0,
                squeeze: float | None = None, friction_scale: float = 1.0, bare_mu: float | None = None,
                brake: float = 30.0, stop_g: float | None = None, cut_before_stop: bool = False,
                power_s: float | None = None, shake_g: float | None = None, hub_contact: bool = True,
                approach: str = "close", hold: str = "servo", record: bool = False) -> dict[str, Any]:
    """One T5 run. `com_offset` (fraction of W) moves the centre of mass toward gripper A, the +x gripper, which
    the stop also loads. `squeeze` is N per side; `bare_mu` replaces the bare printed finger's friction on the
    case; `brake` is N m per gripper; `stop_g=0` leaves out the stop. `hub_contact=False` releases the hub once
    the jaw has closed, so only the fingers carry the case: the hand checks' load path.

    `approach="seated"` is revision 2.13's stop-seated cradle: the jaw stays on its stops and skips the closing
    step. `hold` is what holds the opening once the jaw has closed or seated: "servo" (the drive's position servo),
    "stops" (each finger on its own stop, the gear coupling released) or "thumb_stop" (the outside finger's stop
    removed, so its moment crosses the mesh to the thumb's); with the stops the drive is off, on gear friction."""
    if approach not in ("close", "seated") or hold not in ("servo", "stops", "thumb_stop"):
        raise ValueError(f"unknown T5 approach {approach!r} or hold {hold!r}")
    cfg = b1.get("t5", {}) or {}
    W, H, D = case.thickness, case.height, case.width
    hf = float(cfg.get("hub_front_mm", 48.5)) * 1e-3
    app = float(cfg.get("approach_mm", 30.0)) * 1e-3
    # The L is 45 deg per finger, or the stroke's end where the reading stops it short (44.5 deg from a -0.5 deg meet).
    l_angle = math.radians(float(cfg.get("preset_deg", 45.0)))
    preset = min(l_angle, jaw.phi_max)
    stop_phi = jaw.phi_max
    if approach == "seated" or hold != "servo":
        # A stop that holds the L is set at 45 deg per finger, "never short" (S4's and S5's specs), so it can sit up to
        # half a degree past a stroke read 45 deg from where the fingers meet.
        stop_phi = max(jaw.phi_max, l_angle)
        if approach == "seated":
            preset = stop_phi
    squeeze = float(cfg.get("squeeze_N", 30.0)) if squeeze is None else float(squeeze)
    stop_g = float(cfg.get("stop_g", 1.0)) if stop_g is None else stop_g
    power_s = float((b1.get("power_cut") or {}).get("hold_s", 5.0)) if power_s is None else power_s
    shake_g = float((b1.get("shake_g") or {}).get("T5", 0.25)) if shake_g is None else shake_g
    if bare_mu is not None:
        case = replace(case, material=replace(case.material, bare_friction=bare_mu / friction_scale))
    if preset < jaw.phi_meet + math.radians(30.0):
        return {"ok": False, "reason": f"the stroke ends at {math.degrees(jaw.phi_max):.1f} deg, short of the L"}

    region = hub_region(sc)
    com_x = com_offset * W
    m = case.mass
    k_servo = float(sc.sim.get("servo_kp", 2.0e6))
    kv_carr = 2 * math.sqrt(k_servo * (2.0 + 1.2 + m))
    kv_arm = 2 * math.sqrt(k_servo * (0.6 + m / 2))

    def build(dx: float, dz: float):
        """The model with each arm `dx` further out and `dz` higher than the hub-at-the-corner start."""
        builders = {}
        grippers_xml = []
        assets: list[str] = []
        hub_names: dict[str, list[str]] = {}
        for g, sgn, jw in (("A", 1.0, jaw), ("B", -1.0, mirrored(jaw))):
            b = _GripperBuilder(g, jw, case, sc, b1, 0.0, {1: preset, -1: preset}, friction_scale, 1.0, 0.0)
            builders[g] = b
            fingers = []
            for side in (1, -1):
                xml, m_f, mom = b.finger_xml(side)
                nm = f"{g}{'R' if side > 0 else 'L'}"
                fingers.append(
                    f'<body name="finger{nm}" pos="{_fmt((side * jw.pivot_x * 1e-3, 0, 0))}" gravcomp="1">'
                    f'<joint name="j{nm}" type="hinge" axis="0 {-side} 0" range="{_fmt(jw.phi_meet)} {_fmt(stop_phi + 1e-3)}" '
                    f'armature="{_fmt(float(b1.get("closing", {}).get("armature_kgm2", 0.015)))}" damping="0" '
                    f'solreffriction="{_fmt(2 * b.dt)} 1" solimpfriction="0.999 0.9999 0.001 0.5 2"/>'
                    f'<inertial pos="0 0 {_fmt(-jw.reach * 0.5e-3)}" mass="{_fmt(max(m_f, 1e-3))}" diaginertia="{_fmt((max(mom, 1e-6),) * 3)}"/>'
                    f"{xml}</body>")
            hub_xml, hub_names[g] = _hub_xml(region, g, sgn < 0, b.mu_bare, HUB_WIDTH_MM, b.assets)
            theta = sgn * 0.75 * math.pi
            x0 = sgn * (W / 2 + hf / math.sqrt(2) + app + dx)
            z0 = -hf / math.sqrt(2) + dz
            grippers_xml.append(
                f'<body name="arm{g}" pos="{_fmt((x0, 0, z0))}" gravcomp="1">'
                f'<joint name="a{g}" type="slide" axis="1 0 0"/>'
                '<inertial pos="0 0 0" mass="0.6" diaginertia="1e-3 1e-3 1e-3"/>'
                f'<body name="grip{g}" euler="0 {_fmt(theta)} 0" gravcomp="1">{hub_xml}{"".join(fingers)}</body></body>')
            assets += b.assets
        bA = builders["A"]
        if any(e.seg <= 2.0 + 1e-9 for bb in builders.values() for e in bb.elements):
            bA.substeps = max(bA.substeps, int(round(bA.dt / bA.fine_dt)))
        dt = bA.dt / bA.substeps
        slot = float(cfg.get("slot_mm", 120.0)) * 1e-3
        deck_half = W / 2 - slot
        deck = ""
        if deck_half > 0:
            deck = (f'<geom name="deck" type="box" pos="0 0 -0.01" size="{_fmt((deck_half, 0.4, 0.01))}" '
                    f'friction="{_fmt(float(cfg.get("deck_friction", 0.4)))} 0.005 0.0001" priority="1" contype="{DECK}" '
                    f'conaffinity="{OBJ}" rgba="0.8 0.8 0.8 1"/>')
        xml = f"""<mujoco model="t5">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{_fmt(dt)}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10" iterations="60" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default><geom condim="3" solref="{_fmt(bA.contact_tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"/></default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {deck}
    <body name="case" pos="0 0 {_fmt(H / 2 + 2e-4)}">
      <joint name="cx" type="slide" axis="1 0 0"/><joint name="cz" type="slide" axis="0 0 1"/>
      <joint name="cth" type="hinge" axis="0 1 0"/>
      <inertial pos="{_fmt((com_x, 0, 0))}" mass="{_fmt(m)}" diaginertia="{_fmt((case.inertia,) * 3)}"/>
      <geom name="case" type="box" size="{_fmt((W / 2, D / 2, H / 2))}" mass="0" friction="{_fmt(case.material.friction)} 0.005 0.0001"
            contype="{OBJ}" conaffinity="{GRIP | HUB | DECK}" rgba="0.75 0.6 0.4 1"/>
    </body>
    <body name="carriage" pos="0 0 0" gravcomp="1">
      <joint name="kx" type="slide" axis="1 0 0"/><joint name="kz" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="1e-3 1e-3 1e-3"/>
      {''.join(grippers_xml)}
    </body>
  </worldbody>
  <equality>
    <joint name="gearA" joint1="jAL" joint2="jAR" polycoef="0 1 0 0 0" solref="{_fmt(4 * bA.dt)} 1"/>
    <joint name="gearB" joint1="jBL" joint2="jBR" polycoef="0 1 0 0 0" solref="{_fmt(4 * bA.dt)} 1"/>
  </equality>
  <actuator>
    <position name="kx" joint="kx" kp="{_fmt(k_servo)}" kv="{_fmt(kv_carr)}"/>
    <position name="kz" joint="kz" kp="{_fmt(k_servo)}" kv="{_fmt(kv_carr)}"/>
    <position name="pA" joint="aA" kp="{_fmt(k_servo)}" kv="{_fmt(kv_arm)}"/>
    <position name="pB" joint="aB" kp="{_fmt(k_servo)}" kv="{_fmt(kv_arm)}"/>
    <motor name="fA" joint="aA" gear="1"/>
    <motor name="fB" joint="aB" gear="1"/>
    <motor name="dAR" joint="jAR" gear="1"/><motor name="dAL" joint="jAL" gear="1"/>
    <motor name="dBR" joint="jBR" gear="1"/><motor name="dBL" joint="jBL" gear="1"/>
  </actuator>
</mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        return model, builders, hub_names

    model, builders, hub_names = build(0.0, 0.0)
    dx, dz = _start_shift(model, preset, W, app, float(cfg.get("under_edge_mm", 2.0)) * 1e-3)
    if dx > 1e-6 or dz < -1e-6:
        model, builders, hub_names = build(dx, dz)
    data = mujoco.MjData(model)
    res = _run_t5(model, data, jaw, case, sc, b1, cfg, builders, hub_names, preset, squeeze, stop_g, cut_before_stop,
                  power_s, shake_g, brake, k_servo, com_offset, hub_contact, record, approach, hold, stop_phi)
    res.update(approach=approach, hold=hold, shake_g=shake_g, preset_deg=math.degrees(preset),
               stop_deg_jaw=math.degrees(stop_phi), stop_type=jaw.stop_type or "",
               start_out_mm=dx * 1e3, start_down_mm=-dz * 1e3)
    return res


def _geom_points(model, data, i: int) -> np.ndarray:
    """A geom's corners in the world: a mesh's vertices, or the corners of its bounding box."""
    R = data.geom_xmat[i].reshape(3, 3)
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[i]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a:a + n].astype(float)
    else:
        c, h = model.geom_aabb[i][:3], model.geom_aabb[i][3:]
        v = np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return data.geom_xpos[i] + v @ R.T


def _hub_readings(g: str, sustained: list[float], t: list[float], forces: list[np.ndarray], start: tuple[str, float],
                  sample_s: float, avg_s: float) -> dict[str, Any]:
    """Revision 2.21's hub force for one gripper: sustained, the mean over the final hold's last 1 s; peak, the
    largest `avg_s` average from the stop (or, with no stop, the power cut) to 5 s after it, with that window's
    mean push outward along W and down, and its start after the event."""
    out = {f"{g}_hub_sustained_N": float(np.mean(sustained)) if sustained else math.nan,
           f"{g}_hub_peak_5ms_N": math.nan, f"{g}_hub_peak_5ms_out_N": math.nan, f"{g}_hub_peak_5ms_down_N": math.nan,
           f"{g}_hub_peak_5ms_after_s": math.nan, "hub_peak_from": start[0]}
    n = max(1, int(round(avg_s / sample_s)))
    if len(forces) < n:
        return out
    F = np.asarray(forces)
    kern = np.ones(n) / n
    avg = np.convolve(np.linalg.norm(F, axis=1), kern, "valid")
    k = int(np.argmax(avg))
    sgn = 1.0 if g == "A" else -1.0
    out.update({f"{g}_hub_peak_5ms_N": float(avg[k]), f"{g}_hub_peak_5ms_out_N": sgn * float(F[k:k + n, 0].mean()),
                f"{g}_hub_peak_5ms_down_N": -float(F[k:k + n, 2].mean()), f"{g}_hub_peak_5ms_after_s": t[k] - start[1]})
    return out


def _start_shift(model, preset: float, W: float, app: float, under: float) -> tuple[float, float]:
    """How far to start gripper A out (dx) and down (dz) so that, in the L, its outside finger lies under the case's
    bottom edge and nothing above that edge comes within the approach distance of the wall. With the hub at the
    corner the tip-bearing fingers already sit about 30 mm under the edge; a cradle's 38 mm steps don't."""
    data = mujoco.MjData(model)
    for f in ("AR", "AL"):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{f}")]] = preset
    mujoco.mj_kinematics(model, data)
    out_pts, all_pts = [], []
    for i in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not (nm.startswith("AR") or nm.startswith("AL") or nm.startswith("A_hub")):
            continue
        p = _geom_points(model, data, i)
        all_pts.append(p)
        if nm.startswith("AR"):
            out_pts.append(p)
    z_top = max(float(p[:, 2].max()) for p in out_pts)
    dz = min(0.0, -under - z_top)
    pts = np.vstack(all_pts)
    above = pts[pts[:, 2] + dz > 0.0]
    gap = float(above[:, 0].min()) - W / 2 if len(above) else math.inf
    return max(0.0, app - gap), dz


def _run_t5(model, data, jaw: Jaw, case: GraspObject, sc: Scenario, b1: dict, cfg: dict, builders: dict, hub_names: dict,
            preset: float, squeeze: float, stop_g: float, cut_before_stop: bool, power_s: float, shake_g: float,
            brake: float, k_servo: float, com_offset: float, hub_contact: bool, record: bool,
            approach: str = "close", hold: str = "servo", stop_phi: float | None = None) -> dict[str, Any]:
    name = lambda kind, n: mujoco.mj_name2id(model, kind, n)
    jid = lambda n: name(mujoco.mjtObj.mjOBJ_JOINT, n)
    gid = lambda n: name(mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: name(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    vadr = lambda n: model.jnt_dofadr[jid(n)]
    W, H = case.thickness, case.height
    G_ = ("A", "B")
    # Per gripper: the outside finger is side +1 of the spec's jaw; gripper B carries it on its -1 side.
    out_side = {"A": "R", "B": "L"}
    thumb_side = {"A": "L", "B": "R"}
    fingers = {g: {"out": f"{g}{out_side[g]}", "thumb": f"{g}{thumb_side[g]}"} for g in G_}
    for g in G_:
        for f in ("R", "L"):
            data.qpos[qadr(f"j{g}{f}")] = preset
    mujoco.mj_forward(model, data)
    case_geom = gid("case")
    case_body = name(mujoco.mjtObj.mjOBJ_BODY, "case")
    # Geom tables: gripper, part (0 outside finger, 1 thumb, 2 hub), element index.
    ng = model.ngeom
    g_grip = np.full(ng, -1, dtype=int)
    g_part = np.full(ng, -1, dtype=int)
    g_el = np.full(ng, -1, dtype=int)
    els = []
    for gi, g in enumerate(G_):
        b = builders[g]
        for nm in hub_names[g]:
            g_grip[gid(nm)], g_part[gid(nm)] = gi, 2
        for gname in b.body_geoms:
            i = gid(gname)
            tag = gname.split("_", 1)[0]
            g_grip[i], g_part[i] = gi, 0 if tag == fingers[g]["out"] else 1
        for e in b.elements:
            i = gid(e.geom)
            tag = e.geom.split("_", 1)[0]
            g_grip[i], g_part[i] = gi, 0 if tag == fingers[g]["out"] else 1
            g_el[i] = len(els)
            els.append((gi, g_part[i], e, i))
    n_el = len(els)
    el_gp = np.array([gi * 2 + p for gi, p, _, _ in els], dtype=int)
    el_area = np.array([e.area for _, _, e, _ in els])
    el_grip = np.array([e.kind != "bare" for _, _, e, _ in els])
    el_qadr = np.array([qadr(e.joint) for _, _, e, _ in els], dtype=int)
    el_lo = np.array([model.jnt_range[jid(e.joint), 0] for _, _, e, _ in els])
    el_k = np.array([model.jnt_stiffness[jid(e.joint)] for _, _, e, _ in els])
    el_q0 = np.array([model.qpos_spring[qadr(e.joint)] for _, _, e, _ in els])
    body_of = {f: name(mujoco.mjtObj.mjOBJ_BODY, f"finger{f}") for g in G_ for f in fingers[g].values()}
    axis_y = {f: (-1.0 if f.endswith("R") else 1.0) for f in body_of}
    q_f = {f: qadr(f"j{f}") for f in body_of}
    v_f = {f: vadr(f"j{f}") for f in body_of}
    a_drive = {f: aid(f"d{f}") for f in body_of}
    a_pos = {g: aid(f"p{g}") for g in G_}
    a_frc = {g: aid(f"f{g}") for g in G_}
    q_arm = {g: qadr(f"a{g}") for g in G_}
    a_kx, a_kz = aid("kx"), aid("kz")
    q_cx, q_cz, q_cth = qadr("cx"), qadr("cz"), qadr("cth")
    kv_pos = {g: float(-model.actuator_biasprm[a_pos[g], 2]) for g in G_}
    kv_k = float(-model.actuator_biasprm[a_kx, 2])
    kp_j = float(cfg.get("jaw_servo_Nm_rad", 2.0e4))
    inertia_j = float(b1.get("closing", {}).get("armature_kgm2", 0.015))
    kv_j = 2 * math.sqrt(kp_j * inertia_j)
    tau = jaw.torque
    w_close = float(cfg.get("close_rad_s", 0.3))
    kv_drive = tau / 0.2
    v_app = float(cfg.get("approach_m_s", 0.1))
    app = float(cfg.get("approach_mm", 30.0)) * 1e-3
    sq_s = float(cfg.get("squeeze_s", 0.4))
    inward = {"A": -1.0, "B": 1.0}
    every = 4 * builders["A"].substeps
    dt = model.opt.timestep
    j_f = {f: jid(f"j{f}") for f in body_of}
    eq_gear = {g: name(mujoco.mjtObj.mjOBJ_EQUALITY, f"gear{g}") for g in G_}
    stop_phi = jaw.phi_max if stop_phi is None else stop_phi
    gear_friction = float(cfg.get("gear_friction_Nm", 1.0))  # per gripper, with the drive off
    ccfg = (cfg.get("cases") or {}).get(case.id[3:] if case.id.startswith("t5_") else case.id, {}) or {}
    perf_z = (1.0 - float(ccfg["perforation_from_top"])) * H if ccfg.get("perforation_from_top") is not None else None

    def side_of(f):
        """The spec's finger a gripper's finger is: outside (+1) or thumb (-1)."""
        return 1 if f in (fingers[g]["out"] for g in G_) else -1

    def root_bound(M, f):
        """2D bound on the root stress (MPa): the finger's moment about its pivot over the section beside the
        hub cut, the notch taken through the full width (6 M / w t^2)."""
        src = getattr(jaw, "source", None)
        sf = getattr(src, "right" if side_of(f) > 0 else "left", None)
        t = getattr(getattr(sf, "finger", None), "ligament", math.inf)
        w = jaw.sides[side_of(f)].width_at(30.0)
        if not (math.isfinite(t) and t > 0 and w > 0 and math.isfinite(M)):
            return math.nan
        return 6.0 * abs(M) / (w * 1e-3 * (t * 1e-3) ** 2) / 1e6

    def case_contacts():
        n = data.ncon
        if n == 0:
            return None
        con = data.contact
        g1, g2 = con.geom1[:n], con.geom2[:n]
        adr = con.efc_address[:n]
        sel = ((g1 == case_geom) | (g2 == case_geom)) & (adr >= 0)
        if not sel.any():
            return None
        idx = np.nonzero(sel)[0]
        a = adr[idx]
        fr = con.frame[idx]
        f_on2 = (data.efc_force[a][:, None] * fr[:, 0:3] + data.efc_force[a + 1][:, None] * fr[:, 3:6]
                 + data.efc_force[a + 2][:, None] * fr[:, 6:9])
        other = np.where(g1[idx] == case_geom, g2[idx], g1[idx])
        # Force on the gripper part (or deck) from the case.
        f_part = np.where((g1[idx] == case_geom)[:, None], f_on2, -f_on2)
        return other, f_part, con.pos[idx], data.efc_force[a], con.dist[idx]

    def limit_load(f):
        """The moment (N m) a finger's stroke-end limit, its stop, carries; the meet's limit doesn't count."""
        if float(data.qpos[q_f[f]]) < 0.5 * (jaw.phi_meet + stop_phi):
            return 0.0
        ne = data.nefc
        sel = (data.efc_type[:ne] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (data.efc_id[:ne] == j_f[f])
        return float(np.abs(data.efc_force[:ne][sel]).sum())

    def measure():
        """Per finger: force, opening moment about its pivot, normal load and load-weighted distance from the
        pivot; per gripper: hub force; per element: load from its spring, or from contact at its stop."""
        out = {"F": {}, "M": {}, "N": {}, "r": {}, "hub": {}, "deck": 0.0, "span": {}, "zc": {}, "zmin": {}, "zmax": {},
               "pen_hub": {g: 0.0 for g in G_}, "pen_finger": {g: 0.0 for g in G_}}
        Fc = np.zeros(n_el)
        c = case_contacts()
        for gi, g in enumerate(G_):
            out["hub"][g] = np.zeros(3)
            for part, f in ((0, fingers[g]["out"]), (1, fingers[g]["thumb"])):
                out["F"][f], out["M"][f], out["N"][f], out["r"][f] = np.zeros(3), 0.0, 0.0, math.nan
                out["span"][f], out["zc"][f], out["zmin"][f], out["zmax"][f] = 0.0, math.nan, math.nan, math.nan
        if c is not None:
            other, f_part, pos, fn, dist = c
            gr, pt, el = g_grip[other], g_part[other], g_el[other]
            ok = el >= 0
            np.add.at(Fc, el[ok], fn[ok])
            out["deck"] = float(np.sum(np.where(gr < 0, f_part[:, 2], 0.0)))
            z_bottom = float(data.xpos[case_body][2]) - H / 2
            for gi, g in enumerate(G_):
                h = (gr == gi) & (pt == 2)
                out["hub"][g] = f_part[h].sum(axis=0) if h.any() else np.zeros(3)
                if h.any():
                    out["pen_hub"][g] = max(0.0, float(-dist[h].min()))
                fs_ = (gr == gi) & (pt < 2)
                if fs_.any():
                    out["pen_finger"][g] = max(0.0, float(-dist[fs_].min()))
                for part, f in ((0, fingers[g]["out"]), (1, fingers[g]["thumb"])):
                    s = (gr == gi) & (pt == part)
                    if not s.any():
                        continue
                    piv = data.xpos[body_of[f]]
                    r = pos[s] - piv
                    Ff = f_part[s]
                    my = r[:, 2] * Ff[:, 0] - r[:, 0] * Ff[:, 2]
                    out["F"][f] = Ff.sum(axis=0)
                    out["M"][f] = float(my.sum() * axis_y[f])
                    wts = np.abs(fn[s])
                    out["N"][f] = float(wts.sum())
                    if wts.sum() > 1e-6:
                        rr = np.linalg.norm(r[:, [0, 2]], axis=1)
                        out["r"][f] = float(np.average(rr, weights=wts))
                        live = wts > 0.1
                        if live.any():
                            out["span"][f] = float(rr[live].max() - rr[live].min())
                            zz = pos[s][live, 2] - z_bottom
                            out["zc"][f] = float(np.average(zz, weights=wts[live]))
                            out["zmin"][f], out["zmax"][f] = float(zz.min()), float(zz.max())
        q = data.qpos[el_qadr]
        spring = np.where(el_grip & (q > el_lo + 2e-5), el_k * np.maximum(el_q0 - q, 0.0), np.nan)
        Fe = np.where(np.isnan(spring), Fc, spring)
        out["Fe"] = Fe
        return out

    # State machine: approach, close, squeeze, then the timeline.
    phase = "approach"
    arm_done = {g: False for g in G_}
    jaw_hold = {g: None for g in G_}
    jaw_note = {g: "" for g in G_}
    stall = {g: 0.0 for g in G_}
    first_touch = {g: math.nan for g in G_}
    t_app_end = t_close_end = t_sq_end = math.nan
    tl: T5Timeline | None = None
    brake_on = None
    ref = None
    unstable, unstable_s = False, math.nan
    dropped_in = ""
    t_limit = app / v_app + 2.0
    rec = {k: [] for k in ("t", "phase", "cz", "cth", "tauA", "tauB", "MAo", "MAt", "MBo", "MBt", "hubA", "hubB")} if record else {}
    stats: dict[str, Any] = {
        "hold": {"tau": {g: [] for g in G_}, "M": {f: [] for g in G_ for f in fingers[g].values()},
                 "hub": {g: [] for g in G_}, "r": {f: [] for g in G_ for f in fingers[g].values()},
                 "N": {f: [] for g in G_ for f in fingers[g].values()}},
        "peak_tau": {g: 0.0 for g in G_}, "peak_M": {f: 0.0 for g in G_ for f in fingers[g].values()},
        "peak_hub": {g: 0.0 for g in G_}, "peak_hub_phase": {g: "" for g in G_}, "peak_hub_stop": {g: 0.0 for g in G_},
        "peak_hub_stop_xz": {g: (0.0, 0.0) for g in G_},
        "peak_p": {f: 0.0 for g in G_ for f in fingers[g].values()},
        "peak_at": {f: None for g in G_ for f in fingers[g].values()},
        "peak_N": {f: 0.0 for g in G_ for f in fingers[g].values()},
        "brake_stop": {g: 0.0 for g in G_}, "brake_sat": {g: False for g in G_},
        "tilt": 0.0, "slip": 0.0, "power_slip": 0.0, "stop_slip": 0.0, "stop_tilt": 0.0,
        "peak_tau_phase": {g: "" for g in G_}, "brake_use": {g: 0.0 for g in G_},
        "stop_M": {f: 0.0 for g in G_ for f in fingers[g].values()}, "stop_tau": {g: 0.0 for g in G_},
    }
    stats.update({
        "slip_all": 0.0, "power_tilt": 0.0, "print_slip": 0.0, "failed_at_g": math.nan,
        "off_stop": {f: math.nan for f in body_of}, "stop_load": {f: 0.0 for f in body_of},
        "pen_hub": {g: 0.0 for g in G_}, "pen_finger": {g: 0.0 for g in G_},
    })
    for k in ("span", "zc", "zmin", "zmax", "stop_load"):
        stats["hold"][k] = {f: [] for f in body_of}
    ref_power = None
    ref_stop = None
    ref_print = None
    stops_on = False
    v_seat = float(cfg.get("seat_m_s", 0.05))
    seat_extra = float(cfg.get("seat_extra_mm", 3.0)) * 1e-3
    seat_max = float(cfg.get("seat_max_mm", 60.0)) * 1e-3
    t_seat_off = math.nan
    z_base = 0.0
    i = -1
    t_end = math.inf
    hub_win = hub_peak_win = (math.inf, -math.inf)
    hub_from = ("", math.nan)
    hub_sus = {g: [] for g in G_}
    hub_pk = {"t": [], **{g: [] for g in G_}}

    def start_squeeze(t):
        """The jaw has closed or seated: the arms go over to force control, the tips-only path releases the hub,
        and the stop-held runs turn the drive off and leave the opening to the stops."""
        nonlocal phase, t_close_end, stops_on
        phase, t_close_end = "squeeze", t
        for g in G_:
            model.actuator_gainprm[a_pos[g], 0] = 0.0
            model.actuator_biasprm[a_pos[g], 1] = 0.0
            if not hub_contact:
                for nm in hub_names[g]:
                    model.geom_contype[gid(nm)] = model.geom_conaffinity[gid(nm)] = 0
        if hold != "servo":
            stops_on = True
            for f in body_of:
                model.dof_frictionloss[v_f[f]] = gear_friction / 2.0
            for g in G_:
                if hold == "stops":
                    data.eq_active[eq_gear[g]] = 0
                else:
                    model.jnt_range[j_f[fingers[g]["out"]], 1] = stop_phi + 0.6

    while data.time < t_end:
        i += 1
        t = data.time
        # Arms.
        for g in G_:
            if phase == "approach" and not arm_done[g]:
                data.ctrl[a_pos[g]] = inward[g] * v_app * t
        # Jaws.
        for g in G_:
            for f in fingers[g].values():
                phi = float(data.qpos[q_f[f]])
                om = float(data.qvel[v_f[f]])
                if brake_on is not None or stops_on:
                    u = 0.0
                elif jaw_hold[g] is not None:
                    u = kp_j * (jaw_hold[g] - phi) - kv_j * om
                elif phase == "close":
                    u = -min(max(kv_drive * (w_close + om), 0.0), tau)
                else:
                    u = kp_j * (preset - phi) - kv_j * om
                data.ctrl[a_drive[f]] = u
        if tl is not None:
            x, z, vx, vz = tl(t)
            data.ctrl[a_kx] = x + kv_k / k_servo * vx
            data.ctrl[a_kz] = z_base + z + kv_k / k_servo * vz
            if brake_on is None and t >= tl.t_power:
                for f in body_of:
                    model.dof_frictionloss[v_f[f]] = brake / 2.0
                brake_on = t
        t_step = data.time
        mujoco.mj_step(model, data)
        if (data.time < t_step or any(data.warning[w].number for w in BAD_STATE) or not np.isfinite(data.qacc).all()):
            unstable, unstable_s = True, t_step
            break
        if i % every:
            continue
        t = data.time
        ms = measure()
        load = {g: ms["N"][fingers[g]["out"]] + ms["N"][fingers[g]["thumb"]] + float(np.linalg.norm(ms["hub"][g])) for g in G_}
        if phase == "approach":
            for g in G_:
                if not arm_done[g] and load[g] > APPROACH_N:
                    arm_done[g] = True
                    data.ctrl[a_pos[g]] = float(data.qpos[q_arm[g]])
            if all(arm_done.values()):
                phase, t_app_end = "close", t
                if approach == "seated":
                    # Revision 2.13: the jaw stays on its stops, and the arms rise to lift the case onto the cradle
                    # before they squeeze.
                    for g in G_:
                        jaw_hold[g] = preset
                        jaw_note[g] = "seated on its stops"
                    phase = "seat"
            elif t > t_limit:
                return {"ok": False, "reason": "the arms' approach found no contact"}
            continue
        if phase == "seat":
            rise = v_seat * (t - t_app_end)
            data.ctrl[a_kz] = rise + kv_k / k_servo * v_seat
            if record:
                rec.setdefault("seat", []).append((t, rise, float(data.qpos[q_cz]), float(data.qpos[q_cx]),
                                                   math.degrees(float(data.qpos[q_cth])), ms["deck"],
                                                   ms["N"][fingers["A"]["out"]], ms["N"][fingers["A"]["thumb"]],
                                                   float(np.linalg.norm(ms["hub"]["A"]))))
            # Seated once the case itself has risen off the deck: the deck's reading chatters while a step settles on
            # the wall.
            if float(data.qpos[q_cz]) > 1e-3 and not math.isfinite(t_seat_off):
                t_seat_off = t
            if math.isfinite(t_seat_off) and t - t_seat_off >= seat_extra / v_seat:
                z_base = rise
                start_squeeze(t)
            elif rise > seat_max:
                return {"ok": False, "reason": f"the cradle never lifted the case off the deck in {seat_max * 1e3:.0f} mm"}
            continue
        if phase == "close":
            for g in G_:
                if jaw_hold[g] is not None:
                    continue
                f_o, f_t = fingers[g]["out"], fingers[g]["thumb"]
                touch_o, touch_t = ms["N"][f_o] > TOUCH_N, ms["N"][f_t] > TOUCH_N
                if (touch_o or touch_t) and not math.isfinite(first_touch[g]):
                    first_touch[g] = float(data.qpos[q_f[f_o]])
                om = abs(float(data.qvel[v_f[f_o]]))
                stall[g] = stall[g] + every * dt if (om < 0.02 and (touch_o or touch_t)) else 0.0
                if (touch_o and touch_t) or stall[g] > 0.03:
                    jaw_hold[g] = float(data.qpos[q_f[f_o]])
                    jaw_note[g] = "both fingers touch" if touch_o and touch_t else (
                        "only the outside finger touches" if touch_o else "only the thumb touches")
            if all(v is not None for v in jaw_hold.values()):
                start_squeeze(t)
            elif t - t_app_end > (preset - jaw.phi_meet) / w_close + 1.0:
                return {"ok": False, "reason": "the jaw never touched the case"}
            continue
        if phase == "squeeze":
            ramp = min((t - t_close_end) / (0.5 * sq_s), 1.0)
            for g in G_:
                data.ctrl[a_frc[g]] = inward[g] * squeeze * ramp
            if t - t_close_end >= sq_s:
                for g in G_:
                    model.actuator_gainprm[a_pos[g], 0] = k_servo
                    model.actuator_biasprm[a_pos[g], 1] = -k_servo
                    data.ctrl[a_pos[g]] = float(data.qpos[q_arm[g]])
                phase, t_sq_end = "timeline", t
                tl = T5Timeline(sc, t, shake_g, (sc.shake_lateral_hz, sc.shake_vertical_hz), float(b1.get("shake_s", 3.0)),
                                stop_g, power_s, cut_before_stop, final_s=cfg.get("final_hold_s"))
                t_end = tl.t_end
                # Revision 2.21's hub readings: sustained over the final hold's last 1 s; peak from the stop (or,
                # with no stop, the power cut) to 5 s after it.
                hub_win = (tl.t_final - float(cfg.get("hub_sustained_s", 1.0)), tl.t_final)
                hub_from = ("stop", tl.stop[0]) if tl.stop is not None else ("power cut", tl.t_power)
                hub_peak_win = (hub_from[1], hub_from[1] + float(cfg.get("hub_peak_window_s", 5.0)))
            continue
        ph = tl.phase(t)
        tau_g = {g: -(float(data.actuator_force[a_drive[fingers[g]["out"]]]) + float(data.actuator_force[a_drive[fingers[g]["thumb"]]]))
                 for g in G_}
        if record:
            rec["t"].append(t), rec["phase"].append(ph), rec["cz"].append(float(data.qpos[q_cz])), rec["cth"].append(float(data.qpos[q_cth]))
            rec["tauA"].append(tau_g["A"]), rec["tauB"].append(tau_g["B"])
            rec["MAo"].append(ms["M"][fingers["A"]["out"]]), rec["MAt"].append(ms["M"][fingers["A"]["thumb"]])
            rec["MBo"].append(ms["M"][fingers["B"]["out"]]), rec["MBt"].append(ms["M"][fingers["B"]["thumb"]])
            rec["hubA"].append(float(np.linalg.norm(ms["hub"]["A"]))), rec["hubB"].append(float(np.linalg.norm(ms["hub"]["B"])))
        if hub_win[0] <= t <= hub_win[1]:
            for g in G_:
                hub_sus[g].append(float(np.linalg.norm(ms["hub"][g])))
        if hub_peak_win[0] <= t <= hub_peak_win[1]:
            hub_pk["t"].append(t)
            for g in G_:
                hub_pk[g].append(ms["hub"][g].copy())
        rel = np.array([data.qpos[q_cx] - data.qpos[qadr("kx")], data.qpos[q_cz] - data.qpos[qadr("kz")]])
        if ref_print is None:
            ref_print = rel.copy()
        if brake_on is None:
            # The thumb's slide on the print: the case against the arms from the end of the squeeze to the cut.
            stats["print_slip"] = max(stats["print_slip"], float(np.linalg.norm(rel - ref_print)))
        if ph in ("settle", "lift"):
            continue
        if ref is None:
            ref = rel.copy()
            for f in body_of:
                stats["off_stop"][f] = math.degrees(stop_phi - float(data.qpos[q_f[f]]))
        slip = float(np.linalg.norm(rel - ref))
        tilt = abs(math.degrees(float(data.qpos[q_cth])))
        stats["slip_all"] = max(stats["slip_all"], slip)
        if brake_on is None:
            stats["slip"] = max(stats["slip"], slip)
            stats["tilt"] = max(stats["tilt"], tilt)
        else:
            stats["power_tilt"] = max(stats["power_tilt"], tilt)
        for g in G_:
            stats["pen_hub"][g] = max(stats["pen_hub"][g], ms["pen_hub"][g])
            stats["pen_finger"][g] = max(stats["pen_finger"][g], ms["pen_finger"][g])
        for f in body_of:
            stats["stop_load"][f] = max(stats["stop_load"][f], limit_load(f))
        # The damage limits hold to the power cut, and through it for revision 2.18's mid-move cut.
        scored = brake_on is None or cut_before_stop
        for g in G_:
            for f in fingers[g].values():
                m_ = (el_gp == G_.index(g) * 2 + (0 if f == fingers[g]["out"] else 1)) & (ms["Fe"] > 1e-6)
                if m_.any() and scored:
                    p = ms["Fe"][m_] / el_area[m_]
                    k = int(np.argmax(p))
                    if p[k] > stats["peak_p"][f]:
                        ei = int(np.nonzero(m_)[0][k])
                        r_ = data.geom_xpos[els[ei][3]][[0, 2]] - data.xpos[body_of[f]][[0, 2]]
                        stats["peak_p"][f] = float(p[k])
                        stats["peak_at"][f] = (ei, float(ms["Fe"][ei]), float(np.linalg.norm(r_)) * 1e3, ph)
                stats["peak_M"][f] = max(stats["peak_M"][f], ms["M"][f])
            if brake_on is None and abs(tau_g[g]) > stats["peak_tau"][g]:
                stats["peak_tau"][g], stats["peak_tau_phase"][g] = abs(tau_g[g]), ph
            hub_n = float(np.linalg.norm(ms["hub"][g]))
            if hub_n > stats["peak_hub"][g]:
                stats["peak_hub"][g], stats["peak_hub_phase"][g] = hub_n, ph
            if ph == "stop" and hub_n > stats["peak_hub_stop"][g]:
                # The case on the hub: outward along W and down.
                sgn = 1.0 if g == "A" else -1.0
                stats["peak_hub_stop"][g] = hub_n
                stats["peak_hub_stop_xz"][g] = (sgn * float(ms["hub"][g][0]), -float(ms["hub"][g][2]))
        if ph == "hold" and t > tl.t_lift + 0.2:
            for g in G_:
                stats["hold"]["tau"][g].append(tau_g[g])
                stats["hold"]["hub"][g].append(ms["hub"][g].copy())
                for f in fingers[g].values():
                    stats["hold"]["M"][f].append(ms["M"][f])
                    stats["hold"]["N"][f].append(ms["N"][f])
                    stats["hold"]["stop_load"][f].append(limit_load(f))
                    if math.isfinite(ms["r"][f]):
                        stats["hold"]["r"][f].append(ms["r"][f])
                        stats["hold"]["span"][f].append(ms["span"][f])
                    for k in ("zc", "zmin", "zmax"):
                        if math.isfinite(ms[k][f]):
                            stats["hold"][k][f].append(ms[k][f])
        if ph == "stop":
            if ref_stop is None:
                ref_stop = (rel.copy(), float(data.qpos[q_cth]))
            stats["stop_slip"] = max(stats["stop_slip"], float(np.linalg.norm(rel - ref_stop[0])))
            stats["stop_tilt"] = max(stats["stop_tilt"], abs(math.degrees(float(data.qpos[q_cth]) - ref_stop[1])))
            for g in G_:
                stats["stop_tau"][g] = max(stats["stop_tau"][g], abs(tau_g[g]))
                for f in fingers[g].values():
                    stats["stop_M"][f] = max(stats["stop_M"][f], ms["M"][f])
        for g in G_:
            for f in fingers[g].values():
                stats["peak_N"][f] = max(stats["peak_N"][f], ms["N"][f])
        if brake_on is not None:
            if ref_power is None:
                ref_power = rel.copy()
            stats["power_slip"] = max(stats["power_slip"], float(np.linalg.norm(rel - ref_power)))
            # The brake's own rows: the joints' friction loss, apart from the gear coupling and the stroke limits.
            ne = data.nefc
            fric = data.efc_type[:ne] == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
            ids, frc = data.efc_id[:ne], data.efc_force[:ne]
            for g in G_:
                per = [float(np.abs(frc[fric & (ids == v_f[f])]).sum()) for f in fingers[g].values()]
                stats["brake_use"][g] = max(stats["brake_use"][g], sum(per))
                if ph == "stop":
                    stats["brake_stop"][g] = max(stats["brake_stop"][g], sum(per))
                if max(per) >= 0.98 * brake / 2.0:
                    stats["brake_sat"][g] = True
        if slip > DROP:
            dropped_in = ph
            if ph == "shake":
                stats["failed_at_g"] = shake_g * _envelope(t - tl.t_hold, tl.shake_s, tl.ramp)[0]
            elif ph == "stop" and tl.stop is not None:
                stats["failed_at_g"] = tl.stop[4] / G
            break

    def mean(v):
        return float(np.mean(v)) if len(v) else math.nan

    def peak_at(f):
        """Where a finger's peak element pressure was: the element, its load, its distance from the pivot, the phase."""
        if stats["peak_at"][f] is None:
            return ""
        ei, F, r, ph = stats["peak_at"][f]
        e = els[ei][2]
        return f"{e.kind} {e.face}, {e.seg:.1f} x {e.w * 1e3:.0f} mm, {F:.1f} N, {r:.0f} mm from the pivot, in the {ph}"

    res: dict[str, Any] = {"ok": not unstable, "reason": UNSTABLE if unstable else "", "unstable_s": unstable_s,
                           "reset_flag": unstable, "substeps": builders["A"].substeps, "case_id": case.id, "case_label": case.label,
                           "hub_contact": hub_contact,
                           "mass_kg": case.mass, "com_offset_W": com_offset, "squeeze_N": squeeze, "stop_g": stop_g,
                           "cut_before_stop": cut_before_stop, "brake_Nm": brake, "power_s": power_s,
                           "mu_bare": builders["A"].mu_bare, "mu_pad": builders["A"].mu_pad,
                           "dropped_in": dropped_in, "t_approach_s": t_app_end, "t_close_s": t_close_end}
    for g in G_:
        f_o, f_t = fingers[g]["out"], fingers[g]["thumb"]
        hub = np.array(stats["hold"]["hub"][g]) if stats["hold"]["hub"][g] else np.zeros((1, 3))
        res.update({
            f"{g}_touch_deg": math.degrees(jaw_hold[g]) if jaw_hold[g] is not None else math.nan,
            f"{g}_first_touch_deg": math.degrees(first_touch[g]) if math.isfinite(first_touch[g]) else math.nan,
            f"{g}_close": jaw_note[g],
            f"{g}_hold_Nm": mean(stats["hold"]["tau"][g]),
            f"{g}_peak_Nm": stats["peak_tau"][g], f"{g}_peak_phase": stats["peak_tau_phase"][g],
            f"{g}_out_M_hold_Nm": mean(stats["hold"]["M"][f_o]), f"{g}_thumb_M_hold_Nm": mean(stats["hold"]["M"][f_t]),
            f"{g}_out_M_peak_Nm": stats["peak_M"][f_o], f"{g}_thumb_M_peak_Nm": stats["peak_M"][f_t],
            f"{g}_out_N_hold": mean(stats["hold"]["N"][f_o]), f"{g}_thumb_N_hold": mean(stats["hold"]["N"][f_t]),
            f"{g}_out_bearing_mm": mean(stats["hold"]["r"][f_o]) * 1e3, f"{g}_thumb_bearing_mm": mean(stats["hold"]["r"][f_t]) * 1e3,
            f"{g}_hub_hold_N": float(np.linalg.norm(hub.mean(axis=0))), f"{g}_hub_hold_x": float(hub.mean(axis=0)[0]),
            f"{g}_hub_hold_z": float(hub.mean(axis=0)[2]), f"{g}_hub_peak_N": stats["peak_hub"][g],
            f"{g}_hub_peak_phase": stats["peak_hub_phase"][g],
            f"{g}_hub_stop_peak_N": stats["peak_hub_stop"][g] if tl.stop is not None else math.nan,
            f"{g}_hub_stop_out_N": stats["peak_hub_stop_xz"][g][0] if tl.stop is not None else math.nan,
            f"{g}_hub_stop_down_N": stats["peak_hub_stop_xz"][g][1] if tl.stop is not None else math.nan,
            f"{g}_thumb_kPa": stats["peak_p"][f_t] / 1e3, f"{g}_out_kPa": stats["peak_p"][f_o] / 1e3,
            f"{g}_thumb_peak_at": peak_at(f_t), f"{g}_out_peak_at": peak_at(f_o),
            f"{g}_stop_Nm": stats["stop_tau"][g], f"{g}_stop_out_M_Nm": stats["stop_M"][f_o],
            f"{g}_stop_thumb_M_Nm": stats["stop_M"][f_t], f"{g}_brake_use_Nm": stats["brake_use"][g],
            f"{g}_brake_stop_Nm": stats["brake_stop"][g], f"{g}_brake_saturated": stats["brake_sat"][g],
            f"{g}_out_peak_N": stats["peak_N"][f_o], f"{g}_thumb_peak_N": stats["peak_N"][f_t],
        })
        res.update(_hub_readings(g, hub_sus[g], hub_pk["t"], hub_pk[g], hub_from, every * dt,
                                 float(cfg.get("hub_peak_avg_s", 0.005))))
        n_hub = res[f"{g}_hub_hold_N"]
        n_all = n_hub + res[f"{g}_out_N_hold"] + res[f"{g}_thumb_N_hold"]
        res[f"{g}_hub_share"] = n_hub / n_all if n_all > 1e-9 else math.nan
        res[f"{g}_hub_pen_mm"] = stats["pen_hub"][g] * 1e3
        res[f"{g}_finger_pen_mm"] = stats["pen_finger"][g] * 1e3
        for part, f in (("out", f_o), ("thumb", f_t)):
            res[f"{g}_{part}_contact_mm"] = mean(stats["hold"]["span"][f]) * 1e3
            res[f"{g}_{part}_off_stop_deg"] = stats["off_stop"][f]
            res[f"{g}_{part}_stop_load_Nm"] = stats["stop_load"][f]
            res[f"{g}_{part}_stop_load_hold_Nm"] = mean(stats["hold"]["stop_load"][f])
            res[f"{g}_{part}_root_MPa"] = root_bound(stats["peak_M"][f], f)
            res[f"{g}_{part}_root_hold_MPa"] = root_bound(res[f"{g}_{part}_M_hold_Nm"], f)
        res[f"{g}_thumb_height_mm"] = mean(stats["hold"]["zc"][f_t]) * 1e3
        lo, hi = mean(stats["hold"]["zmin"][f_t]) * 1e3, mean(stats["hold"]["zmax"][f_t]) * 1e3
        res[f"{g}_thumb_zmin_mm"], res[f"{g}_thumb_zmax_mm"] = lo, hi
        if perf_z is not None and math.isfinite(lo):
            zp = perf_z * 1e3
            res[f"{g}_thumb_perf_mm"] = 0.0 if lo <= zp <= hi else min(abs(lo - zp), abs(hi - zp))
    res.update({"tilt_deg": stats["tilt"], "slip_mm": stats["slip"] * 1e3, "stop_slip_mm": stats["stop_slip"] * 1e3,
                "stop_tilt_deg": stats["stop_tilt"], "power_slip_mm": stats["power_slip"] * 1e3,
                "power_tilt_deg": stats["power_tilt"], "slip_all_mm": stats["slip_all"] * 1e3,
                "print_slip_mm": stats["print_slip"] * 1e3, "failed_at_g": stats["failed_at_g"],
                "power_cut_held": (not dropped_in) and not unstable and stats["power_slip"] < DROP})
    thumb_lim = case.material.allow_pressure / 1e3
    thumb = max(res["A_thumb_kPa"], res["B_thumb_kPa"])
    res["thumb_limit_kPa"] = thumb_lim
    res["thumb_bed_margin"] = float(cfg.get("bed_limit_kPa", 25.0)) / max(thumb, 1e-9)
    if record:
        res["trace"] = rec
    if unstable:
        # Test conditions 2.15: a run that blew up is never scored; batch 1 reruns it at a finer timestep.
        res.update(supported=None, passed=None, score=None, limiting="", margins={}, power_cut_held=None)
        return res
    tilt = max(stats["tilt"], stats["power_tilt"]) if cut_before_stop else stats["tilt"]
    margins = {"slip": DROP / max(stats["slip"], 1e-9),
               "tilt": float(cfg.get("max_tilt_deg", 10.0)) / max(tilt, 1e-9),
               "thumb": thumb_lim / max(thumb, 1e-9)}
    if ccfg.get("print_slip_mm") is not None:
        # The shipper: the thumb slides at most 2 mm on the print and bears at least 25 mm from a perforation.
        margins["print slip"] = float(ccfg["print_slip_mm"]) / max(res["print_slip_mm"], 1e-9)
    if perf_z is not None:
        clear = min(res.get(f"{g}_thumb_perf_mm", math.inf) for g in G_)
        if math.isfinite(clear):
            margins["perforation"] = clear / float(ccfg.get("perforation_clear_mm", 25.0))
    if cut_before_stop:
        # Revision 2.18: the mid-move cut passes only if the case stays held through the stop and the 5 s after it.
        margins["power cut"] = DROP / max(stats["power_slip"], 1e-9)
    res["margins"] = margins
    res["score"] = float(min(margins.values()))
    res["limiting"] = min(margins, key=margins.get)
    # Every other run reports the power cut at its end (test conditions, T5 Power cut row) without failing on it.
    res["supported"] = not dropped_in if cut_before_stop else dropped_in in ("", "power")
    res["passed"] = bool(res["supported"] and res["score"] >= 1.0)
    return res
