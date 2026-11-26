"""T4, the crate (test conditions revision 2.22): the geometric T4a lip span, and both cases in MuJoCo.

T4a's crate is a vertical 4 mm wall whose outer face is at x = W, with a lip 6 mm out and 15 mm tall at its top and a
square underside. The jaw opens to phi_d (within the reading), descends with 4 mm of clearance beside the wall and
lip, closes at that height until the finger touches the crate, then lifts. The finger hooks when it touches the
wall's outer face below the lip without overlapping the crate, and a 3 mm lift brings finger material up under the
lip. The widest span is the widest 2 (W + lip) for which the hook's whole path is clear: the descent keeps every
finger point below the crate top out of the corridor beside wall and lip, and the close from phi_d never overlaps
the crate. This is FP's memo, "T4a lip spans" (`fp-t4a-lip-spans.py`), run on the worker's own finger outlines,
body and pads, at their built values.

In MuJoCo (x-z plane, gravity -z) the container is rigid and stands on a table. T4a: one gripper spans the crate
and hooks both lips. T4b: one gripper on each long wall, the outside finger's undercut on the lip and the inside
finger on the inner face; the right gripper is the spec's jaw and the left its mirror, so each outside finger is
the spec's outside finger. Each jaw opens to clear the lips, descends to the hook height, and closes on the drive
torque with its arm free along x, so both fingers find the wall; the arms then lock, squeeze, lift, hold, shake,
make the stop and end on the power cut. Bare faces take the bare finger's friction on HDPE or PP (stainless from
revision 2.34, PA12 on the baseline), and pads a pad's.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import box
from shapely.ops import unary_union

from .jaw import Jaw
from .mj1 import BAD_STATE, DROP, GRIP, OBJ, TABLE, _fmt, _min_jerk
from .objects import G, GraspObject, Material
from .scenario import Scenario
from .sensor import Wrench, worst
from .sensor import config as sensor_config
from .t5 import T5Timeline, _GripperBuilder, mirrored

TOL_MM2 = 0.05  # overlap area treated as touching
TOUCH_N = 0.5  # N on a finger that counts as touching the container
SEATED_N = 1.0  # N up on a lip that counts as seated
LIP = {"out_mm": 6.0, "height_mm": 15.0, "wall_mm": 4.0, "clear_mm": 4.0, "free_wall_mm": 30.0}


def t4_lip(b1: dict[str, Any] | None = None) -> dict[str, float]:
    return {**LIP, **{k: float(v) for k, v in ((b1 or {}).get("t4", {}) or {}).get("lip", {}).items()}}


def _world(poly, phi_deg: float, pivot_x: float):
    s, c = math.sin(math.radians(phi_deg)), math.cos(math.radians(phi_deg))
    return affine_transform(poly, [s, -c, -c, -s, pivot_x, 0.0])


def _mirror(poly):
    return affine_transform(poly, [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def _crate(W: float, zA: float, lip: dict[str, float]):
    p, h, t = lip["out_mm"], lip["height_mm"], lip["wall_mm"]
    return unary_union([box(W - t, zA - 230.0, W, zA + h), box(W, zA, W + p, zA + h)])


def _overlap(a, b) -> bool:
    return a.intersection(b).area > TOL_MM2


def _u_lo(jaw: Jaw, side: int) -> float:
    src = getattr(jaw, "source", None)
    feats = {}
    if src is not None and hasattr(src, "right"):
        feats = (src.right if side > 0 else src.left).features or {}
    uc = feats.get("undercut")
    return float(uc["u_from"]) - 5.0 if uc else jaw.reach - 60.0


def _phis(jaw: Jaw, step: float) -> list[float]:
    lo = max(step, math.ceil(math.degrees(jaw.phi_meet) / step) * step)
    phi_max = math.degrees(jaw.phi_max)
    return [lo + step * i for i in range(int((phi_max - lo) / step + 1e-9) + 1)]


def _seat_gap(F, W: float, zA: float, lip: dict[str, float]) -> float:
    """How far (mm) the finger's seat sits under the lip's underside: the top of its material under the lip."""
    under = F.intersection(box(W + 0.5, zA - 3.0, W + lip["out_mm"], zA))
    return zA - under.bounds[3] if not under.is_empty else math.inf


def _hooked_states(fing, u_lo: float, phis: list[float], px: float, lip: dict[str, float]):
    p = lip["out_mm"]
    distal = fing.intersection(box(u_lo, -60, 400, 80))
    for phi in phis:
        F, D = _world(fing, phi, px), _world(distal, phi, px)
        if D.is_empty:
            continue
        zmin, zmax = D.bounds[1], D.bounds[3]
        zA = zmin + 0.5
        while zA < zmax:
            below = D.intersection(box(-500, -1000, 500, zA))
            if not below.is_empty:
                W = below.bounds[0]
                if not _overlap(F, _crate(W, zA, lip)):
                    if F.intersection(box(W + 0.5, zA - 3.0, W + p, zA)).area > 0.2:
                        yield W, phi, zA
            zA += 0.5


def _reachable(fing, W: float, phi_h: float, zA: float, phi_max: float, px: float, lip: dict[str, float]):
    C = _crate(W, zA, lip)
    corridor = box(W - lip["wall_mm"] - lip["clear_mm"], -1000, W + lip["out_mm"] + lip["clear_mm"], zA + lip["height_mm"])
    phi_d = phi_h
    while phi_d <= phi_max + 1e-9:
        if not _overlap(_world(fing, phi_d, px), corridor):
            p = phi_d
            while p > phi_h + 1e-9:
                if _overlap(_world(fing, p, px), C):
                    break
                p -= 0.25
            else:
                return phi_d
        phi_d += 0.5
    return None


def finger_span(jaw: Jaw, side: int, lip: dict[str, float]) -> dict[str, Any]:
    """The widest lip span (mm) one finger clears and hooks within the jaw's stroke, the angles per finger (deg) at
    which it hooks and opens, or None."""
    fing = jaw.sides[side].outline
    u_lo = _u_lo(jaw, side)
    phi_max = math.degrees(jaw.phi_max)
    for W, phi_h, zA in sorted(_hooked_states(fing, u_lo, _phis(jaw, 0.5), jaw.pivot_x, lip), reverse=True):
        phi_d = _reachable(fing, W, phi_h, zA, phi_max, jaw.pivot_x, lip)
        if phi_d is not None:
            return {"span_mm": 2 * (W + lip["out_mm"]), "hook_deg": phi_h, "open_deg": phi_d, "u_lo": u_lo}
    return {"span_mm": None, "hook_deg": None, "open_deg": None, "u_lo": u_lo}


def t4a_span(jaw: Jaw, lip: dict[str, float]) -> dict[str, Any]:
    """T4a's widest span for the jaw: the narrower finger's, since both hook opposite walls."""
    per = {s: finger_span(jaw, s, lip) for s in ((1,) if jaw.symmetric else (1, -1))}
    worst = min(per.values(), key=lambda r: r["span_mm"] or 0.0)
    out = {"t4a_span_mm": worst["span_mm"], "t4a_hook_deg": worst["hook_deg"], "t4a_open_deg": worst["open_deg"]}
    for s, r in per.items():
        tag = "R" if s > 0 else "L"
        out.update({f"t4a_span_{tag}_mm": r["span_mm"], f"t4a_hook_{tag}_deg": r["hook_deg"],
                    f"t4a_open_{tag}_deg": r["open_deg"]})
    return out


def _depths(FR, FL, zA: float, lip: dict[str, float]) -> dict[str, float]:
    return {"outside_below_lip_mm": zA - FR.bounds[1], "inside_below_rim_mm": zA + lip["height_mm"] - FL.bounds[1]}


def t4a_pose(jaw: Jaw, W: float, lip: dict[str, float]) -> dict[str, Any]:
    """Where a symmetric jaw hooks both lips of a crate whose outer faces are at +-W (mm): the hook angle, the lip's
    underside below the pivot line, and the opening that clears the lips on the way down."""
    if not jaw.symmetric:
        return {"ok": False, "reason": "an asymmetric pair hooks the two walls at different angles"}
    fing, px = jaw.sides[1].outline, jaw.pivot_x
    distal = fing.intersection(box(_u_lo(jaw, 1), -60, 400, 80))
    lo, hi = max(0.25, math.degrees(jaw.phi_meet)), math.degrees(jaw.phi_max)

    def w_below(phi: float, zA: float) -> float:
        below = _world(distal, phi, px).intersection(box(-500, -1000, 500, zA))
        return below.bounds[0] if not below.is_empty else -math.inf

    D_hi = _world(distal, hi, px)
    best = None
    zA = D_hi.bounds[1] + 0.5
    while zA < max(D_hi.bounds[3], _world(distal, lo, px).bounds[3]):
        if w_below(hi, zA) >= W:
            a, b = lo, hi
            if w_below(a, zA) < W:
                for _ in range(30):
                    m = 0.5 * (a + b)
                    a, b = (m, b) if w_below(m, zA) < W else (a, m)
                phi = b
                F = _world(fing, phi, px)
                if not _overlap(F, _crate(W, zA, lip)):
                    gap = _seat_gap(F, W, zA, lip)
                    if math.isfinite(gap) and (best is None or (gap >= 0.25, -gap) > (best[2] >= 0.25, -best[2])):
                        best = (phi, zA, gap)
        zA += 0.25
    if best is None:
        return {"ok": False, "reason": f"no hook on a {2 * (W + lip['out_mm']):.0f} mm lip span within the stroke"}
    phi_h, zA, gap = best
    phi_d = _reachable(fing, W, phi_h, zA, hi, px, lip)
    if phi_d is None:
        return {"ok": False, "reason": f"hooks at {phi_h:.1f} deg but can't clear the lips on the way down"}
    F = _world(fing, phi_h, px)
    return {"ok": True, "hook_deg": phi_h, "open_deg": phi_d, "zA_mm": zA, "W_mm": W, "seat_gap_mm": gap,
            **_depths(F, F, zA, lip)}


def t4b_pose(jaw: Jaw, lip: dict[str, float]) -> dict[str, Any]:
    """Where the jaw closes on a wall with the outside (right) finger's undercut round the lip and the inside finger
    on the inner face: the smallest angle at which the wall fits between them with a hook. The wall's outer face is
    W (mm) from the jaw's centreline and the lip's underside zA below the pivot line."""
    fR, fL, px = jaw.sides[1].outline, jaw.sides[-1].outline, jaw.pivot_x
    t, p = lip["wall_mm"], lip["out_mm"]
    distal = fR.intersection(box(_u_lo(jaw, 1), -60, 400, 80))
    hi = math.degrees(jaw.phi_max)
    for phi in _phis(jaw, 0.25):
        FR, FL, D = _world(fR, phi, px), _mirror(_world(fL, phi, px)), _world(distal, phi, px)
        if D.is_empty:
            continue
        cands = []
        zA = D.bounds[1] + 0.5
        while zA < D.bounds[3]:
            below = D.intersection(box(-500, -1000, 500, zA))
            if not below.is_empty:
                W = below.bounds[0]
                C = _crate(W, zA, lip)
                if not _overlap(FR, C) and not _overlap(FL, C):
                    gap = _seat_gap(FR, W, zA, lip)
                    if math.isfinite(gap):
                        band = FL.intersection(box(-1000, zA - 230.0, W - t, zA + lip["height_mm"]))
                        inner = (W - t) - band.bounds[2] if not band.is_empty else math.inf
                        cands.append((gap >= 0.25, -gap, W, zA, inner))
            zA += 0.25
        if not cands:
            continue
        _, neg_gap, W, zA, inner = max(cands)
        corridor = box(W - t - lip["clear_mm"], -1000, W + p + lip["clear_mm"], zA + lip["height_mm"])
        C = _crate(W, zA, lip)
        phi_d = phi
        while phi_d <= hi + 1e-9:
            if not _overlap(_world(fR, phi_d, px), corridor) and not _overlap(_mirror(_world(fL, phi_d, px)), corridor):
                q = phi_d
                while q > phi + 1e-9:
                    if _overlap(_world(fR, q, px), C) or _overlap(_mirror(_world(fL, q, px)), C):
                        break
                    q -= 0.25
                else:
                    break
            phi_d += 0.5
        else:
            return {"ok": False, "reason": f"closes on the wall at {phi:.2f} deg but can't clear the lip on the way down"}
        return {"ok": True, "hook_deg": phi, "open_deg": phi_d, "zA_mm": zA, "W_mm": W, "seat_gap_mm": -neg_gap,
                "inside_gap_mm": inner, **_depths(FR, FL, zA, lip)}
    return {"ok": False, "reason": "no angle within the stroke at which the wall fits between the fingers with a hook"}


def _material(cfg: dict[str, Any], name: str, friction_scale: float) -> Material:
    m = (cfg.get("materials") or {}).get(name, {})
    E = float(m.get("modulus_MPa", 1000.0)) * 1e6
    return Material(name=name, friction=float(m.get("pad_friction", 0.4)), k_surface=E / (LIP["wall_mm"] * 1e-3),
                    allow_pressure=math.inf, bare_friction=float(m.get("bare_friction", 0.2)))


def container(b1: dict[str, Any], case: str, which: str | None = None) -> dict[str, Any]:
    """The container's dimensions (m), mass and material: T4a's crate, or T4b's tote or milk crate."""
    cfg = b1.get("t4", {}) or {}
    c = dict(cfg["t4a"]) if case == "t4a" else dict(cfg["t4b"][which or "tote"])
    return {"key": "t4a" if case == "t4a" else (which or "tote"), "label": str(c.get("label", case)),
            "across": float(c["across_mm"]) * 1e-3, "length": float(c["length_mm"]) * 1e-3,
            "height": float(c["height_mm"]) * 1e-3, "mass": float(c.get("mass_kg", cfg.get("mass_kg", 15.0))),
            "com_below_rim": float(c.get("com_below_rim_mm", float(c["height_mm"]) / 2)) * 1e-3,
            "material": str(c.get("material", "hdpe"))}


def simulate_t4(jaw: Jaw, sc: Scenario, b1: dict[str, Any], case: str = "t4b", which: str | None = None, *,
                friction_scale: float = 1.0, height_mm: float = 0.0, shake_g: float | None = None,
                stop_g: float | None = None, stop_sign: float = 1.0, brake: float | None = None,
                power_s: float | None = None, tau: float | None = None, record: bool = False) -> dict[str, Any]:
    """One T4 run. `case` is "t4a" (one gripper spans the crate) or "t4b" (two arms, `which` container);
    `height_mm` raises the jaw that far above the planned hook height (the +-3 mm placement corners); `stop_sign`
    is the stop's direction along x, toward gripper A's outside finger when positive; `brake` is N m per gripper."""
    cfg = b1.get("t4", {}) or {}
    lip = t4_lip(b1)
    ct = container(b1, case, which)
    tau = jaw.torque if tau is None else float(tau)
    shake_g = float((b1.get("shake_g") or {}).get("T4", 0.25)) if shake_g is None else float(shake_g)
    stop_g = float(cfg.get("stop_g", 1.0)) if stop_g is None else float(stop_g)
    pc = b1.get("power_cut") or {}
    brake = float(pc.get("brake_Nm_per_gripper", 30.0)) if brake is None else float(brake)
    power_s = float(pc.get("hold_s", 5.0)) if power_s is None else float(power_s)
    Xo = ct["across"] / 2
    row: dict[str, Any] = {"case": case, "container": ct["key"], "friction_scale": friction_scale, "height_mm": height_mm,
                           "shake_g": shake_g, "stop_g": stop_g, "stop_sign": stop_sign, "brake_Nm": brake,
                           "finger_torque_Nm": tau, "mass_kg": ct["mass"]}
    pose = t4a_pose(jaw, Xo * 1e3, lip) if case == "t4a" else t4b_pose(jaw, lip)
    row.update({f"pose_{k}": v for k, v in pose.items() if k not in ("ok", "reason")})
    if not pose["ok"]:
        return {**row, "ok": True, "passed": False, "score": 0.0, "limiting": "geometry",
                "reason": pose["reason"] + "; geometric, no MuJoCo run"}
    free = float(lip["free_wall_mm"])
    room = float(cfg.get("contents_below_rim_mm", 50.0))
    if pose["outside_below_lip_mm"] > free + 1e-6:
        return {**row, "ok": True, "passed": False, "score": free / pose["outside_below_lip_mm"], "limiting": "geometry",
                "reason": f"the outside finger reaches {pose['outside_below_lip_mm']:.1f} mm below the lip, past its "
                          f"{free:g} mm of free wall; geometric, no MuJoCo run"}
    if case == "t4b" and pose["inside_below_rim_mm"] > room + 1e-6:
        return {**row, "ok": True, "passed": False, "score": room / pose["inside_below_rim_mm"], "limiting": "geometry",
                "reason": f"the inside finger reaches {pose['inside_below_rim_mm']:.1f} mm below the rim, into the "
                          f"contents {room:g} mm down; geometric, no MuJoCo run"}
    phi_h, phi_d = math.radians(pose["hook_deg"]), math.radians(pose["open_deg"])
    mat = _material(cfg, ct["material"], friction_scale)
    obj = GraspObject(id=f"t4_{ct['key']}", label=ct["label"], kind="crate", thickness=ct["across"], height=ct["height"],
                      width=ct["length"], mass=ct["mass"], com_below_top=ct["com_below_rim"], inertia=0.0, material=mat)
    k_servo = float(sc.sim.get("servo_kp", 2.0e6))
    m = ct["mass"]
    kv_carr = 2 * math.sqrt(k_servo * (2.0 + 1.2 + m))
    kv_arm = 2 * math.sqrt(k_servo * (0.6 + m / 2))
    grippers = [("A", 1.0, jaw)] if case == "t4a" else [("A", 1.0, jaw), ("B", -1.0, mirrored(jaw))]
    x_axis = 0.0 if case == "t4a" else Xo - pose["W_mm"] * 1e-3
    builders, grip_xml, assets = {}, [], []
    for g, sgn, jw in grippers:
        b = _GripperBuilder(g, jw, obj, sc, b1, 0.0, {1: phi_h, -1: phi_h}, friction_scale, 1.0, 0.0)
        builders[g] = b
        fingers = []
        for side in (1, -1):
            xml, m_f, mom = b.finger_xml(side)
            nm = f"{g}{'R' if side > 0 else 'L'}"
            fingers.append(
                f'<body name="finger{nm}" pos="{_fmt((side * jw.pivot_x * 1e-3, 0, 0))}" gravcomp="1">'
                f'<joint name="j{nm}" type="hinge" axis="0 {-side} 0" range="{_fmt(jw.phi_meet)} {_fmt(jw.phi_max + 1e-3)}" '
                f'armature="{_fmt(float(b1.get("closing", {}).get("armature_kgm2", 0.015)))}" damping="0" '
                f'solreffriction="{_fmt(2 * b.dt)} 1" solimpfriction="0.999 0.9999 0.001 0.5 2"/>'
                f'<inertial pos="0 0 {_fmt(-jw.reach * 0.5e-3)}" mass="{_fmt(max(m_f, 1e-3))}" diaginertia="{_fmt((max(mom, 1e-6),) * 3)}"/>'
                f"{xml}</body>")
        grip_xml.append(
            f'<body name="arm{g}" pos="{_fmt((sgn * x_axis, 0, 0))}" gravcomp="1">'
            f'<joint name="a{g}" type="slide" axis="1 0 0"/>'
            '<inertial pos="0 0 0" mass="0.6" diaginertia="1e-3 1e-3 1e-3"/>'
            f'<body name="grip{g}" gravcomp="1">{"".join(fingers)}</body></body>')
        assets += b.assets
    bA = builders["A"]
    if any(e.seg <= 2.0 + 1e-9 for bb in builders.values() for e in bb.elements):
        bA.substeps = max(bA.substeps, int(round(bA.dt / bA.fine_dt)))
    dt = bA.dt / bA.substeps
    H, L, t, p, h = ct["height"], ct["length"], lip["wall_mm"] * 1e-3, lip["out_mm"] * 1e-3, lip["height_mm"] * 1e-3
    I = m * (ct["across"] ** 2 + H * H) / 12.0
    mu_t = float(cfg.get("table_friction", 0.3))
    walls = []
    for sgn, tag in ((1, "P"), (-1, "N")):
        walls.append(f'<geom name="wall{tag}" type="box" pos="{_fmt((sgn * (Xo - t / 2), 0, -H / 2))}" '
                     f'size="{_fmt((t / 2, L / 2, H / 2))}" mass="0" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.3 0.5 0.75 1"/>')
        walls.append(f'<geom name="lip{tag}" type="box" pos="{_fmt((sgn * (Xo + p / 2), 0, -h / 2))}" '
                     f'size="{_fmt((p / 2, L / 2, h / 2))}" mass="0" contype="{OBJ}" conaffinity="{GRIP | TABLE}" rgba="0.25 0.45 0.7 1"/>')
    walls.append(f'<geom name="floor" type="box" pos="{_fmt((0, 0, -H + t / 2))}" size="{_fmt((Xo - t, L / 2, t / 2))}" '
                 f'mass="0" contype="{OBJ}" conaffinity="{TABLE}" rgba="0.3 0.5 0.75 1"/>')
    # The pivot line sits zA below... the lip's underside sits zA (negative) below the pivot line at the hook.
    z_lip = H - h
    z_piv = z_lip - pose["zA_mm"] * 1e-3 + height_mm * 1e-3
    lows = []
    for g, sgn, jw in grippers:
        FR = _world(jw.sides[1].outline, pose["open_deg"], jw.pivot_x)
        FL = _world(jw.sides[-1].outline, pose["open_deg"], jw.pivot_x)
        lows.append(min(FR.bounds[1], FL.bounds[1]))
    d_up = max(0.0, (H + float(cfg.get("approach_clear_mm", 5.0)) * 1e-3) - (z_piv + min(lows) * 1e-3))
    xml = f"""<mujoco model="t4">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{_fmt(dt)}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10" iterations="60" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default><geom condim="3" solref="{_fmt(bA.contact_tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"/></default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="table" type="box" pos="0 0 -0.01" size="0.6 0.4 0.01" friction="{_fmt(mu_t)} 0.005 0.0001" priority="1"
          contype="{TABLE}" conaffinity="{OBJ | GRIP}" rgba="0.8 0.8 0.8 1"/>
    <body name="crate" pos="0 0 {_fmt(H + 2e-4)}">
      <joint name="cx" type="slide" axis="1 0 0"/><joint name="cz" type="slide" axis="0 0 1"/>
      <joint name="cth" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {_fmt(-ct['com_below_rim'])}" mass="{_fmt(m)}" diaginertia="{_fmt((I, I, I))}"/>
      {''.join(walls)}
    </body>
    <body name="carriage" pos="0 0 {_fmt(z_piv + d_up)}" gravcomp="1">
      <joint name="kx" type="slide" axis="1 0 0"/><joint name="kz" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="1e-3 1e-3 1e-3"/>
      {''.join(grip_xml)}
    </body>
  </worldbody>
  <equality>
    {''.join(f'<joint name="gear{g}" joint1="j{g}L" joint2="j{g}R" polycoef="0 1 0 0 0" solref="{_fmt(4 * bA.dt)} 1"/>' for g, _, _ in grippers)}
  </equality>
  <actuator>
    <position name="kx" joint="kx" kp="{_fmt(k_servo)}" kv="{_fmt(kv_carr)}"/>
    <position name="kz" joint="kz" kp="{_fmt(k_servo)}" kv="{_fmt(kv_carr)}"/>
    {''.join(f'<position name="p{g}" joint="a{g}" kp="{_fmt(k_servo)}" kv="{_fmt(kv_arm)}"/>' for g, _, _ in grippers)}
    {''.join(f'<motor name="d{g}R" joint="j{g}R" gear="1"/><motor name="d{g}L" joint="j{g}L" gear="1"/>' for g, _, _ in grippers)}
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    data = mujoco.MjData(model)
    row.update(z_start_mm=d_up * 1e3, arm_x_mm=x_axis * 1e3, elements=sum(len(bb.elements) for bb in builders.values()),
               substeps=bA.substeps)
    return _run(model, data, sc, b1, cfg, builders, [g for g, _, _ in grippers], case, row, phi_h, phi_d, d_up, tau,
                brake, power_s, shake_g, stop_g, stop_sign, k_servo, kv_arm, record)


def _run(model, data, sc: Scenario, b1: dict, cfg: dict, builders: dict, gs: list[str], case: str, row: dict[str, Any],
         phi_h: float, phi_d: float, d_up: float, tau: float, brake: float, power_s: float, shake_g: float,
         stop_g: float, stop_sign: float, k_servo: float, kv_arm: float, record: bool) -> dict[str, Any]:
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    dadr = lambda n: model.jnt_dofadr[jid(n)]
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    for g in gs:
        data.qpos[qadr(f"j{g}R")] = data.qpos[qadr(f"j{g}L")] = phi_d
    mujoco.mj_forward(model, data)
    # Which finger of which gripper each geom belongs to, and what of the container each container geom is.
    owner = np.full(model.ngeom, -1, dtype=int)
    for gi, g in enumerate(gs):
        b = builders[g]
        for gname, s in b.body_geoms.items():
            owner[gid(gname)] = 2 * gi + (0 if s > 0 else 1)
        for e in b.elements:
            owner[gid(e.geom)] = 2 * gi + (0 if e.side > 0 else 1)
    part = np.full(model.ngeom, -1, dtype=int)
    for k, nm in enumerate(("lipP", "lipN", "wallP", "wallN")):
        part[gid(nm)] = k
    # The finger seating each lip, by owner index (2 x gripper, + 1 for its left finger), and the lip's part: T4a's
    # gripper hooks the +x lip with its right finger and the -x lip with its left; in T4b A's right finger takes the
    # +x lip and B's left, the mirrored jaw's outside finger, the -x lip.
    hook = {0: (0, 0), 1: (1, 1)} if case == "t4a" else {0: (0, 0), 3: (1, 1)}
    f_b = max(hook)
    a_kx, a_kz = aid("kx"), aid("kz")
    arm_act = {g: aid(f"p{g}") for g in gs}
    q_arm = {g: qadr(f"a{g}") for g in gs}
    q_j = {g: (qadr(f"j{g}R"), qadr(f"j{g}L")) for g in gs}
    d_j = {g: (dadr(f"j{g}R"), dadr(f"j{g}L")) for g in gs}
    drive = {g: (aid(f"d{g}R"), aid(f"d{g}L")) for g in gs}
    q_cx, q_cz, q_cth = qadr("cx"), qadr("cz"), qadr("cth")
    bA = builders[gs[0]]
    every = 4 * bA.substeps
    cl = b1.get("closing", {})
    w_slow = float(cfg.get("close_rad_s", cl.get("slow_rad_s", 0.3)))
    kp_j = float(cfg.get("jaw_servo_Nm_rad", 2.0e4))
    kv_j = 2 * math.sqrt(kp_j * float(cl.get("armature_kgm2", 0.015)))
    u_peak = float(b1.get("drive_Nm_per_gripper", {}).get("peak", 25.0)) / 2
    kv_drive = tau / 0.2
    v_desc = float(cfg.get("descend_m_s", 0.05))
    t_desc = max(d_up / v_desc, 0.05)
    t_close0 = t_desc + 0.05
    t_close = t_close0 + (phi_d - phi_h) / w_slow + float(cfg.get("close_extra_s", 0.4))
    t0 = t_close + sc.squeeze_time
    tl = T5Timeline(sc, t0, shake_g, (2.0, 3.0), float(b1.get("shake_s", 3.0)) if shake_g > 0 else 0.0,
                    stop_g or None, power_s, stop_sign=stop_sign, final_s=float(cfg.get("final_hold_s", 1.0)))
    kv_float = float(cfg.get("arm_float_damping_Ns_m", 40.0))
    gain0 = {g: (model.actuator_gainprm[a, 0], model.actuator_biasprm[a, 1], model.actuator_biasprm[a, 2]) for g, a in arm_act.items()}
    for a in arm_act.values():
        model.actuator_gainprm[a, 0] = 0.0
        model.actuator_biasprm[a, 1] = 0.0
        model.actuator_biasprm[a, 2] = -kv_float
    kp_c, kv_c = float(model.actuator_gainprm[a_kx, 0]), float(-model.actuator_biasprm[a_kx, 2])
    nf = 2 * len(gs)
    f6 = np.zeros(6)
    n_avg = max(1, int(round(0.005 / (every * model.opt.timestep))))
    # Rev 2.34: each gripper's wrist sensor; its whole window runs from the end of the close to the power cut.
    wrench = {g: Wrench(model, b1, f"grip{g}", builders[g].geom_depth, every * model.opt.timestep, mirror=g == "B")
              for g in gs}
    hold_s = float(sensor_config(b1).get("hold_s", 1.0))

    def phase_at(t: float) -> str:
        if t < t_desc:
            return "descend"
        if t < t_close:
            return "close"
        return "squeeze" if t < t0 else tl.phase(t)

    def contacts():
        """World force (N) on each lip and wall from each finger: F[finger, part] = (Fx, Fz)."""
        F = np.zeros((nf, 4, 2))
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if owner[g1] >= 0 and part[g2] >= 0:
                f, pr, sgn = owner[g1], part[g2], 1.0
            elif owner[g2] >= 0 and part[g1] >= 0:
                f, pr, sgn = owner[g2], part[g1], -1.0
            else:
                continue
            mujoco.mj_contactForce(model, data, i, f6)
            fr = np.asarray(c.frame).reshape(3, 3)
            w = sgn * (f6[0] * fr[0] + f6[1] * fr[1] + f6[2] * fr[2])
            F[f, pr] += (w[0], w[2])
        return F

    ref = ref_th = None
    slip = slip_pk = tilt_pk = 0.0
    power_ref = None
    power_slip = 0.0
    dropped = ""
    unstable, unstable_s = False, math.nan
    locked = False
    close_state: dict[str, Any] = {}
    seat_series: dict[int, list[float]] = {f: [] for f in hook}
    seat_t: list[float] = []
    seat_min = {f: math.inf for f in hook}
    seat_pk = {f: 0.0 for f in hook}
    seat_buf = {f: [] for f in hook}
    cam = {g: 0.0 for g in gs}
    phi_lock = {}
    brake_on = None
    trace: dict[str, list[float]] = {k: [] for k in ("t", "phi", "slip", "seatA", "seatB", "kz", "cx")} if record else {}
    i = -1
    while data.time < tl.t_end:
        i += 1
        t = data.time
        for g in gs:
            qR, qL = q_j[g]
            phi, om = float(data.qpos[qR]), float(data.qvel[d_j[g][0]])
            if t < t_close0:
                u = min(max(kp_j * (phi_d - phi) - kv_j * om, -u_peak), u_peak)
            elif t < t_close:
                u = -min(max(kv_drive * (w_slow + om), 0.0), tau)
            elif t < tl.t_power:
                u = -tau
            else:
                u = 0.0
            data.ctrl[drive[g][0]] = data.ctrl[drive[g][1]] = u
        if not locked and t >= t_close:
            for g, a in arm_act.items():
                kp0, b1_, b2_ = gain0[g]
                model.actuator_gainprm[a, 0], model.actuator_biasprm[a, 1], model.actuator_biasprm[a, 2] = kp0, b1_, b2_
                data.ctrl[a] = float(data.qpos[q_arm[g]])
            locked = True
            Fc = contacts()
            for gi, g in enumerate(gs):
                phi_lock[g] = float(data.qpos[q_j[g][0]])
                for s, k in ((0, "R"), (1, "L")):
                    f = 2 * gi + s
                    close_state[f"{g}{k}_touch_N"] = float(np.hypot(*Fc[f].sum(axis=0)))
                close_state[f"{g}_close_deg"] = math.degrees(phi_lock[g])
                close_state[f"{g}_arm_shift_mm"] = (float(data.qpos[q_arm[g]])) * 1e3
        if t < t_desc:
            pz, dpz = _min_jerk(t / t_desc)
            zc, vzc = -d_up * pz, -d_up * dpz / t_desc
            xc = vxc = 0.0
        elif t < t0:
            zc, vzc, xc, vxc = -d_up, 0.0, 0.0, 0.0
        else:
            xc, zc, vxc, vzc = tl(t)
            zc -= d_up
        data.ctrl[a_kx] = xc + kv_c / kp_c * vxc
        data.ctrl[a_kz] = zc + kv_c / kp_c * vzc
        if t >= tl.t_power and brake_on is None:
            for g in gs:
                for d in d_j[g]:
                    model.dof_frictionloss[d] = brake / 2.0
            brake_on = t
        t_step = data.time
        mujoco.mj_step(model, data)
        if (data.time < t_step or any(data.warning[w].number for w in BAD_STATE) or not np.isfinite(data.qacc).all()):
            unstable, unstable_s = True, t_step
            break
        if i % every:
            continue
        t_now = data.time
        for g in gs:
            wrench[g].sample(model, data, t_now, phase_at(t_now), t_close <= t_now < tl.t_power,
                             tl.t_final - hold_s <= t_now < tl.t_final)
        if t < tl.t_settle:
            continue
        t = data.time
        rel = np.array([data.qpos[q_cx] - data.qpos[qadr("kx")], data.qpos[q_cz] - data.qpos[qadr("kz")]])
        if ref is None:
            ref, ref_th = rel.copy(), float(data.qpos[q_cth])
        slip = float(np.hypot(*(rel - ref)))
        Fc = contacts()
        for f, (pr, _) in hook.items():
            fz = float(Fc[f, pr, 1])
            seat_buf[f].append(fz)
            if len(seat_buf[f]) > n_avg:
                seat_buf[f].pop(0)
            if t >= tl.t_lift and t < tl.t_power:
                seat_min[f] = min(seat_min[f], sum(seat_buf[f]) / len(seat_buf[f]))
                seat_pk[f] = max(seat_pk[f], sum(seat_buf[f]) / len(seat_buf[f]))
                seat_series[f].append(fz)
        if t >= tl.t_lift and t < tl.t_power:
            seat_t.append(t)
            for g in gs:
                cam[g] = max(cam[g], float(data.qpos[q_j[g][0]]) - phi_lock.get(g, float(data.qpos[q_j[g][0]])))
        if record:
            trace["t"].append(t), trace["phi"].append(float(data.qpos[q_j[gs[0]][0]])), trace["slip"].append(slip)
            trace["seatA"].append(float(Fc[0, hook[0][0], 1])), trace["kz"].append(float(data.qpos[qadr("kz")]))
            trace["seatB"].append(float(Fc[f_b, hook[f_b][0], 1])), trace["cx"].append(float(data.qpos[q_cx]))
        if t >= tl.t_power:
            if power_ref is None:
                power_ref = slip
            power_slip = max(power_slip, slip - power_ref)
            if slip > DROP:
                dropped = "power"
                break
            continue
        slip_pk = max(slip_pk, slip)
        tilt_pk = max(tilt_pk, abs(float(data.qpos[q_cth]) - ref_th))
        if slip > DROP:
            dropped = tl.phase(t)
            break
    for g in gs:
        row.update(wrench[g].readings(f"{g}_"))
    row.update(worst(row, "".join(gs)))
    row.update(close_state)
    row.update({"ok": not unstable and ref is not None, "reset_flag": bool(unstable)})
    if unstable:
        row.update(reason=f"simulation went unstable at {unstable_s:.4f} s; flagged, not scored", unstable_s=unstable_s,
                   passed=None, score=None, limiting=None)
        return row
    if ref is None:
        row.update(passed=False, score=0.0, limiting="grip", reason="never reached the lift")
        return row
    names = {0: "A", 3: "B"} if case == "t4b" else {0: "R", 1: "L"}
    seated = {}
    t_arr = np.asarray(seat_t)
    for f in hook:
        s = np.asarray(seat_series[f])
        nm = names[f]
        tail = s[t_arr >= tl.t_power - 0.2] if len(s) and len(t_arr) == len(s) else s[-10:]
        sus = s[(t_arr >= tl.t_final - 0.5) & (t_arr < tl.t_final)] if len(s) and len(t_arr) == len(s) else s
        seated[nm] = bool(len(tail) and float(np.mean(tail)) > SEATED_N and not dropped)
        row[f"{nm}_seat_N"] = float(np.mean(sus)) if len(sus) else math.nan
        row[f"{nm}_seat_peak_N"] = seat_pk[f]
        row[f"{nm}_seat_min_N"] = seat_min[f] if math.isfinite(seat_min[f]) else math.nan
        row[f"{nm}_seated"] = seated[nm]
        row[f"{nm}_seated_share"] = float(np.mean(s > SEATED_N)) if len(s) else 0.0
    for g in gs:
        row[f"{g}_cam_deg"] = math.degrees(cam[g])
    held = dropped in ("", "power")
    max_slip = sc.max_slip
    margins = {"slip": max_slip / max(slip_pk, 1e-9) if held else 0.0}
    all_seated = all(seated.values())
    score = min(margins.values()) if all_seated else 0.0
    limiting = "slip" if all_seated else "seat"
    passed = held and all_seated and score >= 1.0
    if not held:
        reason = f"dropped in the {dropped}"
    elif not all_seated:
        reason = "the lip comes off its seat on " + ", ".join(k for k, v in seated.items() if not v)
    else:
        reason = "" if passed else f"slip margin {score:.2f}"
    row.update({"passed": bool(passed), "score": float(min(score, 99.0)), "limiting": limiting, "reason": reason,
                "margins": margins, "dropped_in": dropped, "slip_mm": slip_pk * 1e3, "tilt_deg": math.degrees(tilt_pk),
                "power_cut_held": bool(dropped == "" and power_ref is not None and power_slip <= max_slip),
                "power_cut_slip_mm": power_slip * 1e3,
                "pass_rule": "held, each lip seated through the stop, slip within 2 mm"})
    if record:
        row["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    return row
