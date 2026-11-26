"""T3, tips between boxes (test conditions revision 2.21), in MuJoCo's x-z plane: the jaw axis along x, gravity -z.

Three Cheerios Family Size boxes stand face to face on the shelf, the target between two neighbours on the jaw axis,
`gap` apart. A neighbour slides and tips on the shelf until the next box, one more gap out, stops it. In the
first-box-out case one neighbour is a C-flute case wall that flexes 5 mm under 20 N. The jaw opens until each
finger's lowest `entry` mm clear the target's faces (planned for a centred target) and holds that opening on its servo
while it descends until the tips are `entry` below the box tops. It then closes on its set torque, squeezes, lifts
50 mm and holds 1 s, with no shake, and ends with the power cut on the brake.

A tip whose bottom doesn't fit between the target and its neighbour (a placement corner, a closed gap) lands on a box
top: that fails the entry without a MuJoCo run. The box edges are square; the row reports the least edge radius that
would cam each landed tip in, and whether a 0.5 mm radius would (a sensitivity, not scored). In MuJoCo a neighbour
meets each tip as one rigid hull of the tip's outline, through a softer, critically damped edge contact. The descent
stops where a neighbour's push passes 5 N (a 5 ms average), as a guarded move would, and the hold then runs from the
depth reached. A depth stop short of the 20 mm entry sensitivity is reported as one, not as an entry fail.

The target-side clearance is the least over the whole entry depth (SIM round 10), both for the capture range, the
span of offsets at which neither tip lands or meets the target's face, and for running the entry with the target's
contact off. The capture range is also given at the opening that centres the tip ends in the gaps.

Entry is scored on reaching the entry depth, on each neighbour's push at its top edge (3 mm, 5 N as a 5 ms average;
the case wall's force is reported, not capped) and on how far each is dragged up (2 mm). The 10 ms average is
recorded as a convergence check, the single-sample peak as the solver's, not scored, and a 5 ms push over 4 N is
flagged as sensitive to the edge contact's stiffness. The hold is scored on the target's slip on the jaw (2 mm), with
the box limits at the tips. The tips slide in at the drag friction (0.5, x1.2 in the robust run) and grip at the bare
friction (x0.8 in the robust run). No stop is run (revision 2.24: the power cut covers it).
"""

from __future__ import annotations

import collections
import math
from typing import Any

import mujoco
import numpy as np
import shapely.geometry

from .geometry import to_view
from .jaw import Jaw
from .mj1 import BAD_STATE, DROP, GRIP, OBJ, TABLE, _Builder, _fmt, _min_jerk, _p, element_springs, spring_loads
from .objects import G, GraspObject
from .scenario import Scenario
from .screen1 import Limits, _band_min, _geometry, _view
from .sensor import Wrench, layer_depth
from .sensor import config as sensor_config

# a neighbour or the case wall; the next box out, which stops a neighbour; the fingers' rigid tip hulls; the target
# while the tips enter, when it touches only the shelf (only while both fingers' lowest `entry` mm clear its faces,
# and a hull bridges the concave corners on a tip's inner side). The neighbours, the wall and the hulls touch only
# through explicit pairs.
NB, NEXT, OUTLINE, OBJ_ENTRY = 8, 16, 32, 64
TAGS = {1: "R", -1: "L"}
LAND_MM = 0.5  # the bottom of a tip that must fit the gap
LAND_TOL_MM = 0.05
HULL_ABOVE_MM = 2.0  # the tip hull's top above the box tops at full entry
STEP_MM = 2.0  # an inward step on a finger's inner side this big within 0.25 mm of height is a shoulder


def _band_max(polys: list[np.ndarray], z_lo: float, z_hi: float) -> float:
    return -_band_min([np.column_stack([-P[:, 0], P[:, 1]]) for P in polys], z_lo, z_hi)[0]


def _finger(jaw: Jaw, side: int, phi: float) -> list[np.ndarray]:
    grip, body = _geometry(jaw.sides[side])
    return _view(grip + body, phi, jaw.pivot_x)


def _low(polys: list[np.ndarray]) -> float:
    return min(float(P[:, 1].min()) for P in polys)


def entry_opening(jaw: Jaw, side: int, x_face: float, entry: float) -> float:
    """The smallest opening (rad) at which one finger's lowest `entry` mm clear a face at x_face (mm, in the
    finger's own view), or nan if the stroke never gets there."""

    def clear(phi: float) -> bool:
        v = _finger(jaw, side, phi)
        z = _low(v)
        return _band_min(v, z, z + entry)[0] >= x_face

    lo, hi = jaw.phi_meet, jaw.phi_max
    if not clear(hi):
        return math.nan
    if clear(lo):
        return lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if clear(mid) else (mid, hi)
    return hi


def _shoulder(jaw: Jaw, side: int, phi: float, top: float) -> float:
    """The height (mm above the tip's lowest point, below `top`) of the first shoulder on a finger's inner side at
    opening phi, or nan."""
    v = _finger(jaw, side, phi)
    z = _low(v)
    least = math.inf
    for h in np.arange(0.0, top, 0.25):
        x = _band_min(v, z + h, z + h + 0.25)[0]
        if not math.isfinite(x):
            continue
        if math.isfinite(least) and x < least - STEP_MM:
            return float(h)
        least = min(least, x)
    return math.nan


def _land(jaw: Jaw, side: int, phi: float) -> tuple[float, float]:
    """The inner and outer edges (mm, the finger's own view) of a tip's lowest LAND_MM at opening phi."""
    v = _finger(jaw, side, phi)
    z = _low(v)
    return _band_min(v, z, z + LAND_MM)[0], _band_max(v, z, z + LAND_MM)


def centring_opening(jaw: Jaw, half: float, gap: float) -> float:
    """The opening (rad) that puts the tip ends, on the mean of the two fingers, in the middle of the gaps between the
    target's faces at `half` and the neighbours', or nan if the stroke doesn't reach it."""
    mid = half + gap / 2

    def off(phi: float) -> float:
        return sum(0.5 * sum(_land(jaw, s, phi)) - mid for s in (1, -1))

    lo, hi = jaw.phi_meet, jaw.phi_max
    if off(lo) > 0.0 or off(hi) < 0.0:
        return math.nan
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if off(m) < 0.0 else (lo, m)
    return 0.5 * (lo + hi)


def _tips(jaw: Jaw, phi: float, half: float, gap: float, entry: float, lateral: float) -> tuple[dict[str, Any], tuple, tuple]:
    """Each tip's sweep, push and landing numbers at opening phi with the boxes `lateral` mm along +x, and the capture
    range at that opening: by the target-side clearance over the whole entry depth, and by the tip end's alone."""
    views = {s: _finger(jaw, s, phi) for s in (1, -1)}
    z_low = {s: _low(v) for s, v in views.items()}
    z_top = min(z_low.values()) + entry
    out: dict[str, Any] = {"z_low_mm": min(z_low.values()), "z_top_mm": z_top}
    cap, cap_land = [-math.inf, math.inf], [-math.inf, math.inf]
    for s, tag in TAGS.items():
        face = half + s * lateral
        if z_low[s] >= z_top:
            x_in = x_out = math.nan
        else:
            x_in = _band_min(views[s], z_low[s], z_top)[0]
            x_out = _band_max(views[s], z_low[s], z_top)
        b_in = _band_min(views[s], z_low[s], z_low[s] + LAND_MM)[0]
        b_out = _band_max(views[s], z_low[s], z_low[s] + LAND_MM)
        out.update({f"sweep_{tag}_mm": x_out - x_in, f"clear_{tag}_mm": x_in - face,
                    f"push_need_{tag}_mm": max(0.0, x_out - (face + gap)) if math.isfinite(x_out) else 0.0,
                    f"tip_width_{tag}_mm": b_out - b_in,
                    f"land_target_{tag}_mm": face - b_in, f"land_neighbour_{tag}_mm": b_out - (face + gap)})
        for c, inner in ((cap, min(x_in, b_in) if math.isfinite(x_in) else b_in), (cap_land, b_in)):
            lo, hi = b_out - half - gap, inner - half
            c[:] = [max(c[0], lo), min(c[1], hi)] if s > 0 else [max(c[0], -hi), min(c[1], -lo)]
    return out, tuple(cap), tuple(cap_land)


def entry_check(jaw: Jaw, box_mm: float, gap: float, entry: float, lateral: float = 0.0) -> dict[str, Any]:
    """The geometric entry: the opening both fingers share (planned for a centred target), how far each finger's band in
    the gap reaches out past the target's face (its sweep) and the push that leaves on the neighbour. With the boxes
    `lateral` mm along +x, each tip's lowest LAND_MM must still fit between the target's face and the neighbour's, or it
    lands on a box top. `clear_*_mm` is a finger's least clearance to the target's face over the whole entry depth. The
    capture range is the span of offsets at which neither tip lands on the neighbour's top nor meets the target's face
    over the entry depth (`capture_land_*` takes the tip end's clearance alone), at the entry opening and at the opening
    that centres the tip ends in the gaps. A shoulder within the entry depth is a depth stop over the target's top:
    `depth_stop_*_mm` is how deep it lets the tips go, read at the opening that clears the tip below it."""
    half = box_mm / 2
    phis = {s: entry_opening(jaw, s, half, entry) for s in (1, -1)}
    if not all(math.isfinite(p) for p in phis.values()):
        return {"ok": False, "reason": "the open fingers' tips never clear the target's faces"}
    phi = max(phis.values())
    stops = {}
    for s, tag in TAGS.items():
        h = _shoulder(jaw, s, phi, entry)
        if math.isfinite(h):
            p = entry_opening(jaw, s, half, max(h - 0.25, LAND_MM))
            h2 = _shoulder(jaw, s, p, entry + 5.0) if math.isfinite(p) else math.nan
            h = h2 if math.isfinite(h2) else h
        stops[f"depth_stop_{tag}_mm"] = h
    tips, cap, cap_land = _tips(jaw, phi, half, gap, entry, lateral)
    out: dict[str, Any] = {"ok": True, "phi": phi, "entry_opening_deg": math.degrees(phi - jaw.phi_meet), **tips, **stops,
                           "capture_lo_mm": cap[0], "capture_hi_mm": cap[1],
                           "capture_land_lo_mm": cap_land[0], "capture_land_hi_mm": cap_land[1]}
    phi_c = centring_opening(jaw, half, gap)
    out["centred_opening_deg"] = math.degrees(phi_c - jaw.phi_meet)
    if math.isfinite(phi_c):
        tc, cap_c, cap_land_c = _tips(jaw, phi_c, half, gap, entry, 0.0)
        out.update({"centred_capture_lo_mm": cap_c[0], "centred_capture_hi_mm": cap_c[1],
                    "centred_capture_land_lo_mm": cap_land_c[0], "centred_capture_land_hi_mm": cap_land_c[1]})
        out.update({f"centred_{k}": tc[k] for tag in TAGS.values()
                    for k in (f"clear_{tag}_mm", f"land_target_{tag}_mm", f"land_neighbour_{tag}_mm", f"push_need_{tag}_mm")})
    return out


def _tip_hull(jaw: Jaw, side: int, phi: float, z_cut: float):
    """The convex hull (finger-local, shapely) of the finger's outline below view height z_cut at opening phi."""
    s, c = math.sin(phi), math.cos(phi)
    n, t = np.array([c, s]), np.array([-s, c])
    p0, L = -z_cut * n, 1e3
    below = shapely.geometry.Polygon([p0 + L * t, p0 - L * t, p0 - L * t + L * n, p0 + L * t + L * n])
    tip = jaw.sides[side].outline.intersection(below)
    return None if tip.is_empty else tip.convex_hull


def _outline_xml(jaw: Jaw, side: int, tag: str, mu: float, assets: list[str], hull) -> tuple[str, list[str]]:
    """The tip hull as one rigid mesh that only the neighbours collide with. A box's corner sliding up the tip catches
    on the ends of sprung elements, and on the seams between convex pieces."""
    if hull is None:
        return "", []
    js = jaw.sides[side]
    w = js.width_at(float(hull.centroid.x))
    verts = []
    for u, v in np.asarray(hull.exterior.coords)[:-1]:
        X, Z = _p(u, v, side)
        verts += [X, -w * 0.5e-3, Z, X, w * 0.5e-3, Z]
    mesh, gname = f"{tag}o", f"{tag}_outline"
    assets.append(f'<mesh name="{mesh}" vertex="{_fmt(verts)}"/>')
    return (f'<geom name="{gname}" type="mesh" mesh="{mesh}" mass="0" friction="{_fmt(mu)} 0.005 0.0001" '
            f'priority="3" contype="{OUTLINE}" conaffinity="0" rgba="0.3 0.7 0.9 0.35"/>'), [gname]


def simulate_t3(jaw: Jaw, box: GraspObject, sc: Scenario, b1: dict, gap: float | None = None, entry: float | None = None,
                lateral: float = 0.0, friction_scale: float = 1.0, drag_scale: float = 1.0, wall_side: int = 0,
                tau: float | None = None, brake: float = 30.0, power_s: float | None = None,
                land_check: bool = True, record: bool = False) -> dict[str, Any]:
    """One T3 run. `gap` and `entry` (mm) default to the test conditions'; `lateral` (mm) moves the boxes off the jaw
    centreline; `wall_side` (+1 or -1) puts the case wall in place of that neighbour. The run ends with the power cut:
    drive torque to zero with the brake on (`brake`, N m per gripper), held `power_s` at the lifted height."""
    cfg = b1["t3"]
    power_s = float(b1.get("power_cut", {}).get("hold_s", 5.0)) if power_s is None else float(power_s)
    gap = float(cfg["gap_mm"] if gap is None else gap)
    entry = float(cfg["entry_mm"] if entry is None else entry)
    clear = float(cfg.get("clear_mm", 3.0))
    lim = Limits.from_config(b1, "T3", friction_scale=friction_scale)
    tau = jaw.torque if tau is None else tau
    T, H, Wd = box.thickness, box.height, box.width
    row: dict[str, Any] = {"gap_mm": gap, "entry_mm": entry, "lateral_mm": lateral, "wall_side": wall_side,
                           "friction_scale": friction_scale, "drag_scale": drag_scale, "finger_torque_Nm": tau,
                           "brake_Nm": brake, "power_s": power_s}
    geo = entry_check(jaw, T * 1e3, gap, entry, lateral)
    row.update({k: v for k, v in geo.items() if k != "phi"})
    if not geo["ok"]:
        return {**row, "passed": False, "score": 0.0, "limiting": "entry"}
    stop = min(geo["depth_stop_R_mm"], geo["depth_stop_L_mm"], key=lambda h: h if math.isfinite(h) else math.inf)
    if math.isfinite(stop) and stop < entry - 0.05:
        if entry > float(cfg["entry_mm"]) + 0.05:
            return {**row, "passed": None, "score": None, "limiting": "depth stop", "landed": False, "depth_stop_mm": stop,
                    "reason": f"depth stop at {stop:.1f} mm: a shoulder over the target's top ends the entry there, "
                              f"short of the {entry:g} mm sensitivity; geometric, no MuJoCo run"}
        return {**row, "passed": False, "score": stop / entry, "limiting": "entry", "landed": False, "depth_stop_mm": stop,
                "reason": f"a shoulder {stop:.1f} mm up the tips is a depth stop over the target's top: they enter "
                          f"{stop:.1f} of {entry:g} mm; geometric, no MuJoCo run"}
    # The model's carton edges are square. Over a top edge rounded to r, a tip's corner `over` mm onto the top meets
    # the round at sin(a) = (r - over)/r from vertical, and slides off it instead of sticking once tan(a) > mu.
    mu_d = float(cfg.get("drag_friction", 0.5)) * drag_scale
    cam = 1.0 - mu_d / math.hypot(1.0, mu_d)
    r_sens = float(cfg.get("edge_radius_sensitivity_mm", 0.5))
    landed = []
    for tag in TAGS.values():
        over = {key: geo[f"land_{key}_{tag}_mm"] for key in ("target", "neighbour")}
        hits = {key: o for key, o in over.items() if o > LAND_TOL_MM}
        row[f"cam_radius_{tag}_mm"] = max(hits.values()) / cam if hits else math.nan
        landed += [f"the {tag} tip lands on the "
                   f"{'target' if key == 'target' else 'case wall' if TAGS.get(wall_side) == tag else 'neighbour'}'s top, "
                   f"{o:.2f} mm over (a top edge rounded to {o / cam:.2f} mm would cam it in)" for key, o in hits.items()]
    row["landed"] = bool(landed)
    radii = [row[f"cam_radius_{tag}_mm"] for tag in TAGS.values() if math.isfinite(row[f"cam_radius_{tag}_mm"])]
    row["cam_radius_mm"] = max(radii) if radii else math.nan
    row["edge_radius_sensitivity_mm"] = r_sens
    row["enters_at_sensitivity_radius"] = bool(not radii or max(radii) <= r_sens)
    if landed and land_check:
        return {**row, "passed": False, "score": 0.0, "limiting": "entry",
                "reason": "; ".join(landed) + f"; the least top-edge radius that cams it in is {max(radii):.2f} mm; "
                          "geometric, no MuJoCo run"}
    target_off = all(geo[f"clear_{tag}_mm"] >= -LAND_TOL_MM for tag in TAGS.values())
    row["target_contact_in_entry"] = not target_off
    phi_e = geo["phi"]
    z_top_view = geo["z_top_mm"]
    b = _Builder(jaw, box, sc, b1, entry, {1: phi_e, -1: phi_e}, lim.friction_scale, lim.band_factor, z_top_view)
    mu_drag = float(cfg.get("drag_friction", 0.5)) * drag_scale
    L = jaw.reach
    fingers, outlines = [], []
    for side, nm in ((1, "R"), (-1, "L")):
        xml, m_f, mom = b.finger_xml(side)
        hull = _tip_hull(jaw, side, phi_e, z_top_view + HULL_ABOVE_MM)
        if hull is not None:
            hv = to_view(np.asarray(hull.exterior.coords)[:-1], phi_e, jaw.pivot_x)
            z_lo = float(hv[:, 1].min())
            row[f"hull_excess_{nm}_mm"] = _band_max([hv], z_lo, z_top_view) - _band_max(_finger(jaw, side, phi_e), z_lo, z_top_view)
        o_xml, o_names = _outline_xml(jaw, side, nm, mu_drag, b.assets, hull)
        xml += o_xml
        outlines += o_names
        for gname in o_names:
            cu, cv = float(hull.centroid.x), float(hull.centroid.y)
            b.geom_depth[gname] = layer_depth(b1, jaw.sides[side], cu, cv, jaw.sides[side].width_at(cu))
        m_f, I_f = max(m_f, 1e-3), max(mom, 1e-6)
        fingers.append(
            f'<body name="finger{nm}" pos="{_fmt((side * jaw.pivot_x * 1e-3, 0, 0))}" gravcomp="1">'
            f'<joint name="j{nm}" type="hinge" axis="0 {-side} 0" range="{_fmt(jaw.phi_meet)} {_fmt(jaw.phi_max + 1e-3)}" '
            f'armature="{_fmt(float(b1.get("closing", {}).get("armature_kgm2", 0.015)))}" damping="0" '
            f'solreffriction="{_fmt(2 * b.dt)} 1" solimpfriction="0.999 0.9999 0.001 0.5 2"/>'
            f'<inertial pos="0 0 {_fmt(-L * 0.5e-3)}" mass="{_fmt(m_f)}" diaginertia="{_fmt((I_f, I_f, I_f))}"/>'
            f"{xml}</body>")
    if any(e.seg <= 2.0 + 1e-9 for e in b.elements):
        b.substeps = max(b.substeps, int(round(b.dt / b.fine_dt)))
    tc = b.contact_tc
    k_servo = float(sc.sim.get("servo_kp", 2.0e6))
    kv_servo = 2 * math.sqrt(k_servo * (1.6 + box.mass))
    xl = lateral * 1e-3
    gp, blocked = gap * 1e-3, float(cfg.get("blocked_after_mm", 1.3)) * 1e-3
    I = max(box.inertia, 1e-7)
    com_z = H / 2 - box.com_below_top

    def carton(name: str, x: float, rgba: str, contype: int, conaff: int) -> str:
        return (f'<body name="{name}" pos="{_fmt((x, 0, H / 2))}">'
                f'<joint name="{name}x" type="slide" axis="1 0 0"/><joint name="{name}z" type="slide" axis="0 0 1"/>'
                f'<joint name="{name}th" type="hinge" axis="0 1 0"/>'
                f'<inertial pos="0 0 {_fmt(com_z)}" mass="{_fmt(box.mass)}" diaginertia="{_fmt((I, I, I))}"/>'
                f'<geom name="{name}" type="box" size="{_fmt((T / 2, Wd / 2, H / 2))}" mass="0" '
                f'friction="{_fmt(box.material.friction)} 0.005 0.0001" contype="{contype}" conaffinity="{conaff}" rgba="{rgba}"/>'
                "</body>")

    world = [f'<geom name="table" type="box" pos="0 0 -0.01" size="0.8 0.4 0.01" friction="0.4 0.005 0.0001" '
             f'priority="1" contype="{TABLE}" conaffinity="{GRIP | OBJ}" rgba="0.8 0.8 0.8 1"/>',
             carton("obj", xl, "0.9 0.75 0.35 1", OBJ_ENTRY if target_off else OBJ, TABLE if target_off else GRIP | TABLE)]
    pr = f'condim="3" solref="{_fmt(tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"'
    pr_edge = f'condim="3" solref="{_fmt(float(cfg.get("edge_contact_tc_s", 0.002)))} 1" solimp="0.95 0.99 0.0005 0.5 2"'
    mu_c, mu_s = float(cfg.get("carton_friction", 0.4)), float(cfg.get("shelf_friction", 0.35))
    pairs = [f'<pair geom1="{h}" geom2="n{h[0]}" friction="{_fmt((mu_drag, mu_drag, 0.005, 1e-4, 1e-4))}" {pr_edge}/>'
             for h in outlines]
    cw = cfg.get("case_wall", {})
    k_wall = float(cw.get("at_N", 20.0)) / (float(cw.get("flex_mm", 5.0)) * 1e-3)
    for s, tag in TAGS.items():
        if s == wall_side:
            wt = float(cw.get("thickness_mm", 4.0)) * 1e-3
            mw = float(cw.get("mass_kg", 0.05))
            world.append(f'<body name="wall{tag}" pos="{_fmt((xl + s * (T / 2 + gp + wt / 2), 0, H / 2))}">'
                         f'<joint name="wall{tag}x" type="slide" axis="1 0 0" stiffness="{_fmt(k_wall)}" '
                         f'damping="{_fmt(2 * 0.3 * math.sqrt(k_wall * mw))}"/>'
                         f'<inertial pos="0 0 0" mass="{_fmt(mw)}" diaginertia="1e-4 1e-4 1e-4"/>'
                         f'<geom name="n{tag}" type="box" size="{_fmt((wt / 2, Wd / 2, H / 2))}" mass="0" contype="{NB}" '
                         f'conaffinity="0" rgba="0.65 0.5 0.3 1"/></body>')
            pairs.append(f'<pair geom1="obj" geom2="n{tag}" friction="{_fmt((mu_c, mu_c, 0.005, 1e-4, 1e-4))}" {pr}/>')
            continue
        world.append(carton(f"n{tag}", xl + s * (T + gp), "0.85 0.7 0.4 1", NB, 0))
        world.append(f'<geom name="next{tag}" type="box" pos="{_fmt((xl + s * (2 * T + gp + blocked), 0, H / 2))}" '
                     f'size="{_fmt((T / 2, Wd / 2, H / 2))}" contype="{NEXT}" conaffinity="0" rgba="0.7 0.7 0.7 0.4"/>')
        pairs += [f'<pair geom1="obj" geom2="n{tag}" friction="{_fmt((mu_c, mu_c, 0.005, 1e-4, 1e-4))}" {pr}/>',
                  f'<pair geom1="n{tag}" geom2="table" friction="{_fmt((mu_s, mu_s, 0.005, 1e-4, 1e-4))}" {pr}/>',
                  f'<pair geom1="n{tag}" geom2="next{tag}" friction="{_fmt((mu_c, mu_c, 0.005, 1e-4, 1e-4))}" {pr}/>']
    D = (entry + clear) * 1e-3
    z_piv0 = H - z_top_view * 1e-3 + D
    xml = f"""<mujoco model="grasp2d_t3">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{_fmt(b.dt / b.substeps)}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic" impratio="10" iterations="60" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default><geom condim="3" solref="{_fmt(tc)} 1" solimp="0.95 0.99 0.0005 0.5 2"/></default>
  <asset>{''.join(b.assets)}</asset>
  <worldbody>
    {''.join(world)}
    <body name="base" pos="0 0 {_fmt(z_piv0)}" gravcomp="1">
      <joint name="base_x" type="slide" axis="1 0 0"/><joint name="base_z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="1e-3 1e-3 1e-3"/>
      {''.join(fingers)}
    </body>
  </worldbody>
  <contact>{''.join(pairs)}</contact>
  <equality>
    <joint name="gear" joint1="jL" joint2="jR" polycoef="0 1 0 0 0" solref="{_fmt(4 * b.dt)} 1"/>
  </equality>
  <actuator>
    <position name="servo_x" joint="base_x" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <position name="servo_z" joint="base_z" kp="{_fmt(k_servo)}" kv="{_fmt(kv_servo)}"/>
    <motor name="driveR" joint="jR" gear="1"/>
    <motor name="driveL" joint="jL" gear="1"/>
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    data = mujoco.MjData(model)
    row["outline_pieces"] = len(outlines)
    return _run(model, data, jaw, box, sc, b1, cfg, lim, b, row, phi_e, D, clear * 1e-3, tau, wall_side, k_wall,
                mu_drag, outlines, brake, power_s, record)


def _run(model, data, jaw: Jaw, box: GraspObject, sc: Scenario, b1: dict, cfg: dict, lim: Limits, b: _Builder,
         row: dict[str, Any], phi_e: float, D: float, clear: float, tau: float, wall_side: int, k_wall: float,
         mu_drag: float, outlines: list[str], brake: float, power_s: float, record: bool) -> dict[str, Any]:
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = lambda n: model.jnt_qposadr[jid(n)]
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    a_jR, d_jR, d_jL = qadr("jR"), model.jnt_dofadr[jid("jR")], model.jnt_dofadr[jid("jL")]
    data.qpos[a_jR] = phi_e
    data.qpos[qadr("jL")] = phi_e
    mujoco.mj_forward(model, data)
    T, H = box.thickness, box.height
    els = b.elements
    n_el = len(els)
    el_geom = np.array([gid(e.geom) for e in els], dtype=int)
    el_of_geom = np.full(model.ngeom, -1, dtype=int)
    el_of_geom[el_geom] = np.arange(n_el)
    el_side = np.array([e.side for e in els])
    el_grip = np.array([e.kind != "bare" for e in els])
    el_area = np.array([e.area for e in els])
    el_ext = np.array([e.ext for e in els]) * 1e-3
    el_sp = element_springs(model, els)
    finger_side = np.zeros(model.ngeom, dtype=int)
    for gname, s in b.body_geoms.items():
        finger_side[gid(gname)] = s
    finger_side[el_geom] = el_side
    body_like = np.zeros(model.ngeom, dtype=bool)
    body_like[[gid(g) for g in b.body_geoms]] = True
    body_like[el_geom[~el_grip]] = True
    g_obj = gid("obj")
    g_nb = {s: gid(f"n{tag}") for s, tag in TAGS.items()}
    nb_body = {s: bid(f"n{tag}") for s, tag in TAGS.items() if s != wall_side}
    wall_q = qadr(f"wall{TAGS[wall_side]}x") if wall_side else -1
    obj_body = bid("obj")
    a_oth, a_bx, a_bz = qadr("objth"), qadr("base_x"), qadr("base_z")
    is_outline = np.zeros(model.ngeom, dtype=bool)
    is_outline[[gid(g) for g in outlines]] = True
    stall_n = float(cfg.get("push_limit_N", 5.0))

    cl = b1.get("closing", {})
    w_fast, w_slow = float(cl.get("fast_rad_s", 1.0)), float(cl.get("slow_rad_s", 0.3))
    v_desc = float(cfg.get("descend_m_s", 0.05))
    t_desc = D / v_desc
    t_close0 = t_desc + float(cfg.get("entry_settle_s", 0.1))
    t_close = t_close0 + max(sc.squeeze_time, 0.15)
    t_settle = t_close + sc.settle_time
    lift = float(cfg.get("lift_mm", 50.0)) * 1e-3
    t_lift = t_settle + sc.lift_time
    t_cut = t_lift + float(cfg.get("hold_s", 1.0))
    t_end = t_cut + power_s
    kp_j = float(cfg.get("jaw_servo_Nm_rad", 2.0e4))
    kv_j = 2 * math.sqrt(kp_j * float(cl.get("armature_kgm2", 0.015)))
    u_peak = float(b1.get("drive_Nm_per_gripper", {}).get("peak", 25.0)) / 2
    kv_drive = tau / 0.2
    # SIM round 19: the close is a speed servo that drives and brakes within its torque, as a geared drive's is;
    # `closing: {brakes: false}` keeps the push-only governor the rows before block C ran.
    brake_floor = -tau if cl.get("brakes", True) else 0.0
    kp, kv = float(model.actuator_gainprm[0, 0]), float(-model.actuator_biasprm[0, 2])
    dt = model.opt.timestep
    # Every physics step is read, for the descent's stall and the first touch and for every 5 ms average (SIM round
    # 18): the model's first impacts last about 0.25 ms. The rated single-sample peaks (the band, the panel, the bed and
    # the body) keep one step in `every`, as they were rated; their every-step maxima are kept beside them.
    every = max(1, int(round(float(cfg.get("peak_sample_s", 4 * b.substeps * dt)) / dt)))
    # Rev 2.34: the wrist sensor over the whole jaw. T3's whole window runs from the descent, since the entry is its
    # load case, to the power cut; the final hold's mean is over the hold's last 1 s before the cut.
    wrench = Wrench(model, b1, "base", b.geom_depth, dt)
    hold_s = float(sensor_config(b1).get("hold_s", 1.0))

    def phase_at(t: float) -> str:
        for name, end in (("descend", t_desc), ("entry", t_close0), ("close", t_close), ("settle", t_settle),
                          ("lift", t_lift), ("hold", t_cut)):
            if t < end:
                return name
        return "cut"

    box_H = H
    fill_rel = -box.headspace
    band = lim.band_mm * 1e-3

    def corner(s: int) -> np.ndarray:
        R = data.xmat[nb_body[s]].reshape(3, 3)
        return data.xpos[nb_body[s]] + R @ np.array([-s * T / 2, 0.0, H / 2])

    c0 = {s: corner(s).copy() for s in nb_body}
    z0 = {s: float(data.xpos[nb_body[s]][2]) for s in nb_body}

    def measure():
        n = data.ncon
        Fc = np.zeros(n_el)
        push_f = {1: 0.0, -1: 0.0}
        body = {1: 0.0, -1: 0.0}
        spring = spring_loads(data, el_sp, el_grip)
        if n == 0:
            return np.where(np.isnan(spring), 0.0, spring), push_f, body
        con = data.contact
        g1, g2 = con.geom1[:n], con.geom2[:n]
        adr = con.efc_address[:n]
        ok = adr >= 0
        fn = np.where(ok, data.efc_force[np.maximum(adr, 0)], 0.0)
        for ga, gb in ((g1, g2), (g2, g1)):
            on_obj = ok & (gb == g_obj) & (finger_side[ga] != 0)
            e = el_of_geom[ga]
            m = on_obj & (e >= 0)
            np.add.at(Fc, e[m], fn[m])
            for s in (1, -1):
                body[s] += float(fn[on_obj & body_like[ga] & (finger_side[ga] == s)].sum())
                push_f[s] += float(fn[ok & (gb == g_nb[s]) & is_outline[ga]].sum())
        F = np.where(np.isnan(spring), Fc, spring)
        return F, push_f, body

    def zones(F: np.ndarray) -> dict[int, tuple[float, float, float]]:
        pos = data.geom_xpos[el_geom]
        R = data.xmat[obj_body].reshape(3, 3)
        z_rel = ((pos - data.xpos[obj_body]) @ R)[:, 2] - box_H / 2
        lo, hi = z_rel - el_ext / 2, z_rel + el_ext / 2
        span = np.maximum(hi - lo, 1e-6)
        a_band = np.clip((hi - (-band)) / span, 0.0, 1.0)
        a_fill = np.clip((hi - fill_rel) / span, 0.0, 1.0)
        p = F / el_area
        out = {}
        for sd in (1, -1):
            m = (el_side == sd) & el_grip
            fb = float(np.sum(F[m] * a_band[m]))
            fp = float(np.sum(F[m] * np.maximum(a_fill[m] - a_band[m], 0.0)))
            bed = m & (a_fill < 0.5)
            out[sd] = (fb, fp, float(p[bed].max()) if bed.any() else 0.0)
        return out

    pk = {k: 0.0 for k in ("band", "panel", "bed", "body", "i_band", "i_panel", "i_bed", "i_body", "slip", "rot")}
    pk_step = {k: 0.0 for k in ("band", "panel", "bed", "body", "i_band", "i_panel", "i_bed", "i_body")}
    push = {s: 0.0 for s in (1, -1)}
    push_entry = {s: 0.0 for s in (1, -1)}
    push_n = {s: 0.0 for s in (1, -1)}
    push_raw = {s: 0.0 for s in (1, -1)}
    n_avg = max(1, int(round(float(cfg.get("push_average_s", 0.005)) / dt)))
    n_avg10 = max(1, int(round(float(cfg.get("push_check_average_s", 0.010)) / dt)))
    pf_buf = {s: collections.deque(maxlen=n_avg) for s in (1, -1)}
    pf_buf10 = {s: collections.deque(maxlen=n_avg10) for s in (1, -1)}
    push_n10 = {s: 0.0 for s in (1, -1)}
    push_n_entry = {s: 0.0 for s in (1, -1)}
    drag = {s: 0.0 for s in (1, -1)}
    wall_mm = 0.0
    ref = ref_th = p_local = None
    grip_n = 0.0
    dropped = ""
    unstable, unstable_s = False, math.nan
    first_touch = math.nan
    trace: dict[str, list[float]] = {k: [] for k in ("t", "phi", "z", "push_R", "push_L", "F_R", "F_L", "slip")} if record else {}
    gripping = False
    z_stop, stall_s, z_min = None, math.nan, 0.0
    brake_on = power_ref = None
    power_slip, brake_net = 0.0, []
    i = -1
    while data.time < t_end:
        i += 1
        t = data.time
        phi = float(data.qpos[a_jR])
        om = float(data.qvel[d_jR])
        if t < t_close0:
            u = min(max(kp_j * (phi_e - phi) - kv_j * om, -u_peak), u_peak)
        else:
            if not gripping:
                # the body-level filter is compiled from its geoms' bits and must follow them
                model.geom_contype[g_obj] = model.body_contype[obj_body] = OBJ
                model.geom_conaffinity[g_obj] = model.body_conaffinity[obj_body] = GRIP | TABLE
                gripping = True
            if t < t_close:
                target = w_fast if phi > phi_e + math.radians(float(cl.get("slow_deg", 5.0))) else w_slow
                u = -min(max(kv_drive * (target + om), brake_floor), tau)
            elif t < t_cut:
                u = -tau
            else:
                u = 0.0
                if brake_on is None:
                    model.dof_frictionloss[d_jR] = model.dof_frictionloss[d_jL] = brake / 2.0
                    brake_on = t
        data.ctrl[2] = u
        data.ctrl[3] = u
        z_bot = -D if z_stop is None else z_stop
        if t < t_desc and z_stop is None:
            p, dp = _min_jerk(t / t_desc)
            z, vz = -D * p, -D * dp / t_desc
        elif t < t_settle:
            z, vz = z_bot, 0.0
        else:
            p, dp = _min_jerk((t - t_settle) / sc.lift_time)
            z = z_bot + lift * p
            vz = lift * dp / sc.lift_time if t < t_lift else 0.0
        data.ctrl[0] = 0.0
        data.ctrl[1] = z + kv / kp * vz
        t_step = data.time
        mujoco.mj_step(model, data)
        if (data.time < t_step or any(data.warning[w].number for w in BAD_STATE)
                or not np.isfinite(data.qacc).all() or abs(data.qpos[a_jR]) > 3.0):
            unstable, unstable_s = True, t_step
            break
        keep = i % every == 0
        t = data.time
        F, pf, body = measure()
        wrench.sample(model, data, t, phase_at(t), t < t_cut, t_cut - hold_s <= t < t_cut)
        zn = zones(F)
        for s in nb_body:
            c = corner(s)
            push[s] = max(push[s], s * (c[0] - c0[s][0]) * 1e3)
            drag[s] = max(drag[s], (float(data.xpos[nb_body[s]][2]) - z0[s]) * 1e3)
        pf_avg = {}
        for s in (1, -1):
            pf_buf[s].append(pf[s])
            pf_buf10[s].append(pf[s])
            pf_avg[s] = sum(pf_buf[s]) / len(pf_buf[s])
            push_n[s] = max(push_n[s], pf_avg[s])
            push_n10[s] = max(push_n10[s], sum(pf_buf10[s]) / len(pf_buf10[s]))
            push_raw[s] = max(push_raw[s], pf[s])
        if t < t_close0:
            z_min = min(z_min, float(data.qpos[a_bz]))
            if z_stop is None and t < t_desc and any(pf_avg[s] > stall_n for s in nb_body):
                z_stop, stall_s = float(data.qpos[a_bz]), t
        if wall_side:
            wall_mm = max(wall_mm, wall_side * float(data.qpos[wall_q]) * 1e3)
        if not math.isfinite(first_touch) and F.sum() + body[1] + body[-1] > 1e-3:
            first_touch = float(data.qpos[a_jR])
        if record and keep:
            trace["t"].append(t), trace["phi"].append(float(data.qpos[a_jR])), trace["z"].append(float(data.qpos[a_bz]))
            trace["push_R"].append(push[1]), trace["push_L"].append(push[-1]), trace["F_R"].append(pf[1])
            trace["F_L"].append(pf[-1])
        if t < t_settle:
            for s in (1, -1):
                push_entry[s], push_n_entry[s] = push[s], push_n[s]
            if t >= t_close0:
                for p in (pk_step, pk) if keep else (pk_step,):
                    for sd in (1, -1):
                        p["i_band"] = max(p["i_band"], zn[sd][0])
                        p["i_panel"] = max(p["i_panel"], zn[sd][1])
                        p["i_bed"] = max(p["i_bed"], zn[sd][2])
                        p["i_body"] = max(p["i_body"], body[sd])
            continue
        if ref is None:
            loaded = F > 1e-3
            zg = float(np.average(data.geom_xpos[el_geom[loaded], 2], weights=F[loaded])) if loaded.any() else float(data.xpos[obj_body][2])
            R = data.xmat[obj_body].reshape(3, 3)
            p_local = R.T @ (np.array([float(data.xpos[obj_body][0]), 0.0, zg]) - data.xpos[obj_body])
            ref_th = float(data.qpos[a_oth])
            grip_n = float(F[(el_side == 1) & el_grip].sum())
        R = data.xmat[obj_body].reshape(3, 3)
        pw = data.xpos[obj_body] + R @ p_local
        rel = np.array([pw[0] - data.qpos[a_bx], pw[2] - data.qpos[a_bz]])
        if ref is None:
            ref = rel.copy()
        slip = float(np.hypot(*(rel - ref)))
        if record and keep:
            trace["slip"].append(slip)
        if t >= t_cut:
            if power_ref is None:
                power_ref = slip
            ne = data.nefc
            if ne:
                fr = data.efc_type[:ne] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
                f, ids = data.efc_force[:ne], data.efc_id[:ne]
                # net over both motors: the gear couples the fingers, so their two frictions can fight each other
                brake_net.append(abs(float(f[fr & (ids == d_jR)].sum() + f[fr & (ids == d_jL)].sum())))
            power_slip = max(power_slip, slip - power_ref)
            if slip > DROP:
                dropped = "power"
                break
            continue
        pk["slip"] = max(pk["slip"], slip)
        pk["rot"] = max(pk["rot"], abs(float(data.qpos[a_oth]) - ref_th))
        for p in (pk_step, pk) if keep else (pk_step,):
            for sd in (1, -1):
                p["band"] = max(p["band"], zn[sd][0])
                p["panel"] = max(p["panel"], zn[sd][1])
                p["bed"] = max(p["bed"], zn[sd][2])
                p["body"] = max(p["body"], body[sd])
        if slip > DROP:
            dropped = "lift" if t < t_lift else "hold"
            break

    row.update(wrench.readings())
    row.update({"ok": not unstable and ref is not None, "elements": n_el, "substeps": b.substeps, "reset_flag": bool(unstable),
                "close_s": t_close - t_close0, "close_brakes": brake_floor < 0.0,
                "contact_angle_deg": math.degrees(first_touch - jaw.phi_meet) if math.isfinite(first_touch) else math.nan})
    if unstable:
        row.update(reason=f"simulation went unstable at {unstable_s:.4f} s; flagged, not scored", unstable_s=unstable_s,
                   passed=None, score=None, limiting=None)
        return row
    if ref is None:
        row.update(reason="never reached the lift", passed=False, score=0.0, limiting="entry")
        return row
    held = dropped in ("", "power")
    nbs = [s for s in (1, -1) if s != wall_side]
    worst_push = max(push[s] for s in nbs)
    worst_n = max(push_n[s] for s in nbs)
    worst_drag = max(drag[s] for s in nbs)
    margins = {"push": float(cfg.get("push_limit_mm", 3.0)) / max(worst_push, 1e-9),
               "push_force": float(cfg.get("push_limit_N", 5.0)) / max(worst_n, 1e-9),
               "drag": float(cfg.get("drag_limit_mm", 2.0)) / max(worst_drag, 1e-9),
               "slip": sc.max_slip / max(pk["slip"], 1e-9) if held else 0.0,
               "rotation": sc.max_rotation / max(pk["rot"], 1e-9)}
    body_lim = float(b1.get("body_contact_limit_N", 1.0))
    for k, lim_v, key in (("band", lim.band_force, "band"), ("panel", lim.panel_force, "panel"), ("bed", lim.bed_pressure, "pressure")):
        sf = sc.pressure_sf if key == "pressure" else 1.0
        if pk[k] > 1e-6:
            margins[key] = lim_v / (pk[k] * sf)
        if pk["i_" + k] > 1e-6:
            margins["impact_" + key] = lim_v / (pk["i_" + k] * sf)
    if pk["body"] > 0.1 * body_lim:
        margins["body"] = body_lim / pk["body"]
    if pk["i_body"] > 0.1 * body_lim:
        margins["impact_body"] = body_lim / pk["i_body"]
    margins = {k: float(min(v, 99.0)) for k, v in margins.items()}
    robust = lim.friction_scale < 1.0 - 1e-9
    rule = {k: margins[k] for k in ("push", "push_force", "drag", "slip", "rotation")} if robust else margins
    worst = min(rule, key=rule.get)
    entry_mm = float(row["entry_mm"])
    reached = (-z_min - clear) * 1e3
    reached_ok = reached >= entry_mm - 0.05
    passed = held and reached_ok and rule[worst] >= 1.0
    if not reached_ok:
        score, limiting = (min(reached / entry_mm, rule[worst]) if held else 0.0), "entry"
        reason = (f"the tips stall {reached:.1f} mm in, {entry_mm - reached:.1f} mm short: a neighbour takes over "
                  f"{stall_n:g} N" if z_stop is not None else f"the tips reach {reached:.1f} of {entry_mm:g} mm")
        if not held:
            reason += "; then dropped in the " + dropped
    elif not held:
        score, limiting, reason = 0.0, "slip", "dropped in the " + dropped
    else:
        score, limiting = rule[worst], worst
        reason = "" if passed else f"{worst} margin {rule[worst]:.2f}"
    sens_n = float(cfg.get("edge_sensitive_N", 4.0))
    any_n = max(push_n.values())
    edge_sensitive = any_n > sens_n
    if edge_sensitive:
        reason = "; ".join(x for x in (reason, f"a 5 ms push of {any_n:.1f} N is over {sens_n:g} N: "
                                               "sensitive to the edge contact's stiffness") if x)
    row.update({
        "passed": bool(passed), "score": score, "limiting": limiting, "reason": reason,
        "pass_rule": ("entry depth, then push, push force, drag, slip and rotation" if robust
                      else "entry depth, then all margins"),
        "margins": margins, "dropped_in": dropped,
        "entry_reached_mm": reached, "stalled": z_stop is not None, "stall_s": stall_s,
        "push_R_mm": push[1], "push_L_mm": push[-1], "push_R_N": push_n[1], "push_L_N": push_n[-1],
        "push_R_N_10ms": push_n10[1], "push_L_N_10ms": push_n10[-1], "edge_sensitive": bool(edge_sensitive),
        # single samples: the solver's, not scored
        "push_R_peak_N": push_raw[1], "push_L_peak_N": push_raw[-1],
        "entry_push_R_mm": push_entry[1], "entry_push_L_mm": push_entry[-1],
        "entry_push_R_N": push_n_entry[1], "entry_push_L_N": push_n_entry[-1],
        "drag_R_mm": drag[1], "drag_L_mm": drag[-1],
        "wall_mm": wall_mm if wall_side else math.nan, "wall_N": k_wall * wall_mm * 1e-3 if wall_side else math.nan,
        "wall_finger_N": push_n[wall_side] if wall_side else math.nan,
        "slip_mm": pk["slip"] * 1e3, "rotation_deg": math.degrees(pk["rot"]), "grip_N": grip_n,
        "band_N": pk["band"], "panel_N": pk["panel"], "bed_kPa": pk["bed"] / 1e3, "body_N": pk["body"],
        "impact_band_N": pk["i_band"], "impact_body_N": pk["i_body"],
        "sample_s": dt, "push_average_s": n_avg * dt, "peak_sample_s": every * dt,
        # the rated single-sample peaks' every-step maxima, not rated
        "band_step_N": pk_step["band"], "panel_step_N": pk_step["panel"], "bed_step_kPa": pk_step["bed"] / 1e3,
        "body_step_N": pk_step["body"], "impact_band_step_N": pk_step["i_band"],
        "impact_panel_step_N": pk_step["i_panel"], "impact_bed_step_kPa": pk_step["i_bed"] / 1e3,
        "impact_body_step_N": pk_step["i_body"],
        "mu_drag": mu_drag,
        "mu_bare": box.material.bare_friction * lim.friction_scale,
        "power_cut_held": bool(dropped == "" and power_ref is not None and power_slip <= sc.max_slip),
        "power_cut_slip_mm": power_slip * 1e3,
        "brake_use_Nm": float(np.median(brake_net[len(brake_net) // 2:])) if brake_net else 0.0,
        "brake_peak_Nm": max(brake_net) if brake_net else 0.0,
        "brake_on_s": brake_on if brake_on is not None else math.nan,
    })
    if record:
        row["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    return row
