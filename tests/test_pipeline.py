import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from graspsim2d.analytic import CORNER_ZONE, band_min_x, pass_opening, solve_grip, tip_half_gap
from graspsim2d.geometry import build_finger, to_view
from graspsim2d.objects import load_objects
from graspsim2d.pads import compression_modulus, shore_a_to_E
from graspsim2d.scenario import load_scenario
from graspsim2d.schema import SCHEMA, load_concepts, load_design_space

CFG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def setup():
    space = load_design_space(CFG / "design_space.yaml")
    concepts = {c.name: c for c in load_concepts(CFG / "concepts.yaml", space)}
    lib = load_objects(CFG / "objects.yaml")
    sc = load_scenario(CFG / "scenario.yaml")
    return space, concepts, lib, sc


def test_every_schema_variable_has_a_range(setup):
    space = setup[0]
    assert set(space.ranges) == {v.name for v in SCHEMA}


def test_bounding_box_swap_is_a_config_change(tmp_path, setup):
    raw = yaml.safe_load((CFG / "design_space.yaml").read_text())
    raw["variables"]["finger_reach"] = {"range": [7, 8], "unit": "in", "placeholder": False}
    raw["variables"]["pad_angle"] = {"value": 0.05, "unit": "rad"}
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump(raw))
    space = load_design_space(p)
    assert space["finger_reach"].lower == pytest.approx(177.8)
    assert space["finger_reach"].upper == pytest.approx(203.2)
    assert space["pad_angle"].fixed and space["pad_angle"].lower == pytest.approx(math.degrees(0.05))
    concepts = load_concepts(CFG / "concepts.yaml", space)
    for c in concepts:
        assert "pad_angle" not in c.active
        d = c.make(space)
        assert 177.8 - 1e-9 <= d.finger_reach <= 203.2 + 1e-9


def test_fixed_value_sets_the_nominal():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        space = load_design_space(CFG / "design_space_stroke45.yaml")
    v = space["max_opening"]
    assert v.fixed and v.lower == v.upper == v.nominal == pytest.approx(45.0)
    assert all(c.make(space).max_opening == pytest.approx(45.0) for c in load_concepts(CFG / "concepts.yaml", space))


def test_unknown_variable_is_rejected(tmp_path):
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump({"variables": {"finger_lenght": {"range": [1, 2], "unit": "mm"}}}))
    with pytest.raises(ValueError, match="finger_lenght"):
        load_design_space(p)


def test_jaw_gap_matches_closed_form(setup):
    space, concepts, _, _ = setup
    d = concepts["baseline_paddle"].make(space)
    f = build_finger(d)
    phi = math.radians(7.0)
    u, v = 120.0, 9.0
    x = to_view(np.array([[u, v]]), phi, f.pivot_x)[0, 0]
    assert 2 * x == pytest.approx(d.pivot_spacing + 2 * u * math.sin(phi) - 2 * v * math.cos(phi))


def test_band_min_x_uses_edge_crossings():
    sq = np.array([[1.0, -10.0], [5.0, -10.0], [5.0, 10.0], [1.0, 10.0]])
    tri = np.array([[0.0, 20.0], [4.0, -20.0], [8.0, 20.0]])
    assert band_min_x([sq], -1, 1) == pytest.approx(1.0)
    assert band_min_x([tri], -1, 1) == pytest.approx(1.9)


def test_shore_conversion_and_bonded_modulus():
    assert shore_a_to_E(40) == pytest.approx(1.69e6, rel=0.05)
    thin = compression_modulus(40, 0.04, 0.04, 0.002)
    thick = compression_modulus(40, 0.04, 0.04, 0.012)
    assert thin > thick > shore_a_to_E(40)


def test_objects_have_sourced_physical_values(setup):
    lib = setup[2]
    kinds = {o.kind for o in lib.objects}
    assert kinds == {"box", "sheet"}
    for o in lib.objects:
        assert o.mass > 0 and o.material.friction > 0 and o.material.allow_pressure > 0
        if o.kind == "box":
            assert 0 < o.com_below_top < o.height and o.headspace > 0
        else:
            assert o.material.bending_stiffness > 0 and o.material.bending_strength > 0
    hbo = lib.by_id("hbo_28oz")
    assert hbo.mass == pytest.approx(0.8938, abs=0.002)


BOX_COLUMNS = ("product_title,size_label,upc,net_mass_g,box_mass_g,height_mm,width_mm,depth_mm,"
               "fill_height_mm,com_height_mm,bulk_density_g_cm3,source,confidence")


def _objects_with(tmp_path, box_defaults=None, box_tables=None):
    raw = yaml.safe_load((CFG / "objects.yaml").read_text())
    raw["box_defaults"].update(box_defaults or {})
    if box_tables is None:
        box_tables = [{**t, "file": str(CFG / t["file"])} for t in raw["box_tables"]]
    raw["box_tables"] = box_tables
    (tmp_path / "objects.yaml").write_text(yaml.safe_dump(raw))
    return load_objects(tmp_path / "objects.yaml")


def test_box_table_rows_become_boxes(tmp_path, setup):
    gm = setup[2].by_id("gm_cheerios_family_18oz")
    assert gm.kind == "box" and gm.label == "Cheerios Family Size 18 oz"
    assert (gm.thickness, gm.width, gm.height) == pytest.approx((0.094, 0.196, 0.325))
    assert gm.fill_height == pytest.approx(0.234) and gm.com_height == pytest.approx(0.126)
    assert gm.headspace == pytest.approx(0.325 - 0.234) and gm.mass == pytest.approx(0.626)
    # Blank fill height: bulk density over the footprint, then fill_fraction; blank CoM: shell plus contents.
    # Without `names`, a row's id is the prefix plus its UPC, else its title.
    (tmp_path / "t.csv").write_text(BOX_COLUMNS + "\nTest,A,,300,60,300,200,50,,,0.15,x,low"
                                    "\nTest,B,,300,60,300,200,50,,,,x,low\nTest,C,0123,300,60,300,200,50,,,,x,placeholder\n")
    lib = _objects_with(tmp_path, box_tables=[{"file": "t.csv", "material": "cereal_carton_flakes"}])
    a, b = lib.by_id("test_a"), lib.by_id("test_b")
    assert a.fill_height == pytest.approx(0.2)
    assert b.fill_height == pytest.approx(0.24) and b.com_height == pytest.approx((60 * 0.15 + 300 * 0.12) / 360)
    assert lib.by_id("0123").label == "Test C (placeholder)"
    (tmp_path / "t.csv").write_text("product_title,net_mass_g,height_mm,width_mm\nTest,300,300,200\n")
    with pytest.raises(ValueError, match="depth_mm"):
        load_objects(tmp_path / "objects.yaml")


def test_box_grasp_axis_is_configurable(tmp_path, setup):
    deep = setup[2].by_id("hbo_28oz")
    wide = _objects_with(tmp_path, box_defaults={"grasp_axis": "width"}).by_id("hbo_28oz")
    assert (deep.grasp_axis, wide.grasp_axis) == ("depth", "width")
    assert (wide.thickness, wide.width) == pytest.approx((deep.width, deep.thickness))
    assert wide.mass == pytest.approx(deep.mass) and wide.com_height == pytest.approx(deep.com_height)
    assert wide.inertia > deep.inertia
    with pytest.raises(ValueError, match="grasp_axis"):
        _objects_with(tmp_path, box_defaults={"grasp_axis": "diagonal"})


def test_cheerios_family_size_validation(setup):
    """Platform finger on the 94 mm deep Cheerios Family Size box, against the CAD study in the
    project store (docs/general-mills-cereal-boxes.md): the jaws open 17.4 deg each (94 mm tip gap)
    to pass the box, then meet only its top edges, so the squeeze pushes the box out and holding
    needs mu >= tan(theta): 0.36 at 20 deg per jaw, with the box top 18 mm inside the tips."""
    space, concepts, lib, sc = setup
    box = lib.by_id("gm_cheerios_family_18oz")
    base = concepts["baseline_paddle"].make(space)
    f = build_finger(base, sc.keep_out, sc.closed_limit)
    phi = pass_opening(f, box, sc)
    assert math.degrees(phi) == pytest.approx(17.4, abs=0.1) and 2 * tip_half_gap(f, phi) == pytest.approx(94.0)
    # Rigid contact, as in the CAD study: the printed face (pad at 90 A, face kept where the CAD has
    # it) on a rigid box edge. Insertion 20.8 mm puts the box top 18 mm inside the open tips.
    stock = base.replace(pad_shore=90.0)
    rigid = dataclasses.replace(box, material=dataclasses.replace(box.material, k_surface=1e10))
    r = solve_grip(stock, rigid, sc, depth=0.0208)
    assert r.ok and r.info["pads"] == "tab"
    assert r.info["phi_touch_deg"] == pytest.approx(20.0, abs=0.1)
    assert r.info["cop_below_top_mm"] < 1.0 and r.info["contact_length_mm"] < CORNER_ZONE
    tilt = r.info["contact_face_tilt_deg"]
    assert tilt == pytest.approx(20.0, abs=0.3)
    assert r.info["squeeze_out_ratio"] == pytest.approx(math.tan(math.radians(tilt)), abs=0.005)
    assert r.info["squeeze_out_ratio"] == pytest.approx(0.36, abs=0.01)
    # Deeper, the top edges land on the paddle's main face. The CAD study's deepest case has the
    # box top 51 mm below the pivots (99 mm insertion) and 65 mm inside the open tips, touching
    # 55 mm from the pivot line at 46.5 deg per jaw, where holding needs mu 1.05 = tan(46.5 deg).
    # Loaded, the jaws sit about half a degree further closed, and the screen's mu is tan of the
    # loaded face angle; the drive torque alone sets the normal force, with no overshoot.
    r = solve_grip(stock, rigid, sc, depth=0.099)
    assert r.ok and r.info["pads"] == "paddle"
    assert r.info["phi_touch_deg"] == pytest.approx(46.5, abs=0.3)
    assert math.tan(math.radians(r.info["phi_touch_deg"])) == pytest.approx(1.05, abs=0.01)
    assert r.info["contact_face_tilt_deg"] == pytest.approx(r.info["phi_touch_deg"] - 0.5, abs=0.2)
    assert r.info["squeeze_out_ratio"] == pytest.approx(math.tan(math.radians(r.info["contact_face_tilt_deg"])), abs=0.005)
    assert r.info["squeeze_out_ratio"] == pytest.approx(1.04, abs=0.01)
    assert r.info["normal_force_N"] * r.info["arm_mm"] / 1000 == pytest.approx(stock.finger_torque, rel=1e-6)
    assert r.info["phi_touch_deg"] < r.info["approach_opening_deg"] < r.info["phi_touch_deg"] + 1.0
    # With the modelled box wall, the screen's own pick puts the corner on the main face, where the
    # squeeze needs more friction than the carton's 0.5, so the hold fails.
    r = solve_grip(base, box, sc)
    assert r.ok and r.info["pads"] == "paddle" and r.margins["hold"] < 1.0
    assert r.info["squeeze_out_ratio"] > box.material.friction


def test_force_balance_of_the_analytic_grip(setup):
    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    r = solve_grip(d, lib.by_id("mini_1oz"), sc)
    assert r.ok
    # drive torque = normal force x lever arm about the pivot
    assert r.info["normal_force_N"] * r.info["arm_mm"] / 1000 == pytest.approx(d.finger_torque, rel=1e-6)


def test_least_grip_for_hold_and_pitch(setup):
    """The screen's least grip per finger for hold and for pitch: each margin reaches 1 where the
    applied grip passes it, and the safety factors only raise it."""
    space, concepts, lib, sc = setup
    seen = set()
    for name in ("baseline_paddle", "self_aligning", "crowned"):
        d = concepts[name].make(space)
        for oid in ("mini_1oz", "hbo_28oz", "gm_cheerios_family_18oz"):
            r = solve_grip(d, lib.by_id(oid), sc)
            assert r.ok
            i = r.info
            assert (r.margins["hold"] >= 1.0) == (i["min_grip_hold_sf_N"] <= i["normal_force_N"])
            assert (r.margins["pitch"] >= 1.0) == (i["min_grip_pitch_sf_N"] <= i["normal_force_loaded_N"])
            assert i["min_grip_hold_N"] <= i["min_grip_hold_sf_N"] and i["min_grip_pitch_N"] <= i["min_grip_pitch_sf_N"]
            seen.add(r.margins["hold"] >= 1.0)
    assert seen == {True, False}
    # with no squeeze-out the hold needs about half the load over the friction coefficient; the
    # friction moment about the pivot shifts it by under a tenth here
    o = lib.by_id("chipboard_1p5mm")
    r = solve_grip(concepts["baseline_paddle"].make(space), o, sc)
    w = o.mass * (9.81 + sc.vertical_accel)
    assert r.info["squeeze_out_ratio"] == 0.0
    assert r.info["min_grip_hold_N"] == pytest.approx(w / (2 * o.material.friction * sc.friction_scale), rel=0.1)


def test_diverging_box_and_converging_sheet(setup):
    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    box = solve_grip(d, lib.by_id("cheerios_8p9oz"), sc)
    sheet = solve_grip(d, lib.by_id("chipboard_1p5mm"), sc)
    assert box.ok and sheet.ok
    assert box.info["phi_deg"] > 5.0
    assert sheet.info["phi_deg"] < 0.0


def test_baseline_matches_the_platform_finger(setup):
    space, concepts, _, sc = setup
    d = concepts["baseline_paddle"].make(space)
    f = build_finger(d, sc.keep_out, sc.closed_limit)
    assert f.feasible, f.issues
    paddle, tab = f.pads
    assert (paddle.width, tab.width) == (95.0, 16.0)
    assert paddle.face_u - paddle.length / 2 == pytest.approx(49.9)
    assert tab.face_u + tab.length / 2 == pytest.approx(d.finger_reach)
    assert tab.face_u - tab.length / 2 == pytest.approx(129.5)  # the arm's recessed notch ends here
    # CAD: faces parallel and about 2.5 mm apart when closed
    assert d.pivot_spacing - 2 * paddle.face_v == pytest.approx(2.4)
    assert tab.face_v == pytest.approx(paddle.face_v)
    assert 5.0 <= f.ligament < d.finger_thickness and f.cut_body
    # clears the hub by the CAD's 0.86 mm gap over the whole stroke
    assert f.hub_clearance == pytest.approx(sc.keep_out.clearance, abs=0.01)


def test_stroke_runs_from_where_the_fingers_meet(setup):
    space, concepts, lib, sc = setup
    for c in concepts.values():
        d = c.make(space)
        f = build_finger(d, sc.keep_out, sc.closed_limit)
        assert d.max_opening == 90.0
        assert f.phi_max - f.phi_meet == pytest.approx(math.radians(90.0))
        for o in lib.objects:
            r = solve_grip(d, o, sc, finger=f)
            if not r.ok:
                continue
            # each grasp opens only as far as the object needs, and closes from there
            assert r.info["phi_touch_deg"] < r.info["approach_opening_deg"] <= math.degrees(f.phi_max)
            assert r.info["approach_stroke_deg"] == pytest.approx(r.info["approach_opening_deg"] - math.degrees(f.phi_meet))


def test_thick_fingers_meet_at_the_pad(setup):
    space, concepts, _, sc = setup
    d = concepts["cranked_flat"].make(space).replace(finger_thickness=30.5, crank_offset=15.8, crank_position=0.58,
                                                     pad_thickness=9.9, pad_angle=14.6)
    f = build_finger(d, sc.keep_out, sc.closed_limit)
    # the beam's square end behind the pivot crosses the centreline mid-stroke on a finger this thick
    assert min(to_view(b.poly, math.radians(45.0), f.pivot_x)[:, 0].min() for b in f.body) < 0.0
    assert min(to_view(p.outline(), f.phi_meet, f.pivot_x)[:, 0].min() for p in f.pads) == pytest.approx(0.0, abs=0.05)
    assert math.degrees(f.phi_meet) < 20.0


def test_brackets_clear_the_baseline_over_the_whole_stroke(setup):
    from graspsim2d.geometry import clearance_profile

    space, concepts, _, sc = setup
    ko = sc.keep_out
    assert [b.name for b in ko.brackets] == ["FT Dynamic bracket (top)", "FT Dynamic bracket (bot)"]
    d = concepts["baseline_paddle"].make(space)
    f = build_finger(d, ko, sc.closed_limit)
    assert f.feasible, f.issues
    assert f.hub_clearance == pytest.approx(ko.clearance, abs=0.01)
    assert f.bracket_clearance > 10.0
    prof = clearance_profile(f, ko, np.radians(np.linspace(0.0, 90.0, 19)))
    assert set(prof) == {"hub", *(b.name for b in ko.brackets)}
    assert min(v.min() for k, v in prof.items() if k != "hub") == pytest.approx(f.bracket_clearance, abs=0.5)
    assert prof["hub"].min() == pytest.approx(f.hub_clearance, abs=0.05)
    # A full-width root at the brackets' heights would run through their fillets beside the
    # hub; the platform finger's root has material there only on its outer side.
    full = build_finger(d, dataclasses.replace(ko, root_length=0.0), sc.closed_limit)
    assert full.bracket_clearance < 0.0 and any("FT Dynamic bracket" in s for s in full.issues)
    # the root clears the brackets in 3D between the jaw angles root_clear; nothing else stops the jaw before that
    from graspsim2d.geometry import widest_opening

    w, why = widest_opening(f, ko)
    assert math.degrees(w) == pytest.approx(130.5 - math.degrees(f.phi_meet), abs=0.01) and "root" in why
    assert not build_finger(d.replace(max_opening=135.0), ko, sc.closed_limit).feasible
    assert not build_finger(d, dataclasses.replace(ko, root_clear=(0.0, ko.root_clear[1])), sc.closed_limit).feasible


def test_keep_out_rejects_pads_in_the_bracket_and_limits_insertion(setup):
    space, concepts, lib, sc = setup
    c = concepts["baseline_paddle"]
    i = c.active.index("pad_length")
    x = list(c.to_vector(c.make(space)))
    x[i] = 70.0  # paddle would start 31.5 mm from the pivot line, inside the bracket hub
    f = build_finger(c.make(space, x), sc.keep_out, sc.closed_limit)
    assert not f.feasible and any("keep-out" in s for s in f.issues)
    assert build_finger(c.make(space, x)).feasible  # fine without the bracket
    d = c.make(space)
    r = solve_grip(d, lib.by_id("mini_1oz"), sc)
    z_top = -d.finger_reach + r.info["insertion_mm"]
    (cx, cz, rad), = sc.keep_out.radii()
    assert z_top <= cz - rad + 1e-6


def test_contact_width_is_capped_by_the_object(setup):
    from graspsim2d.analytic import _pad_models

    space, concepts, lib, sc = setup
    f = build_finger(concepts["baseline_paddle"].make(space), sc.keep_out, sc.closed_limit)
    mini, cheerios = lib.by_id("mini_1oz"), lib.by_id("cheerios_8p9oz")
    assert _pad_models(f, mini)[0].w == pytest.approx(mini.width)
    assert _pad_models(f, cheerios)[0].w == pytest.approx(0.095)


def test_collision_model_leaves_the_hub_notch_open(setup):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from graspsim2d.geometry import collision_rects, rect_polygon
    from graspsim2d.mjsim import build_model

    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    f = build_finger(d, sc.keep_out, sc.closed_limit)
    material = unary_union([Polygon(P) for P in f.cut_body])
    uncut = unary_union([Polygon(p.poly) for p in f.body if p.rect is not None])
    rects = collision_rects(f)
    boxes = unary_union([Polygon(rect_polygon(*r)) for _, r in rects])
    assert uncut.difference(material).area == pytest.approx(253.0, abs=5.0)  # the platform paddle's hub notch
    assert boxes.difference(material).area < 0.01
    assert boxes.area > 0.98 * uncut.intersection(material).area
    xml, _ = build_model(d, lib.by_id("mini_1oz"), sc, 0.03, finger=f)
    assert xml.count('<geom name="R_beam') == len(rects) > 1


def test_close_time_grows_with_the_start_opening(setup):
    from graspsim2d.mjsim import close_time

    space, concepts, _, sc = setup
    f = build_finger(concepts["crowned"].make(space), sc.keep_out, sc.closed_limit)
    assert close_time(sc, f, f.phi_meet + math.radians(10.0)) == sc.close_time
    assert close_time(sc, f, f.phi_max) == pytest.approx(math.radians(90.0) / sc.closing_speed + sc.squeeze_time)


def test_dropped_runs_keep_every_margin(setup):
    from graspsim2d.mjsim import simulate

    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space).replace(finger_torque=0.25)
    r = simulate(d, lib.by_id("hbo_28oz"), sc, record=False)
    assert not r.passed and r.reason == "dropped" and r.score == 0.0
    m = r.metrics["margins"]
    assert m["slip"] == 0.0 and {"rotation", "pressure", "panel"} <= set(m)
    assert m["panel"] == pytest.approx(lib.by_id("hbo_28oz").unsupported_force / r.metrics["panel_force_N"])


def test_a_run_that_resets_is_flagged_and_rerun(setup, monkeypatch):
    from graspsim2d import mjsim

    space, concepts, lib, sc = setup
    d, o = concepts["self_aligning"].make(space), lib.by_id("mini_1oz")
    depth = solve_grip(d, o, sc).info["insertion_mm"] / 1e3
    ref = mjsim.simulate(d, o, sc, depth=depth, record=False)
    assert ref.passed and ref.metrics["resets"] == 0 and not ref.blowups

    step, seen = mjsim.mujoco.mj_step, {}

    def reset_once(m, dd):
        seen["autoreset_off"] = bool(m.opt.disableflags & int(mjsim.mujoco.mjtDisableBit.mjDSBL_AUTORESET))
        if dd.time >= 0.2 and "reset" not in seen:
            seen["reset"] = True
            mjsim.mujoco.mj_resetData(m, dd)  # what MuJoCo's auto-reset does to a run that blows up
        step(m, dd)

    monkeypatch.setattr(mjsim.mujoco, "mj_step", reset_once)
    r = mjsim.simulate(d, o, sc, depth=depth, record=False)
    dt = sc.sim["timestep_s"]
    assert seen["autoreset_off"]
    assert r.metrics["resets"] == 1 and r.metrics["reset_s"] == pytest.approx(0.2, abs=1e-3)
    assert r.blowups == [{"t_s": pytest.approx(0.2, abs=1e-3), "phase": "close", "timestep_s": pytest.approx(dt)}]
    assert r.metrics["timestep_s"] == pytest.approx(dt / 2)
    m, m_ref = r.metrics["margins"], ref.metrics["margins"]
    assert r.passed and min(m, key=m.get) == min(m_ref, key=m_ref.get)
    assert r.score == pytest.approx(ref.score, rel=0.1)


def test_a_run_that_keeps_blowing_up_is_never_scored(setup, monkeypatch, capsys):
    from graspsim2d import mjsim
    from graspsim2d.cli import _passes, _print_outcome
    from graspsim2d.simulator import MujocoSimulator, verdict

    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space).replace(finger_torque=0.25)
    step, warned = mjsim.mujoco.mj_step, []

    def blow_up(m, dd):
        if dd.time >= 0.1:
            dd.qvel[:] = 1e11  # past MuJoCo's bad-state limit, on every try
        step(m, dd)
        warned.append(any(dd.warning[w].number for w in mjsim.BAD_STATE))

    monkeypatch.setattr(mjsim.mujoco, "mj_step", blow_up)
    r = mjsim.simulate(d, lib.by_id("hbo_28oz"), sc, record=False)
    dt = sc.sim["timestep_s"]
    assert warned[-1]
    assert (r.passed, r.reason, r.score, "margins" in r.metrics) == (None, "simulation went unstable", None, False)
    assert r.metrics["resets"] == 4 and r.metrics["timestep_s"] == pytest.approx(dt / 8)
    assert [b["timestep_s"] for b in r.blowups] == pytest.approx([dt, dt / 2, dt / 4, dt / 8])
    assert all(b["phase"] == "close" and b["t_s"] == pytest.approx(0.1, abs=1e-3) for b in r.blowups)
    out = MujocoSimulator().run(d, lib.by_id("hbo_28oz"), sc).to_dict()
    assert (out["passed"], out["score"], verdict(out)) == (None, None, "UNSCORED")
    _print_outcome(out)
    line = capsys.readouterr().out
    assert "UNSCORED  score   n/a  simulation went unstable" in line and "flagged, not scored" in line
    assert _passes([out, {**out, "passed": True}, {**out, "passed": False}]) == "1/2 scored, 1 unscored (blew up at every timestep)"


def test_an_unscored_run_is_counted_apart_never_as_a_fail(tmp_path, setup, monkeypatch):
    from graspsim2d import sweep
    from graspsim2d.cli import CONFIG_FILES
    from graspsim2d.mjsim import UNSTABLE
    from graspsim2d.search import SearchConfig, SearchRun, objective_from_scores

    space, concepts, lib, sc = setup
    base = concepts["baseline_paddle"].make(space)

    def run(oid, passed, score, reason=""):
        return {"object_id": oid, "passed": passed, "score": score, "reason": reason, "limiting": reason or "hold",
                "metrics": {} if passed is None else {"margins": {"hold": score}},
                "analytic_score": 1.0, "analytic_limiting": "hold", "analytic_margins": {"hold": 1.0}}

    rows = [{"point": 0, **run("a", True, 1.4)}, {"point": 0, **run("b", None, None, UNSTABLE)},
            {"point": 0, **run("c", False, 0.6, "hold margin 0.60")}]
    monkeypatch.setattr(sweep, "run_batch", lambda *a, **k: rows)
    (s,) = sweep.run_sweep(base, [("finger_torque", [3.0])], ["a", "b", "c"], {"objects": "", "scenario": ""})["summary"]
    assert (s["n_pass"], s["n_unscored"], s["n_objects"], s["worst_object"]) == (1, 1, 3, "c")
    assert s["J"] == pytest.approx(objective_from_scores([1.4, 0.6], 0.25))

    # "flagged" is unscored on the object that "steady" fails, and both pass the rest
    search = SearchRun({k: str(CFG / v) for k, v in CONFIG_FILES.items()}, tmp_path, SearchConfig(), log=lambda s: None)
    search._carried = {}
    ids = search.lib.ids()
    x = ids[0]
    fin = [{"name": n, "concept": "baseline_paddle", "why": [n], "design": dict(base.to_dict(), label=n), "J": 5.0, "n_pass": 3}
           for n in ("steady", "flagged")]
    dyn = [dict(run(o, True, 1.5), design_name=n) for n in ("steady", "flagged") for o in ids[1:]]
    dyn += [dict(run(x, False, 0.5, "hold margin 0.50"), design_name="steady"), dict(run(x, None, None, UNSTABLE), design_name="flagged")]
    rnd = search._summarise_round(1, {}, {}, fin, dyn, {})
    by = {d["name"]: d for d in rnd["designs"]}
    tally = lambda d: (d["sim_pass"], d["sim_fail"], d["sim_unscored"], d["unscored"])
    assert tally(by["flagged"]) == (len(ids) - 1, 0, 1, [x]) and tally(by["steady"]) == (len(ids) - 1, 1, 0, [])
    final = search._final([rnd])
    assert final["ranking"] == ["flagged", "steady"] and final["flagged"] == {"flagged": [x]}
    assert final["per_object"][x]["design"] == "steady"


def test_drawn_length_includes_the_cranked_tip(setup):
    from graspsim2d.geometry import drawn_length

    space, concepts, _, sc = setup
    straight = concepts["baseline_paddle"].make(space)
    assert drawn_length(build_finger(straight, sc.keep_out, sc.closed_limit)) == pytest.approx(straight.finger_reach)
    cranked = concepts["cranked_flat"].make(space)
    assert cranked.crank_offset > 0
    f = build_finger(cranked, sc.keep_out, sc.closed_limit)
    assert drawn_length(f) == pytest.approx(cranked.finger_reach + cranked.finger_thickness / 4)


def test_fingers_are_geared_one_to_one(setup):
    import mujoco

    from graspsim2d.mjsim import build_model

    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    xml, meta = build_model(d, lib.by_id("chipboard_1p5mm"), sc, 0.03)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("jR", "jL")}
    start = math.radians(30.0)  # well inside the stroke, wider than this sheet needs
    assert meta["phi0"] < start
    for n in adr.values():
        data.qpos[n] = start
    for _ in range(1500):  # drive the right finger only; the gear must carry the left one
        data.ctrl[2] = -d.finger_torque
        mujoco.mj_step(model, data)
    jR, jL = data.qpos[adr["jR"]], data.qpos[adr["jL"]]
    assert start - jR > math.radians(2.0)
    assert jL == pytest.approx(jR, abs=1e-3)


def test_dynamic_check_agrees_with_screen_on_static_grip(setup):
    from graspsim2d.mjsim import simulate

    space, concepts, lib, sc = setup
    d = concepts["self_aligning"].make(space)
    o = lib.by_id("mini_1oz")
    an = solve_grip(d, o, sc)
    sim = simulate(d, o, sc, depth=an.info["insertion_mm"] / 1000)
    assert sim.metrics["grip_N"] == pytest.approx(an.info["normal_force_N"], rel=0.05)
    assert sim.metrics["slip_mm"] < sc.max_slip * 1000
    assert sim.metrics["peak_pressure_kPa"] == pytest.approx(an.info["loaded_pressure_kPa"], rel=0.25)


def test_simulator_interface_wraps_both_levels(setup):
    from graspsim2d.mjsim import simulate
    from graspsim2d.simulator import make_simulator

    space, concepts, lib, sc = setup
    d, o = concepts["baseline_paddle"].make(space), lib.by_id("mini_1oz")
    ref = solve_grip(d, o, sc)
    an = make_simulator("analytic").run(d, o, sc)
    assert (an.score, an.limiting, an.passed) == (pytest.approx(ref.score), ref.limiting, ref.passed)
    assert an.insertion_mm == pytest.approx(ref.info["insertion_mm"])
    mj = make_simulator("mujoco").run(d, o, sc)
    direct = simulate(d, o, sc, depth=ref.info["insertion_mm"] / 1e3, record=False)
    assert (mj.score, mj.passed) == (pytest.approx(direct.score), direct.passed)
    assert mj.screen["score"] == pytest.approx(ref.score)
    assert set(an.to_dict()) == set(mj.to_dict())
    with pytest.raises(ValueError):
        make_simulator("fem")


def test_sweep_runs_the_grid_and_saves_it(tmp_path, setup):
    import csv

    from graspsim2d.cli import main
    from graspsim2d.sweep import parse_vary

    space, concepts, lib, sc = setup
    out = tmp_path / "sweep"
    main(["--config", str(CFG), "sweep", "baseline_paddle", "--vary", "finger_torque=2:4:3", "--vary", "pad_shore=40,60",
          "--objects", "mini_1oz,b_flute_3mm", "--out", str(out), "--workers", "2"])
    rows = list(csv.DictReader((out / "sweep.csv").open()))
    summary = list(csv.DictReader((out / "summary.csv").open()))
    assert len(rows) == 12 and [(float(r["finger_torque"]), float(r["pad_shore"])) for r in summary] == [
        (2, 40), (2, 60), (3, 40), (3, 60), (4, 40), (4, 60)]
    row = next(r for r in rows if float(r["finger_torque"]) == 3 and float(r["pad_shore"]) == 60 and r["object_id"] == "mini_1oz")
    d = concepts["baseline_paddle"].make(space).replace(finger_torque=3.0, pad_shore=60.0)
    assert float(row["score"]) == pytest.approx(solve_grip(d, lib.by_id("mini_1oz"), sc).score)
    assert (out / "figures" / "sweep.png").exists() and (out / "objects.yaml").exists()
    with pytest.raises(ValueError, match="unknown design variable"):
        parse_vary("finger_length=1,2")
    one = tmp_path / "sweep1"
    main(["--config", str(CFG), "sweep", "baseline_paddle", "--vary", "finger_torque=1:3:3", "--objects", "mini_1oz,hbo_28oz",
          "--out", str(one), "--workers", "2"])
    (one / "figures" / "sweep.png").unlink()
    main(["--config", str(CFG), "figures", str(one)])
    assert (one / "figures" / "sweep.png").exists()


def test_run_directory_keeps_its_box_tables(tmp_path):
    from graspsim2d.cli import CONFIG_FILES, save_inputs

    raw = yaml.safe_load((CFG / "objects.yaml").read_text())
    raw["box_tables"][0]["file"] = str(CFG / "general_mills_cereal_boxes.csv")
    (tmp_path / "objects.yaml").write_text(yaml.safe_dump(raw))
    paths = {k: str(CFG / v) for k, v in CONFIG_FILES.items()} | {"objects": str(tmp_path / "objects.yaml")}
    save_inputs(tmp_path / "run", paths)
    assert yaml.safe_load((tmp_path / "run" / "objects.yaml").read_text())["box_tables"][0]["file"] == "general_mills_cereal_boxes.csv"
    assert "gm_cheerios_family_18oz" in load_objects(tmp_path / "run" / "objects.yaml").ids()


def test_bed_limit_applies_only_below_the_fill_line(setup):
    """Test conditions rev 2.11: a grip wholly on the empty panel takes the panel force limit alone."""
    space, concepts, lib, sc = setup
    fs = lib.by_id("gm_cheerios_family_18oz")
    d = concepts["baseline_paddle"].make(space)
    shallow = solve_grip(d, fs, sc, depth=0.06)
    assert shallow.ok and "pressure" not in shallow.margins and "panel" in shallow.margins
    assert shallow.info["bed_pressure_kPa"] == 0.0 < shallow.info["peak_pressure_kPa"]



@pytest.fixture(scope="module")
def batch1():
    from graspsim2d import batch1 as b

    return b


def test_batch1_specs_build_feasible_jaws(batch1):
    for key in batch1.SPEC_FILES:
        j = batch1.get_jaw(key, 45.0)
        assert j.feasible, (key, j.issues)
        assert j.right.faces and all(f.width > 0 for f in j.right.faces)


def test_batch1_comb_layers_keep_their_widths(batch1):
    from graspsim2d.jaw import element_layout

    def widths(face):
        lay = element_layout(face, 2.0, 0.5)
        return lay["c"][:, 0], lay["w"]

    # body_width_mm already gives each comb step its half of the paddle: S5's 75 mm on both fingers, S4's thumb 70.
    for rd in (45.0, 90.0):
        j = batch1.get_jaw("S5", rd)
        for side in (1, -1):
            faces = {f.name: f for f in j.sides[side].faces}
            for name in ("bare0", "bare1", "bare2"):
                assert widths(faces[name])[1] == pytest.approx(75.0), (rd, side, name)
    j = batch1.get_jaw("S4", 45.0)
    thumb = {f.name: f for f in j.sides[-1].faces}
    for name in ("bare1", "bare2", "bare3"):
        assert widths(thumb[name])[1] == pytest.approx(70.0), name
    # S4's outside floor crosses three parts, element by element: the 80 mm layer the relief leaves, half of the 97 mm
    # station where the thumb's step sweeps past u 78, then the table's 97 and 60.
    floor = next(f for f in j.sides[1].faces if f.name == "bare9")
    for u, w in zip(*widths(floor)):
        assert w == pytest.approx(80.0 if u < 78 else 48.5 if u < 82.8 else 97.0 if u < 90 else 60.0), u


def test_batch1_screen_bed_and_depth_stop(batch1):
    from graspsim2d.screen1 import Limits, scan_box, solve

    c = batch1.ctx()
    lim = Limits.from_config(c["b1"], "T1")
    fs = batch1.get_obj("gm_cheerios_family_18oz")
    j = batch1.get_jaw("r2-09", 45.0)
    g = solve(j, fs, c["sc"], lim, 30.0)
    assert g.ok and "pressure" not in g.margins and g.info["bed_pressure_kPa"] == 0.0
    r = scan_box(j, fs, c["sc"], lim)
    assert r.ended.startswith("depth stop") and math.isfinite(r.stop_mm)
    assert r.depth <= r.stop_mm - lim.placement + 1e-6


def test_batch1_screen_moment_goes_to_friction_first(batch1):
    from graspsim2d.screen1 import Limits, scan_box, solve

    c = batch1.ctx()
    lim = Limits.from_config(c["b1"], "T1")
    j = batch1.get_jaw("S2", 45.0)
    # SIM round 9: on S2's 8 mm tooth pad, 100 mm deep at loose fill, MuJoCo's bed pressure matches the screen with no
    # moment term (21.8 - 23.4 kPa at 1.75 N m, 32.9 - 34.4 at 3), since the friction couple carries the moment.
    for tau, lo, hi in ((1.75, 21.5, 23.7), (3.0, 32.6, 34.7)):
        for box in ("gm_021908455587", "gm_016000275492", "gm_016000222083"):
            g = solve(j, batch1.get_obj(box, "loose"), c["sc"], lim, 100.0, tau=tau)
            i = g.info
            assert g.ok and i["faces"] == "tooth" and i["short_pads"] and lo < i["bed_pressure_kPa"] < hi
            assert i["bed_moment_Nm"] == pytest.approx(max(0.0, i["moment_Nm"] - i["friction_couple_Nm"]) / 2)
            # Within 10 deg the short pad's pressure couple gives about 0.035 N m, not 2 N L / 3.
            assert i["pressure_couple_Nm"] < 0.04 < 2 * i["grip_loaded_N"] * i["contact_length_mm"] * 1e-3 / 3
    # A long pad keeps its whole pressure couple.
    g = solve(batch1.get_jaw("r2-09", 45.0), batch1.get_obj("gm_016000275492", "catalog"), c["sc"], lim, 60.0)
    assert g.ok and not g.info["short_pads"]
    assert g.info["pressure_couple_Nm"] == pytest.approx(2 * g.info["grip_loaded_N"] * g.info["contact_length_mm"] * 1e-3 / 3)
    # VAL R9.4: S2's dam top is a contact face, and each row reports its best grip with the pad below the fill line.
    assert [f for f in j.right.grip_faces if f.rigid and 1.0 < f.pad.length < 3.0
            and 128.0 < float(f.pad.face_samples(2)[0][0, 0]) < 136.0]
    row = scan_box(j, batch1.get_obj("gm_021908455587", "loose"), c["sc"], lim).row()
    assert row["deep_depth_mm"] >= row["depth_mm"] and row["deep_score"] > 0.0 and row["deep_bed_kPa"] > 0.0


def test_batch1_brake_holds_the_power_cut(batch1):
    from graspsim2d import mj1

    c = batch1.ctx()
    j = batch1.get_jaw("r2-09", 45.0)
    o = batch1.get_obj("gm_016000275492")
    b1 = {**c["b1"], "shake_s": 1.0}
    held = mj1.simulate(j, o, c["sc"], b1, "T1", 60.0, stops=[], power_s=0.5)
    assert held["power_cut_held"] and held["brake_on_s"] - held["power_cut_s"] < 1e-3
    assert held["drive_hold_Nm"] == pytest.approx(2 * j.torque, rel=0.02)
    assert held["brake_use_Nm"] == pytest.approx(held["drive_hold_Nm"], rel=0.05)
    assert held["mount_fatigue_flag"] and held["finger_torque_Nm"] == j.torque
    # Test conditions 2.16: friction use is both fingers' 10 ms mean, reported but outside every pass rule,
    # and element pressure comes from the element springs.
    assert "friction" not in held["margins"] and held["pass_rule"] == "all margins"
    assert held["substeps"] == 2  # r2-09's short faces split at the 1.5 mm fine pitch
    assert 0.0 < held["friction_use"] <= held["friction_use_sample_peak"] + 1e-9
    assert held["pressure_kPa"] > 0.0 and held["pressure_kPa_contact_sample"] > 0.0
    loose = mj1.simulate(j, o, c["sc"], b1, "T1", 60.0, stops=[], power_s=0.5, brake=1.0, friction_scale=0.8)
    assert not loose["power_cut_held"] and loose["pass_rule"] == "holds: slip and rotation"
    hold = loose["dropped_in"] != "main" and loose["margins"]["slip"] >= 1.0 and loose["margins"]["rotation"] >= 1.0
    assert loose["passed"] == hold and loose["limiting"] in ("slip", "rotation")


def test_batch1_blown_up_runs_stop_instead_of_restarting(batch1, monkeypatch):
    from graspsim2d import mj1

    c = batch1.ctx()
    step, seen = mj1.mujoco.mj_step, {}

    def reset_once(m, d):
        seen["autoreset_off"] = bool(m.opt.disableflags & int(mj1.mujoco.mjtDisableBit.mjDSBL_AUTORESET))
        if d.time >= 0.2 and not seen.get("reset"):
            seen["reset"] = True
            mj1.mujoco.mj_resetData(m, d)  # what MuJoCo's auto-reset does to a run that blows up
        step(m, d)

    monkeypatch.setattr(mj1.mujoco, "mj_step", reset_once)
    j = batch1.get_jaw("r2-09", 45.0)
    r = mj1.simulate(j, batch1.get_obj("c_flute_4mm"), c["sc"], c["b1"], "T2", 30.0)
    assert seen["autoreset_off"] and r["reason"].startswith("simulation went unstable") and r["reset_flag"]
    assert r["unstable_s"] == pytest.approx(0.2, abs=1e-3) and r["unstable_phase"] == "close"
    assert r["passed"] is None and r["score"] is None and r["finger_torque_Nm"] == j.torque
    assert r["substeps"] == 2  # sheets run at the fine timestep (test conditions 2.16)
    job = {"block": "t", "design": "r2-09", "reading": 45.0, "job": "mj", "test": "T2", "object": "c_flute_4mm",
           "depth_mm": 30.0, "case": "pinch 30"}
    row = {**job, "kind": "mujoco", **r}
    again = batch1.rerun_job(job, row)
    assert again["b1_over"] == {"substeps": 4} and again["rerun_of_reset"] and again["case"] == job["case"]
    assert batch1.rerun_job(again, {**row, "substeps": 4})["b1_over"] == {"substeps": 8}
    assert batch1.rerun_job(again, {**row, "substeps": 8}) is None
    assert batch1.rerun_job(job, {**row, "reset_flag": False}) is None


def test_batch1_verdict_needs_the_friction_x08_hold(batch1):
    base = {"block": "t", "design": "S2", "reading": 45.0, "kind": "mujoco", "test": "T1", "object": "box", "depth_mm": 30.0,
            "lateral_mm": 0.0}
    rows = [{**base, "fill": "catalog", "case": "nominal", "passed": True},
            {**base, "fill": "catalog", "case": "friction x0.8", "friction_scale": 0.8, "power_s": 1.0, "passed": False},
            {**base, "fill": "loose", "case": "nominal", "passed": True},
            {**base, "test": "T2", "object": "sheet", "case": "pinch 30", "reset_flag": True, "passed": None, "substeps": 2},
            {**base, "test": "T2", "object": "sheet", "case": "pinch 30", "rerun_of_reset": True, "passed": True, "substeps": 4},
            {**base, "test": "T2", "object": "sheet", "case": "friction x0.8", "friction_scale": 0.8, "power_s": 1.0,
             "passed": True}]
    batch1.add_verdicts(rows)
    assert [r.get("verdict") for r in rows] == [False, None, True, None, True, None]
    assert rows[0]["verdict_basis"] == "nominal passes, x0.8 does not hold"
    assert rows[2]["verdict_basis"].startswith("nominal only") and "rerun carries" in rows[3]["verdict_basis"]


def test_batch1_select_runs_part_of_a_block(batch1):
    jobs = batch1.block_refs()
    platform = batch1.select(jobs, ["platform-padded"])
    assert platform and {j["design"] for j in platform} == {"platform-padded"}
    assert len(platform) + len(batch1.select(jobs, ["r2-09"])) == len(jobs)
    one = batch1.select(jobs, ["r2-09"], ["gm_016000163492"])
    assert one and {(j["design"], j["object"]) for j in one} == {("r2-09", "gm_016000163492")}


def test_batch1_rescore_uses_the_first_pass_objects(batch1):
    lib = batch1.fp_library()
    assert len(lib.objects) == 14 and lib.by_id("b_flute_3mm").material.bare_friction == pytest.approx(0.35)
    assert lib.by_id("cheerios_8p9oz").material.bare_friction == pytest.approx(0.3)
    readings = {k: v["reading"] for k, v in batch1.pr2_designs().items()}
    assert readings == {"r2-13 at 90": 90.0, "r2-07 at 90": 90.0, "r2-14 at 45": 45.0, "r2-08 at 45": 45.0}
    assert batch1.get_jaw("platform-padded", 45.0, {"finger_torque": 1.25}).torque == pytest.approx(1.25)
    jobs = batch1.block_pr2()
    assert all(j["library"] == "fp" and j["nobrake"] is False for j in jobs)
    scan = {**next(j for j in jobs if j["job"] == "scan" and j.get("nominal_only")), "simulate": True, "depth_mm": 20.0,
            "table_cap_mm": 200.0, "stop_mm": None}
    assert [f["case"] for f in batch1.t1_followups(scan)] == ["nominal"]


def test_batch1_family_size_loose_runs_its_x08(batch1):
    jobs = batch1.block_fs_loose()
    lengths = {(j["design"], j.get("reach_cap")) for j in jobs}
    assert len(jobs) == len(lengths) == 10 and {("S2", None), ("S2", 170.0), ("S5", 170.0)} <= lengths
    assert all(j["fill"] == "loose" and j["object"] == batch1.FAMILY_SIZE and j["x08_only"] for j in jobs)
    scan = {**jobs[0], "simulate": True, "depth_mm": 30.0, "table_cap_mm": 200.0, "stop_mm": None}
    follow = batch1.t1_followups(scan)
    assert [f["case"] for f in follow] == ["nominal", "friction x0.8"] and follow[1]["fill"] == "loose"


def test_batch1_mount_flag_includes_the_limit(batch1):
    assert batch1.ctx()["b1"]["mount_fatigue_Nm_per_finger"] == pytest.approx(3.9)  # test conditions 2.16
    j = batch1.get_jaw("S3", 45.0)
    assert j.torque == pytest.approx(4.0)
    assert batch1._torque_flag(j) == {"finger_torque_Nm": j.torque, "mount_fatigue_flag": True}
    assert not batch1._torque_flag(batch1.get_jaw("S1", 45.0))["mount_fatigue_flag"]
    assert batch1._torque_flag(batch1.get_jaw("platform-padded", 45.0, {"finger_torque": 3.9}))["mount_fatigue_flag"]
    assert not batch1._torque_flag(batch1.get_jaw("platform-padded", 45.0, {"finger_torque": 3.85}))["mount_fatigue_flag"]
    failed = batch1.run_job({"block": "t", "design": "S3", "reading": 45.0, "job": "mj", "test": "T1", "object": "nope"})
    assert failed["kind"] == "error" and failed["finger_torque_Nm"] == pytest.approx(4.0) and failed["mount_fatigue_flag"]


def test_t5_two_arm_lift_with_the_mid_move_cut(batch1):
    from graspsim2d.t5 import case_object, mirrored, simulate_t5

    c = batch1.ctx()
    j = batch1.get_jaw("S2", 45.0)
    m = mirrored(j)
    assert m.sides[1].body_mj == j.sides[-1].body_mj and {f.side for f in m.sides[1].faces} == {1}
    assert len(m.sides[-1].faces) == len(j.sides[1].faces)
    case = case_object(c["b1"]["t5"], "1ft", c["lib"].materials)
    assert (case.thickness, case.height, case.mass) == (pytest.approx(0.305), pytest.approx(0.254), 5.0)
    assert case.material.allow_pressure == pytest.approx(105e3)
    assert case_object(c["b1"]["t5"], "shipper", c["lib"].materials).material.allow_pressure == pytest.approx(25e3)
    r = simulate_t5(j, case, c["sc"], {**c["b1"], "shake_s": 0.0}, cut_before_stop=True, power_s=0.3)
    assert r["ok"] and r["supported"] and r["power_cut_held"] and r["passed"]
    assert set(r["margins"]) == {"slip", "tilt", "thumb", "power cut"} and r["score"] == min(r["margins"].values())
    for g in "AB":
        # Test conditions, T5: the tips touch at about 33 deg per finger, about 145 mm from each pivot.
        assert r[f"{g}_close"] == "both fingers touch" and 28.0 < r[f"{g}_touch_deg"] < 38.0
        assert 125.0 < r[f"{g}_out_bearing_mm"] < 160.0
    assert r["A_hold_Nm"] == pytest.approx(r["B_hold_Nm"], rel=0.1, abs=0.05)
    assert r["mu_bare"] == pytest.approx(0.35)
    # The brake's reading is its friction loss alone, not the gear coupling's or the stroke limits' forces.
    assert all(0.0 <= r[f"{g}_brake_use_Nm"] <= r["brake_Nm"] + 1e-6 for g in "AB")
    assert r["A_thumb_peak_N"] >= r["A_thumb_N_hold"] > 0.0 and r["A_thumb_peak_at"]
    # The stop along W drives the case outward onto A's hub; the run's peak covers the stop's.
    assert r["A_hub_peak_N"] >= r["A_hub_stop_peak_N"] > 0.0 and r["A_hub_stop_out_N"] > 0.0 and r["A_hub_peak_phase"]
    # Revision 2.21: sustained over the final hold's last 1 s, peak the largest 5 ms average after the stop.
    assert r["hub_peak_from"] == "stop" and 0.0 <= r["A_hub_peak_5ms_after_s"] <= 5.0
    assert r["A_hub_stop_peak_N"] >= r["A_hub_peak_5ms_N"] >= r["A_hub_sustained_N"] > 0.0 and r["A_hub_peak_5ms_out_N"] > 0.0
    assert r["B_hub_sustained_N"] > 0.0


def test_t5_stop_seated_cradle_and_stop_held_runs(batch1):
    from graspsim2d.t5 import case_object, simulate_t5

    c = batch1.ctx()
    j = batch1.get_jaw("S4", 45.0)
    case = case_object(c["b1"]["t5"], "2ft", c["lib"].materials)
    with pytest.raises(ValueError):
        simulate_t5(j, case, c["sc"], c["b1"], approach="pinch")
    r = simulate_t5(j, case, c["sc"], {**c["b1"], "shake_s": 0.0}, approach="seated", hold="stops", stop_g=0.0,
                    power_s=0.2)
    assert r["ok"] and (r["approach"], r["hold"]) == ("seated", "stops")
    # S4's stops are set at 45 deg per finger, never short: half a degree past a stroke read from its -0.5 deg meet.
    assert r["stop_deg_jaw"] == pytest.approx(45.0) and r["preset_deg"] == pytest.approx(45.0)
    assert r["start_out_mm"] > 0.0 and r["t_approach_s"] > 0.1
    for g in "AB":
        assert r[f"{g}_close"] == "seated on its stops" and abs(r[f"{g}_out_off_stop_deg"]) < 0.5
        # The cradle carries the case clear of the hub, and the outside finger's moment goes into its own stop.
        assert r[f"{g}_hub_hold_N"] < 1.0 and r[f"{g}_out_stop_load_hold_Nm"] > 1.0
        assert r[f"{g}_thumb_N_hold"] > 0.0 and r[f"{g}_thumb_contact_mm"] >= 0.0
    # The arm lifts the case onto the cradle before it squeezes, so the case doesn't move on the fingers in the lift.
    assert r["print_slip_mm"] < 1.0
    # With no stop, the hub's peak is read from the power cut.
    assert r["hub_peak_from"] == "power cut" and r["A_hub_sustained_N"] < 1.0

    jobs = batch1.block_t5()
    assert len(jobs) == len({batch1._job_key(x) for x in jobs}) == 312 and all(x["hub_contact"] for x in jobs)
    assert {(x["design"], x["reading"]) for x in jobs if x.get("approach") == "seated"} == {("S4", 45.0), ("S5", 45.0)}
    held = [x for x in jobs if x.get("hold")]
    assert len(held) == 24 and {(x["design"], x["reading"]) for x in held} == {("S3", 45.0), ("S4", 45.0), ("S5", 45.0)}
    tips = batch1.block_t5_tips()
    assert len(tips) == 312 and not any(x["hub_contact"] for x in tips)
    # Until Parth says whether a case may rest on the hub, the tips-only path rates T5.
    assert all(x["rated"] and x["t5_path"] == "tips only" for x in tips)
    assert not any(x["rated"] for x in jobs) and {x["rated"] for x in batch1.block_t5_cut()} == {True, False}


def test_t3_tips_enter_the_gap_and_hold(batch1):
    from graspsim2d.t3 import entry_check, simulate_t3

    c = batch1.ctx()
    box = batch1.get_obj("gm_cheerios_family_18oz", "catalog")
    j = batch1.get_jaw("S2", 45.0)
    g = entry_check(j, box.thickness * 1e3, 1.3, 15.0)
    # SR's toed-in chisels: on Family Size the tips sweep 2.02 - 2.06 mm, a 0.7 - 0.8 mm push at the 1.3 mm gap.
    assert g["ok"] and 2.0 < g["sweep_R_mm"] < 2.1 and 0.7 < g["push_need_R_mm"] < 0.8
    # The entry opening is the least that clears the target over the entry depth, so by SIM's rule (the target-side
    # clearance over the whole depth) it captures no offset; the tip ends alone would take +-0.11 mm.
    assert g["clear_R_mm"] == pytest.approx(0.0, abs=0.01)
    assert g["capture_lo_mm"] == pytest.approx(0.0, abs=0.01) and g["capture_hi_mm"] == pytest.approx(0.0, abs=0.01)
    assert g["capture_land_lo_mm"] < -0.05 and 0.05 < g["capture_land_hi_mm"] < 0.5
    # Opened to centre the tip ends in the gaps, the tips clear the target up the depth and capture an offset either way.
    assert g["centred_opening_deg"] > g["entry_opening_deg"]
    assert g["centred_land_target_R_mm"] == pytest.approx(g["centred_land_neighbour_R_mm"], abs=0.01)
    assert g["centred_clear_R_mm"] > 0.0 and g["centred_capture_lo_mm"] < 0.0 < g["centred_capture_hi_mm"]
    # A tip 3 mm off, or at a closed gap, lands on a box top: the entry fails without a MuJoCo run.
    for kw in ({"lateral": 3.0}, {"gap": 0.0}):
        r = simulate_t3(j, box, c["sc"], c["b1"], **kw)
        assert r["landed"] and not r["passed"] and r["limiting"] == "entry" and "elements" not in r
        assert r["cam_radius_R_mm"] > 1.0 and r["cam_radius_mm"] >= r["cam_radius_R_mm"]
        assert not r["enters_at_sensitivity_radius"]
    # The shoulder 15.5 mm up S2's tips is a depth stop over the target's top: the 20 mm sensitivity reports it.
    r = simulate_t3(j, box, c["sc"], c["b1"], entry=20.0)
    assert r["passed"] is None and r["limiting"] == "depth stop" and 15.0 < r["depth_stop_mm"] < 16.0
    assert r["reason"].startswith("depth stop at 15.") and "elements" not in r
    assert math.isnan(g["depth_stop_R_mm"])
    r = simulate_t3(j, box, c["sc"], c["b1"], friction_scale=0.8, drag_scale=1.2, power_s=0.2)
    assert r["ok"] and not r["reset_flag"] and not r["stalled"] and not r["target_contact_in_entry"]
    assert r["entry_reached_mm"] == pytest.approx(15.0, abs=0.05)
    # The neighbours tip out of the way: the push matches the geometry, under 5 N on the 5 ms average.
    for s in "RL":
        assert r[f"push_{s}_mm"] == pytest.approx(g[f"push_need_{s}_mm"], abs=0.15)
        assert r[f"push_{s}_N"] < 5.0 and r[f"drag_{s}_mm"] < 0.5
        assert 0.0 < r[f"push_{s}_N_10ms"] <= r[f"push_{s}_N"] + 1e-9 <= r[f"push_{s}_peak_N"] + 1e-9
    assert r["edge_sensitive"] == (max(r["push_R_N"], r["push_L_N"]) > 4.0)
    assert r["passed"] and r["pass_rule"].startswith("entry depth, then push") and r["slip_mm"] < 0.5
    assert r["band_N"] > 10.0 and r["power_cut_held"] and 0.0 <= r["brake_use_Nm"] <= 30.0

    jobs = batch1.block_t3()
    assert len(jobs) == len({batch1._job_key(x) for x in jobs})
    assert {x["case"] for x in jobs if x["design"] == "S2"} == {
        "nominal", "friction x0.8", "corner +3 mm", "gap 0 mm", "gap 3 mm", "entry 20 mm", "case wall L"}
    assert {"corner -3 mm", "case wall R"} <= {x["case"] for x in jobs if x["design"] == "S4"}
    # Revision 2.23: the corners and the 20 mm entry are reported, not rated.
    assert {x["case"] for x in jobs if x["design"] == "S4" and not x["rated"]} == {
        "corner +3 mm", "corner -3 mm", "entry 20 mm"}
    row = batch1.run_job({**next(x for x in jobs if x["design"] == "S2" and x["case"] == "gap 0 mm")})
    assert row["kind"] == "mujoco" and row["landed"] and row["gap_mm"] == 0.0 and row["passed"] is False
    rows = [row, {**row, "case": "nominal", "passed": True}, {**row, "case": "corner +3 mm", "rated": False}]
    batch1.add_verdicts(rows)
    assert rows[1]["verdict"] is False and rows[1]["verdict_basis"] == "rated rows fail: gap 0 mm"


def test_batch1_t5_cut_block_runs_revision_2_18(batch1):
    jobs = batch1.block_t5_cut()
    assert len(jobs) == len({batch1._job_key(j) for j in jobs}) == 40
    assert {j["design"] for j in jobs} == {"S1", "S2", "S3", "S6"}
    assert all(j["job"] == "t5" and j["cut_before_stop"] and j["object"] == "2ft" and j["reading"] == 45.0 for j in jobs)
    assert {j["mass_kg"] for j in jobs if j["design"] == "S2"} == {12.8}
    assert {(j["friction_scale"], j["bare_mu"]) for j in jobs} == {(1.0, None), (0.8, None), (1.0, 0.24)}
    assert {j["hub_contact"] for j in jobs} == {True, False}
