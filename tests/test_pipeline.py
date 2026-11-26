import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from graspsim2d.analytic import band_min_x, solve_grip
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


def test_force_balance_of_the_analytic_grip(setup):
    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    r = solve_grip(d, lib.by_id("mini_1oz"), sc)
    assert r.ok
    # drive torque = normal force x lever arm about the pivot
    assert r.info["normal_force_N"] * r.info["arm_mm"] / 1000 == pytest.approx(d.finger_torque, rel=1e-6)


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


def test_fingers_are_geared_one_to_one(setup):
    import mujoco

    from graspsim2d.mjsim import build_model

    space, concepts, lib, sc = setup
    d = concepts["baseline_paddle"].make(space)
    xml, meta = build_model(d, lib.by_id("chipboard_1p5mm"), sc, 0.03)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("jR", "jL")}
    for n in adr.values():
        data.qpos[n] = meta["phi0"]
    for _ in range(1500):  # drive the right finger only; the gear must carry the left one
        data.ctrl[2] = -d.finger_torque
        mujoco.mj_step(model, data)
    jR, jL = data.qpos[adr["jR"]], data.qpos[adr["jL"]]
    assert meta["phi0"] - jR > math.radians(2.0)
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
