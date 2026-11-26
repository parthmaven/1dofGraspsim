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
    d = concepts["straight_flat"].make(space)
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
    d = concepts["straight_flat"].make(space)
    r = solve_grip(d, lib.by_id("mini_1oz"), sc)
    assert r.ok
    # drive torque = normal force x lever arm about the pivot
    assert r.info["normal_force_N"] * r.info["arm_mm"] / 1000 == pytest.approx(d.finger_torque, rel=1e-6)


def test_diverging_box_and_converging_sheet(setup):
    space, concepts, lib, sc = setup
    d = concepts["straight_flat"].make(space)
    box = solve_grip(d, lib.by_id("cheerios_8p9oz"), sc)
    sheet = solve_grip(d, lib.by_id("chipboard_1p5mm"), sc)
    assert box.ok and sheet.ok
    assert box.info["phi_deg"] > 5.0
    assert sheet.info["phi_deg"] < 0.0


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
