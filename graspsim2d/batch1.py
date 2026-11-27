"""Batch 1 of the concept review: each design's jaw, the screen and MuJoCo over the batch-1 tests in a
process pool, and the result rows (the review team's batch-1 plan, "Run order").

Run it from a snapshot of the repo (a git worktree), so commits on the branch can't change a run in
flight, and stop it with `pkill -9 -f graspsim2d.batch1`, never Ctrl-C:

    python -m graspsim2d.batch1 refs --out runs/batch1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
B1 = CFG / "batch1"
SPEC_FILES = {"P0": "p0-platform-step", "S1": "s1-platform-plus", "S2": "s2-deep-reach", "S3": "s3-top-edge-clamp",
              "S4": "s4-asymmetric-pair", "S5": "s5-l-cradle-crank", "S6": "s6-monolithic-crowned-face"}
FILLS = ("catalog", "loose", "settled")
PINCHES = (20.0, 30.0, 50.0)
BLANKS = ("chipboard_0p5mm", "chipboard_0p5mm_short")
# The regression cases (results check, line 2): PR #2's own numbers at f510113 for the first pass's
# r2-09 and the 45 deg re-run's padded platform, on Cheerios Family Size and the unchanged flutes.
REGRESSION = [("r2-09", 35.0, "gm_cheerios_family_18oz", 37.348), ("platform-padded", 45.0, "gm_cheerios_family_18oz", 74.3)]
REGRESSION += [(d, r, o, 30.0) for d, r in (("r2-09", 35.0), ("platform-padded", 45.0))
               for o in ("e_flute_1p6mm", "b_flute_3mm", "c_flute_4mm", "bc_double_6p8mm")]

_CTX: dict[str, Any] = {}


def commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


_FINGER: list[str | None] = [None]  # the finger material in use; None takes the config's `finger_material`


def finger_values(d: Any, finger: str) -> Any:
    """A config tree as fingers of `finger`'s material see it: on the PA12 baseline each `key_pa12` value replaces
    `key`'s; the suffixed keys drop out either way."""
    if isinstance(d, list):
        return [finger_values(x, finger) for x in d]
    if not isinstance(d, dict):
        return d
    out = {k: finger_values(v, finger) for k, v in d.items() if not (isinstance(k, str) and k.endswith("_pa12"))}
    if finger == "pa12":
        out.update({k[:-5]: finger_values(v, finger) for k, v in d.items() if isinstance(k, str) and k.endswith("_pa12")})
    return out


def _shared() -> dict[str, Any]:
    if not _CTX:
        from .jaw import load_marks
        from .objects import load_objects
        from .scenario import load_scenario
        sc = load_scenario(B1 / "scenario.yaml")
        _CTX.update(sc=sc, lib_raw=load_objects(B1 / "objects.yaml"), lay=yaml.safe_load((B1 / "layouts.yaml").read_text()),
                    marks=load_marks(B1 / "marked_faces.yaml"), jaws={}, by_finger={})
    return _CTX


def finger() -> str:
    return _FINGER[0] or str(_shared()["sc"].raw["batch1"].get("finger_material", "pa12"))


@contextlib.contextmanager
def finger_material(name: str | None):
    """Run what's inside on fingers of `name`'s material (None keeps the one in use)."""
    old = _FINGER[0]
    if name is not None:
        known = _shared()["sc"].raw["batch1"].get("finger_materials") or {}
        if name not in known:
            raise ValueError(f"unknown finger material {name!r}; have {sorted(known)}")
        _FINGER[0] = name
    try:
        yield
    finally:
        _FINGER[0] = old


def ctx(material: str | None = None) -> dict[str, Any]:
    """The batch-1 context for one finger material: the scenario, layouts, marks and jaws are shared, while `b1` and
    the object library carry that material's bare friction and density."""
    base = _shared()
    f = material or finger()
    per = base["by_finger"]
    if f not in per:
        from .objects import with_finger
        b1 = finger_values(base["sc"].raw["batch1"], f)
        b1["finger_material"] = f
        per[f] = {"sc": base["sc"], "lay": base["lay"], "marks": base["marks"], "jaws": base["jaws"], "b1": b1,
                  "lib": with_finger(base["lib_raw"], f), "finger": f}
    return per[f]


def fp_library():
    """The first pass's 14 objects (config/objects.yaml) with batch 1's bare friction table, for the finger in use."""
    c = ctx()
    if "lib_fp" not in c:
        from dataclasses import replace
        from .objects import load_objects
        mats = c["lib"].materials
        lib = load_objects(CFG / "objects.yaml")
        lib.objects = [replace(o, material=replace(o.material, bare_friction=mats[o.material.name].bare_friction,
                                                   printed_friction=mats[o.material.name].printed_friction))
                       if o.material.name in mats else o for o in lib.objects]
        c["lib_fp"] = lib
    return c["lib_fp"]


def pr2_designs() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in json.loads((B1 / "pr2-designs.json").read_text()).items() if not k.startswith("_")}


def platform_padded():
    """The padded platform stand-in: concept baseline_paddle's nominal (PR #2's baseline as drawn)."""
    from .schema import load_concepts, load_design_space
    space = load_design_space(CFG / "design_space_stroke45.yaml")
    c = next(c for c in load_concepts(CFG / "concepts.yaml", space) if c.name == "baseline_paddle")
    return c.make(space, label="platform finger, padded stand-in")


def get_jaw(design: str, reading: float, values: dict[str, float] | None = None, reach_cap: float | None = None):
    c = ctx()
    key = (design, reading, json.dumps(values or {}, sort_keys=True), reach_cap)
    if key in c["jaws"]:
        return c["jaws"][key]
    from dataclasses import replace
    from .jaw import jaw_from_design, jaw_from_spec
    from .schema import GripperDesign
    parametric = None
    if design == "r2-09":
        parametric = GripperDesign.from_dict(json.loads((B1 / "r2-09.json").read_text()))
    elif design == "platform-padded":
        parametric = platform_padded()
    elif design.startswith("pr2 "):
        parametric = GripperDesign.from_dict(pr2_designs()[design[4:]]["design"])
    if parametric is not None:
        j = jaw_from_design(replace(parametric, **(values or {})), c["sc"], reading=reading, concept=design)
    else:
        from .outline import build_gripper
        from .specs import SpecDesign, load_spec
        sp = load_spec(B1 / "specs" / f"{SPEC_FILES[design]}.yaml", reach_cap=reach_cap)
        dd = SpecDesign.make(sp, values or {}, reading, f"{design} " + (", ".join(f"{k} {v:g}" for k, v in (values or {}).items()) or "nominal"))
        j = jaw_from_spec(build_gripper(sp, dd, c["sc"].keep_out, c["sc"].closed_limit, c["lay"]), c["marks"])
    c["jaws"][key] = j
    return j


def get_obj(oid: str, fill: str = "catalog", library: str = "b1"):
    from .objects import with_fill
    c = ctx()
    o = (fp_library() if library == "fp" else c["lib"]).by_id(oid)
    if o.kind == "box":
        o = with_fill(o, float(c["b1"]["fills"][fill]), fill)
    return o


def _flat(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 5)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


# ---------------------------------------------------------------- jobs (run in the pool's workers)

def job_scan(job: dict[str, Any]) -> dict[str, Any]:
    """T1 screen: the full insertion scan on one box at one fill."""
    from .screen1 import Limits, insertion_cap, scan_box, TABLE_CLEARANCE
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    o = get_obj(job["object"], job["fill"], job.get("library", "b1"))
    lim = Limits.from_config(c["b1"], "T1")
    r = scan_box(j, o, c["sc"], lim)
    row = {**job, "kind": "screen", "test": "T1", **{k: v for k, v in r.row().items() if k != "object"}}
    row["table_cap_mm"] = o.height * 1e3 - 2 * TABLE_CLEARANCE
    g = r.nominal
    row["status"] = g.status if g is not None else "nogrip"
    row["simulate"] = bool(g is not None and g.status != "nogrip" and math.isfinite(r.depth) and (g.ok or g.status == "mujoco"))
    if not j.symmetric:
        row["scan_basis"] = "each finger as a mirrored pair, the worse at each depth; the depth only, scored in MuJoCo"
    if not row["simulate"]:
        row["not_simulated"] = (g.reason if g is not None else "") or r.ended
    return {**row, **_torque_flag(j)}


def _torque_flag(j) -> dict[str, Any]:
    lim = float(ctx()["b1"].get("mount_fatigue_Nm_per_finger", 4.0))
    return {"finger_torque_Nm": j.torque, "mount_fatigue_flag": bool(j.torque >= lim - 1e-9)}


def job_sheet_screen(job: dict[str, Any]) -> dict[str, Any]:
    from .screen1 import Limits, solve_sheet
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    o = get_obj(job["object"], library=job.get("library", "b1"))
    g = solve_sheet(j, o, c["sc"], Limits.from_config(c["b1"], "T2"), job["depth_mm"])
    return {**job, "kind": "screen", "test": "T2", "status": g.status, "ok": g.ok, "score": g.score, "limiting": g.limiting,
            "reason": g.reason, **_flat("m_", g.margins), **_flat("d_", g.demand),
            **{k: v for k, v in g.info.items() if not isinstance(v, (list, dict))}, **_torque_flag(j)}


def job_mj(job: dict[str, Any]) -> dict[str, Any]:
    """One MuJoCo run."""
    from . import mj1
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    o = get_obj(job["object"], job.get("fill", "catalog"), job.get("library", "b1"))
    b1 = {**c["b1"], **job.get("b1_over", {})}
    t0 = time.time()
    hz = job.get("shake_hz")
    r = mj1.simulate(j, o, c["sc"], b1, job["test"], job["depth_mm"], lateral=job.get("lateral_mm", 0.0),
                     shake_hz=(hz, c["sc"].shake_vertical_hz) if hz else None, friction_scale=job.get("friction_scale", 1.0),
                     brake=job.get("brake", 30.0), band_factor=job.get("band_factor"), stops=job.get("stops"),
                     power_s=job.get("power_s"))
    out = {**job, "kind": "mujoco", "seconds": round(time.time() - t0, 1)}
    for k, v in r.items():
        if k == "margins":
            out.update(_flat("m_", v))
        elif k == "demand":
            out.update(_flat("d_", v))
        elif k == "trace":
            continue
        elif k in ("depth_mm", "lateral_mm", "task"):
            continue
        else:
            out[k] = v
    out["held_levels_g"] = [s["level_g"] for s in r.get("stops", []) if s.get("held")]
    return out


def job_t5(job: dict[str, Any]) -> dict[str, Any]:
    """One T5 two-arm lift. A job's `object` is a case key of the scenario's t5 block. With `dump_dir`, the run's
    close record goes to a gzipped JSON file there, and the row's `close_dump` names it. `close_only` stops the run
    at the end of the squeeze and keeps only its close. `approach_m_s` and `close_rad_s` name a design's own lower
    contact speeds (SIM round 18 b): the arms' approach and the jaw's close. `close_torque_Nm` names its close torque
    per finger, and `close_gear_friction` puts the gear's friction on the close (SIM round 20)."""
    from .t5 import case_object, simulate_t5
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    b1 = {**c["b1"], **job.get("b1_over", {})}
    close = {k: float(job[k]) for k in ("approach_m_s", "close_rad_s", "close_torque_Nm") if job.get(k) is not None}
    if job.get("close_gear_friction"):
        close["close_gear_friction"] = True
    if close:
        b1 = {**b1, "t5": {**b1["t5"], **close}}
    case = case_object(b1["t5"], job["object"], c["lib"].materials, job.get("mass_kg"))
    t0 = time.time()
    r = simulate_t5(j, case, c["sc"], b1, com_offset=job.get("com_offset", 0.0), squeeze=job.get("squeeze_N"),
                    friction_scale=job.get("friction_scale", 1.0), bare_mu=job.get("bare_mu"),
                    brake=job.get("brake", 30.0), stop_g=job.get("stop_g"), cut_before_stop=job.get("cut_before_stop", False),
                    power_s=job.get("power_s"), shake_g=job.get("shake_g"), hub_contact=job.get("hub_contact", True),
                    approach=job.get("approach", "close"), hold=job.get("hold", "servo"),
                    close_dump=bool(job.get("dump_dir")), close_only=bool(job.get("close_only")))
    out = {**job, "kind": "mujoco", "seconds": round(time.time() - t0, 1)}
    dump = r.pop("close_dump", None)
    for k, v in r.items():
        if k == "margins":
            out.update(_flat("m_", v))
        elif k != "trace":
            out[k] = v
    if dump is not None:
        out["close_dump"] = write_close_dump(Path(job["dump_dir"]), job, out, dump)
    return {**out, **_torque_flag(j)}


def write_close_dump(folder: Path, job: dict[str, Any], row: dict[str, Any], dump: dict[str, Any]) -> str:
    """A T5 run's close record, named by its design, case and path and a hash of the whole job."""
    key = {k: v for k, v in job.items() if k != "dump_dir"}
    tag = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]
    sg = row.get("stop_g")
    name = "_".join(str(p) for p in (
        job.get("design"), job.get("object"), f"{row.get('mass_kg') or 0:g}kg", "com" if job.get("com_offset") else "",
        job.get("t5_path") or ("hub" if job.get("hub_contact", True) else "tips"),
        "cut" if job.get("cut_before_stop") else "", "close" if job.get("close_only") else (f"{sg:g}g" if sg is not None else ""),
        tag) if p != "")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json.gz"
    keep = ("ok", "reason", "case_label", "mass_kg", "com_offset_W", "hub_contact", "approach", "hold", "stop_g",
            "cut_before_stop", "close_only", "finger_material", "mu_bare", "mu_hub", "t_first_contact_s", "t_approach_s",
            "t_close_s", "t_squeeze_end_s", "A_close", "B_close",
            "A_first_touch_deg", "B_first_touch_deg", "A_touch_deg", "B_touch_deg", "substeps", "sample_s",
            "deck_off_before_squeeze", "reset_flag")
    with gzip.open(path, "wt") as fh:
        json.dump({"job": key, "commit": commit(), "row": {k: row.get(k) for k in keep}, **dump}, fh,
                  separators=(",", ":"), default=str)
    return str(path)


def job_t3(job: dict[str, Any]) -> dict[str, Any]:
    """One T3 run, tips between boxes. A job's `depth_mm` is the entry depth below the box tops."""
    from .t3 import simulate_t3
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    b1 = {**c["b1"], **job.get("b1_over", {})}
    o = get_obj(job["object"], job.get("fill", "catalog"), job.get("library", "b1"))
    t0 = time.time()
    r = simulate_t3(j, o, c["sc"], b1, gap=job.get("gap_mm"), entry=job.get("depth_mm"), lateral=job.get("lateral_mm", 0.0),
                    friction_scale=job.get("friction_scale", 1.0), drag_scale=job.get("drag_scale", 1.0),
                    wall_side=job.get("wall_side", 0), brake=job.get("brake", 30.0), power_s=job.get("power_s"))
    out = {**job, "kind": "mujoco", "seconds": round(time.time() - t0, 1)}
    for k, v in r.items():
        if k == "margins":
            out.update(_flat("m_", v))
        elif k != "trace" and out.get(k) is None:
            out[k] = v
    return {**out, **_torque_flag(j)}


def edge_note(material: str | None = None) -> dict[str, str]:
    """Revision 2.39: the material's edge-break note, which every row on it carries."""
    mats = _shared()["sc"].raw["batch1"].get("finger_materials") or {}
    note = (mats.get(material or finger()) or {}).get("edge_note")
    return {"edge_note": " ".join(str(note).split())} if note else {}


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """One job, on the finger material its `finger` names (the config's selection without one)."""
    try:
        f = {"scan": job_scan, "sheet_screen": job_sheet_screen, "mj": job_mj, "t5": job_t5, "t3": job_t3}[job["job"]]
        with finger_material(job.get("finger")):
            return _clean({**f(job), "finger_material": finger(), **edge_note()})
    except Exception as e:  # noqa: BLE001 - a failed job is a row, not a dead batch
        import traceback
        row = {**job, "kind": "error", "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:],
               "finger_material": job.get("finger") or finger(), **edge_note(job.get("finger"))}
        try:
            row.update(_torque_flag(get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))))
        except Exception:  # noqa: BLE001 - the jaw itself may be what failed
            row.update(finger_torque_Nm=math.nan, mount_fatigue_flag=False)
        return row


# ---------------------------------------------------------------- blocks

def _base(block: str, design: str, reading: float, **kw: Any) -> dict[str, Any]:
    return {"block": block, "design": design, "reading": reading, **kw}


def t1_jobs(block: str, design: str, reading: float, boxes: Iterable[str], sens: list | None = None,
            fills: Iterable[str] = FILLS, **kw: Any) -> list[dict[str, Any]]:
    """`sens`: extra MuJoCo cases at the nominal depth on the catalog fill, as (case, overrides).
    `nominal_only=True` in `kw` drops the corners and x0.8; `nobrake=False` the no-brake case."""
    base = _base(block, design, reading, **kw)
    if sens:
        base["sens"] = sens
    return [{**base, "job": "scan", "object": b, "fill": f} for b in boxes for f in fills]


def t1_followups(scan: dict[str, Any], symmetric: bool = True) -> list[dict[str, Any]]:
    """MuJoCo at the screen's depth, the two +-3 mm corners (deeper and shallower, offset sideways) and,
    at catalog fill, friction x0.8."""
    if not scan.get("simulate"):
        return []
    keep = {k: scan[k] for k in ("block", "design", "reading", "object", "fill") if k in scan}
    keep.update({k: scan[k] for k in ("values", "reach_cap", "library", "stops", "power_s") if scan.get(k) is not None})
    c = ctx()
    place = float(c["b1"].get("placement_mm", 3.0))
    d = float(scan["depth_mm"])
    table = float(scan["table_cap_mm"])
    deep = d + place
    if scan.get("stop_mm") is not None and math.isfinite(scan["stop_mm"] or math.nan):
        deep = min(deep, float(scan["stop_mm"]))
    deep = min(deep, table)
    out = [{**keep, "job": "mj", "test": "T1", "case": "nominal", "depth_mm": d, "lateral_mm": 0.0}]
    x08 = {**keep, "job": "mj", "test": "T1", "case": "friction x0.8", "depth_mm": d, "lateral_mm": 0.0,
           "friction_scale": float(c["b1"].get("robust_friction", 0.8)), "power_s": 1.0}
    if scan.get("nominal_only"):
        return out
    if scan.get("x08_only"):
        return out + [x08]
    sides = (place,) if symmetric else (place, -place)
    for s in sides:
        out.append({**keep, "job": "mj", "test": "T1", "case": f"corner deep{'+' if s > 0 else '-'}", "depth_mm": deep,
                    "lateral_mm": s, "power_s": 1.0})
        out.append({**keep, "job": "mj", "test": "T1", "case": f"corner shallow{'+' if s > 0 else '-'}",
                    "depth_mm": max(d - place, place), "lateral_mm": s, "power_s": 1.0})
    if scan["fill"] == "catalog":
        out.append(x08)
        for case, over in scan.get("sens") or []:
            out.append({**keep, "job": "mj", "test": "T1", "case": case, "depth_mm": d, "lateral_mm": 0.0, "power_s": 1.0,
                        **over})
    return out


def t2_jobs(block: str, design: str, reading: float, sheets: Iterable[str], pinches: Iterable[float] = PINCHES,
            extras: bool = True, **kw: Any) -> list[dict[str, Any]]:
    """`extras`: friction x0.8 at 30 mm and, for batch 1's carton blank, the shake sweep."""
    c = ctx()
    hz_sweep = [float(h) for h in c["b1"].get("sheet_shake_hz", [1.5, 2.0, 3.0])]
    base = _base(block, design, reading, **kw)
    out = []
    for s in sheets:
        for p in pinches:
            out.append({**base, "job": "sheet_screen", "object": s, "depth_mm": p, "case": f"pinch {p:g}"})
            out.append({**base, "job": "mj", "test": "T2", "object": s, "depth_mm": p, "case": f"pinch {p:g}"})
        if not extras:
            continue
        out.append({**base, "job": "mj", "test": "T2", "object": s, "depth_mm": 30.0, "case": "friction x0.8",
                    "friction_scale": float(c["b1"].get("robust_friction", 0.8)), "power_s": 1.0})
        if s in BLANKS and kw.get("library", "b1") == "b1":
            for hz in hz_sweep:
                if abs(hz - c["sc"].shake_lateral_hz) > 1e-6:
                    out.append({**base, "job": "mj", "test": "T2", "object": s, "depth_mm": 30.0, "case": f"shake {hz:g} Hz",
                                "shake_hz": hz, "power_s": 1.0})
    return out


def regression_jobs(block: str) -> list[dict[str, Any]]:
    out = []
    for design, reading, oid, depth in REGRESSION:
        test = "T1" if oid.startswith("gm_") else "T2"
        base = {"block": block, "design": design, "reading": reading, "object": oid, "fill": "catalog", "test": test,
                "depth_mm": depth, "job": "mj", "stops": [], "power_s": 0.5}
        out.append({**base, "case": "regression, PR #2 settings", "band_factor": 1.0,
                    "b1_over": {"fine_pitch_mm": 4.0}})
        out.append({**base, "case": "regression, batch-1 settings"})
    return out


def block_refs() -> list[dict[str, Any]]:
    """Block 1: r2-09 and the padded platform stand-in, at the 45 deg reading, on the boxes and sheets."""
    c = ctx()
    boxes = [o.id for o in c["lib"].objects if o.kind == "box"]
    sheets = [o.id for o in c["lib"].objects if o.kind == "sheet"]
    jobs = []
    for d in ("r2-09", "platform-padded"):
        jobs += t1_jobs("refs", d, 45.0, boxes)
        jobs += t2_jobs("refs", d, 45.0, sheets)
    return jobs + regression_jobs("refs")


def _boxes_sheets() -> tuple[list[str], list[str]]:
    lib = ctx()["lib"]
    return [o.id for o in lib.objects if o.kind == "box"], [o.id for o in lib.objects if o.kind == "sheet"]


def block_s5() -> list[dict[str, Any]]:
    """Block 3: S5 at 160 and 170 mm (the flagged run), boxes, at the 45 deg reading."""
    boxes, _ = _boxes_sheets()
    return (t1_jobs("s5", "S5", 45.0, boxes)
            + t1_jobs("s5", "S5", 45.0, boxes, values={"finger_reach": 170.0}, reach_cap=170.0))


def block_s2() -> list[dict[str, Any]]:
    """Block 2: S2 at 160 and 170 mm (the flagged run), boxes, and its sheets, at the 45 deg reading."""
    boxes, sheets = _boxes_sheets()
    return (t1_jobs("s2", "S2", 45.0, boxes)
            + t1_jobs("s2", "S2", 45.0, boxes, values={"finger_reach": 170.0}, reach_cap=170.0)
            + t2_jobs("s2", "S2", 45.0, sheets))


def block_s4() -> list[dict[str, Any]]:
    """Block C: S4's pinches at the 45 deg reading (plan item 8), boxes and its sheets. Its jaw is asymmetric, so the
    boxes' placement corners run on both sides."""
    boxes, sheets = _boxes_sheets()
    return t1_jobs("s4", "S4", 45.0, boxes) + t2_jobs("s4", "S4", 45.0, sheets)


def block_s3_sheets_platform() -> list[dict[str, Any]]:
    """Block 4: S3 at the 15 N top band with its sensitivities, the sheets for S1, S3 and S5, and the
    platform as built."""
    boxes, sheets = _boxes_sheets()
    sens = [("band x1 stiffness", {"band_factor": 1.0})]
    jobs = t1_jobs("s3p0", "S3", 45.0, boxes, sens=sens)
    for d in ("S1", "S3", "S5"):
        jobs += t2_jobs("s3p0", d, 45.0, sheets)
    jobs += t1_jobs("s3p0", "P0", 45.0, boxes) + t2_jobs("s3p0", "P0", 45.0, sheets)
    return jobs


def block_s1_s6() -> list[dict[str, Any]]:
    """S1 and S6 at nominal on the boxes, and S6's sheets: the T1 counts beside VAL's hand numbers."""
    boxes, sheets = _boxes_sheets()
    return t1_jobs("s1s6", "S1", 45.0, boxes) + t1_jobs("s1s6", "S6", 45.0, boxes) + t2_jobs("s1s6", "S6", 45.0, sheets)


TORQUE_BAND = tuple(0.25 * k for k in range(1, 25))


def block_pr2() -> list[dict[str, Any]]:
    """PR #2's four 13-of-14 designs and the platform finger at its 5 N m, rescored with batch-1 scoring on the
    first pass's 14 objects; then the platform with its torque set per object (0.25 - 6 N m per finger)."""
    lib = fp_library()
    boxes = [o.id for o in lib.objects if o.kind == "box"]
    sheets = [o.id for o in lib.objects if o.kind == "sheet"]
    kw = {"library": "fp", "nobrake": False, "stops": [], "power_s": 0.5}
    designs = [(f"pr2 {k}", float(v["reading"])) for k, v in pr2_designs().items()] + [("platform-padded", 45.0)]
    jobs = []
    for d, reading in designs:
        jobs += t1_jobs("pr2", d, reading, boxes, fills=("catalog",), **kw)
        jobs += t2_jobs("pr2", d, reading, sheets, **kw)
    nominal = platform_padded().finger_torque
    for tau in TORQUE_BAND:
        if abs(tau - nominal) < 1e-9:
            continue
        v = {"finger_torque": tau}
        jobs += t1_jobs("pr2", "platform-padded", 45.0, boxes, fills=("catalog",), values=v, nominal_only=True, **kw)
        jobs += t2_jobs("pr2", "platform-padded", 45.0, sheets, pinches=(30.0,), extras=False, values=v, **kw)
    return jobs


def block_pr2_depths() -> list[dict[str, Any]]:
    """PR #2's four 13-of-14 designs in MuJoCo at PR #2's own insertion depth on each box, scored by batch 1:
    separates the batch-1 scan's depth choice from its limits."""
    lib = fp_library()
    boxes = [o.id for o in lib.objects if o.kind == "box"]
    jobs = []
    for k, v in pr2_designs().items():
        for b in boxes:
            jobs.append({"block": "pr2depth", "design": f"pr2 {k}", "reading": float(v["reading"]), "job": "mj", "test": "T1",
                         "object": b, "fill": "catalog", "library": "fp", "case": "PR #2 depth",
                         "depth_mm": float(v["pr2"]["per_object"][b]["insertion_mm"]), "lateral_mm": 0.0, "stops": [],
                         "power_s": 0.5, "nobrake": False})
    return jobs


FAMILY_SIZE = "gm_cheerios_family_18oz"


def block_fs_loose() -> list[dict[str, Any]]:
    """Family Size at loose fill, the one loose-fill box the test conditions' Fill row names, with its friction
    x0.8 run, for every T1 design and length on its own block's settings."""
    jobs = []
    for block in (block_refs, block_s2, block_s5, block_s3_sheets_platform, block_s1_s6):
        jobs += [{**j, "block": "fsloose", "x08_only": True} for j in block()
                 if j["job"] == "scan" and j["object"] == FAMILY_SIZE and j["fill"] == "loose"]
    return jobs


def block_t5_cut() -> list[dict[str, Any]]:
    """Revision 2.18: S1, S2, S3 and S6, which hold on the brake, cut power mid-move and then make T5's 1 g stop
    along W, at each one's rated mass on the 2 ft case. The bare printed finger on corrugated runs at 0.35, 0.28
    (the x0.8 run) and 0.24 (a sensitivity), with the hub carrying the corner and, as the hand checks have it,
    released once the jaw closes; the 10 % centre-of-mass offset runs at 0.35 and 0.28."""
    cfg = ctx()["b1"]["t5"]["midmove_cut"]
    mus = ((1.0, None, "mu 0.35"), (0.8, None, "x0.8, mu 0.28"),
           (1.0, float(cfg["bare_mu_sensitivity"]), f"mu {float(cfg['bare_mu_sensitivity']):g}"))
    jobs = []
    for d in cfg["designs"]:
        for fs, mu, tag in mus:
            for hub in (True, False):
                for com in (0.0, float(ctx()["b1"]["t5"]["com_offset_W"])):
                    if com and mu is not None:
                        continue
                    label = f"mid-move cut, {tag}, {'hub' if hub else 'tips only'}" + (f", COM +{com:.0%} W" if com else "")
                    jobs.append(_base("t5cut", d, 45.0, job="t5", test="T5", object=cfg["case"],
                                      mass_kg=float(cfg["rated_kg"][d]), cut_before_stop=True, friction_scale=fs,
                                      bare_mu=mu, hub_contact=hub, com_offset=com, case=label, **_t5_path(hub)))
    return jobs


def _t5_path(hub: bool) -> dict[str, Any]:
    """Until Parth says whether a case may rest on the hub, T5 is rated on the tips-only path and the hub path is
    reported alongside."""
    return {"t5_path": "hub" if hub else "tips only", "rated": not hub}


def t5_jobs(hub: bool) -> list[dict[str, Any]]:
    """T5 for block 5, with the hub as a contact body (`hub`) or released once the jaw closes (the hand checks'
    tips-only path, which rates T5). Per design and reading:
    - the hold test, 30 N squeeze: the five cases at their masses and the two heavy sensitivities, centred and at
      the 60/40 share, with the 1 g stop at 23 kg;
    - the strength check: every size at the 23 kg ceiling with 0.5 m g of squeeze, at both shares;
    - the no-brake power cut, about 1 N m of gear friction, once; the 0.5 g shake as a sensitivity, once;
    - for a fixed L-angle stop, at 45 deg: the stop-held pair, each finger on its own stop with the gear released
      and the outside finger's stop removed, on the 2 ft and 3 ft cases at both shares.
    S4, and S5 at 45 deg, take revision 2.13's stop-seated approach; everything else closes until both fingers touch."""
    b1 = ctx()["b1"]
    cfg = b1["t5"]
    blk = cfg["block"]
    ceiling = float(cfg["ceiling_kg"])
    off = float(cfg["com_offset_W"])
    sq = float(cfg["strength_squeeze_mg"])
    cases = [(k, float(c["mass_kg"])) for k, c in cfg["cases"].items()]
    cases += [(k, float(m)) for k, m in cfg["sensitivities"]]
    sizes = [k for k in cfg["cases"] if k != "shipper"]
    tag = "hub" if hub else "tips only"
    name = "t5" if hub else "t5tips"
    jobs = []

    def add(d, reading, label, **kw):
        jobs.append(_base(name, d, reading, job="t5", test="T5", hub_contact=hub, case=f"{label}, {tag}", **_t5_path(hub),
                          **kw))

    for d, readings in blk["designs"].items():
        for reading in readings:
            reading = float(reading)
            seated = reading in [float(r) for r in blk["stop_seated"].get(d, [])]
            ap = {"approach": "seated"} if seated else {}
            for com in (0.0, off):
                share = f", COM +{com:.0%} W" if com else ""
                for k, m in cases:
                    add(d, reading, f"hold, {m:g} kg{share}", object=k, mass_kg=m, com_offset=com,
                        stop_g=1.0 if m >= ceiling - 1e-9 else 0.0, **ap)
                for k in sizes:
                    add(d, reading, f"strength, {ceiling:g} kg{share}", object=k, mass_kg=ceiling, com_offset=com,
                        squeeze_N=sq * ceiling * 9.81, stop_g=0.0, **ap)
            k = blk["nobrake_case"]
            add(d, reading, "hold, no brake", object=k, mass_kg=float(cfg["cases"][k]["mass_kg"]), stop_g=0.0,
                brake=float(cfg["gear_friction_Nm"]), **ap)
            sh = blk["shake_sensitivity"]
            add(d, reading, f"hold, {float(sh['g']):g} g shake", object=sh["case"],
                mass_kg=float(cfg["cases"][sh["case"]]["mass_kg"]), stop_g=0.0, shake_g=float(sh["g"]), **ap)
            if reading in [float(r) for r in blk["stop_held"].get(d, [])]:
                for hold, what in (("stops", "on its stops, gear released"), ("thumb_stop", "outside stop removed")):
                    for k, m in blk["stop_held_cases"]:
                        m = float(cfg["cases"][k]["mass_kg"] if m is None else m)
                        for com in (0.0, off):
                            share = f", COM +{com:.0%} W" if com else ""
                            add(d, reading, f"stop-held, {what}, {m:g} kg{share}", object=k, mass_kg=m, com_offset=com,
                                hold=hold, stop_g=1.0 if m >= ceiling - 1e-9 else 0.0, **ap)
    return jobs


def block_t5() -> list[dict[str, Any]]:
    return t5_jobs(True)


def block_t5_tips() -> list[dict[str, Any]]:
    return t5_jobs(False)


def block_t3() -> list[dict[str, Any]]:
    """T3, tips between boxes, per design and reading: nominal and friction x0.8 (drag x1.2) at the full-case gap
    and the 15 mm entry; then, at nominal friction, the +-3 mm placement corners, the 0 and 3 mm gaps, the 20 mm
    entry and the first box out against the case wall. An asymmetric pair takes both corners and the wall on each
    side. Only the nominal run holds its power cut the full 5 s. A row is `rated` unless its case is one of
    `reported_cases` (revision 2.23: the corners and the 20 mm entry)."""
    b1 = ctx()["b1"]
    cfg = b1["t3"]
    place = float(b1.get("placement_mm", 3.0))
    reported = tuple(cfg.get("reported_cases", ()))
    base = dict(job="t3", test="T3", object=cfg["box"], fill=cfg.get("fill", "catalog"), depth_mm=float(cfg["entry_mm"]),
                gap_mm=float(cfg["gap_mm"]), lateral_mm=0.0)
    jobs = []
    for d, readings in cfg["designs"].items():
        for reading in readings:
            reading = float(reading)
            sides = (1,) if get_jaw(d, reading).symmetric else (1, -1)

            def add(case: str, **kw: Any) -> None:
                jobs.append(_base("t3", d, reading, **{**base, "case": case, "rated": not case.startswith(reported), **kw}))

            add("nominal")
            add("friction x0.8", friction_scale=float(b1.get("robust_friction", 0.8)),
                drag_scale=float(cfg.get("robust_drag", 1.2)), power_s=1.0)
            for s in sides:
                add(f"corner {'+' if s > 0 else '-'}{place:g} mm", lateral_mm=s * place, power_s=1.0)
            for g in cfg["gap_sensitivities_mm"]:
                add(f"gap {float(g):g} mm", gap_mm=float(g), power_s=1.0)
            e = float(cfg["entry_sensitivity_mm"])
            add(f"entry {e:g} mm", depth_mm=e, power_s=1.0)
            for s in sides:
                add(f"case wall {'L' if s > 0 else 'R'}", wall_side=-s, power_s=1.0)
    return jobs


BLOCKS = {"refs": block_refs, "s2": block_s2, "s4": block_s4, "s5": block_s5, "s3p0": block_s3_sheets_platform,
          "s1s6": block_s1_s6, "pr2": block_pr2, "pr2depth": block_pr2_depths, "fsloose": block_fs_loose,
          "t5cut": block_t5_cut, "t5": block_t5, "t5tips": block_t5_tips, "t3": block_t3}


def nobrake_job(rows: list[dict[str, Any]], block: str, design: str) -> list[dict[str, Any]]:
    """The no-brake power cut (about 1 N m of gear friction), once per design, on its best-held box."""
    c = ctx()
    cands = [r for r in rows if r.get("design") == design and r.get("kind") == "screen" and r.get("test") == "T1"
             and r.get("fill") == "catalog" and r.get("simulate")]
    if not cands:
        return []
    cands = [r for r in cands if r.get("nobrake", True)]
    if not cands:
        return []
    best = max(cands, key=lambda r: r.get("robust") or 0.0)
    keep = {k: best[k] for k in ("design", "reading", "object", "fill") if k in best}
    keep.update({k: best[k] for k in ("values", "reach_cap", "library") if best.get(k) is not None})
    return [{**keep, "block": block, "job": "mj", "test": "T1", "case": "no brake", "depth_mm": best["depth_mm"],
             "brake": float(c["b1"].get("power_cut", {}).get("no_brake_Nm_per_gripper", 1.0)), "stops": []}]


MAX_SUBSTEPS = 8


def rerun_job(job: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    """Test conditions 2.15: a MuJoCo run that blew up is rerun at half its timestep, down to dt / 8; its own
    row stays flagged and unscored. A job's result is its row without `reset_flag`, if any."""
    if row.get("kind") != "mujoco" or not row.get("reset_flag"):
        return None
    used = int(row.get("substeps") or (job.get("b1_over") or {}).get("substeps", 1))
    if 2 * used > MAX_SUBSTEPS:
        return None
    return {**job, "b1_over": {**(job.get("b1_over") or {}), "substeps": 2 * used}, "rerun_of_reset": True}


def _job_key(r: dict[str, Any], drop: tuple[str, ...] = ()) -> str:
    keys = ("block", "design", "reading", "values", "reach_cap", "library", "test", "object", "fill", "case", "depth_mm",
            "lateral_mm", "friction_scale", "shake_hz", "band_factor", "brake", "power_s")
    return json.dumps({k: r.get(k) for k in keys if k not in drop}, sort_keys=True, default=str)


def add_verdicts(rows: list[dict[str, Any]]) -> None:
    """Test conditions 2.16: a nominal run's verdict is its own pass and, where a friction x0.8 run exists (T1 at
    catalog fill, T2 at 30 mm), that run holding. A run that blew up counts through its rerun, or not at all."""
    done = {}
    for r in rows:
        if r.get("kind") == "mujoco" and not r.get("reset_flag"):
            done[_job_key(r)] = r
    x08 = {}
    t3: dict[tuple, list[dict[str, Any]]] = {}
    for r in done.values():
        if r.get("case") == "friction x0.8":
            x08[_job_key({**r, "case": None, "friction_scale": None, "power_s": None}, ("lateral_mm",))] = r
        if r.get("test") == "T3" and r.get("rated"):
            t3.setdefault((r.get("block"), r.get("design"), r.get("reading"), r.get("values")), []).append(r)
    for r in rows:
        if r.get("kind") != "mujoco" or r.get("case") not in ("nominal", "pinch 30"):
            continue
        if r.get("reset_flag"):
            r.update(verdict=None, verdict_basis="blew up: flagged, not scored; its rerun carries the verdict"
                     if _job_key(r) in done else "blew up: flagged, not scored, and no rerun held together")
            continue
        if r.get("test") == "T3":
            # Revision 2.23: the centred runs, the gap sensitivities and the first box out; the corners and the 20 mm
            # entry are reported only.
            rated = t3.get((r.get("block"), r.get("design"), r.get("reading"), r.get("values")), [])
            fails = sorted(x["case"] for x in rated if not x.get("passed"))
            r.update(verdict=not fails, verdict_basis=(f"rated rows fail: {', '.join(fails)}" if fails else
                                                       f"all {len(rated)} rated rows pass"))
            continue
        f = x08.get(_job_key({**r, "case": None, "friction_scale": None, "power_s": None}, ("lateral_mm",)))
        if f is None:
            r.update(verdict=bool(r.get("passed")), verdict_basis="nominal only: no friction x0.8 run")
        else:
            r.update(verdict=bool(r.get("passed")) and bool(f.get("passed")),
                     verdict_basis=f"nominal {'passes' if r.get('passed') else 'fails'}, x0.8 "
                                   f"{'holds' if f.get('passed') else 'does not hold'}")


def run(jobs: list[dict[str, Any]], workers: int, log) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(workers) as ex:
        pending = {ex.submit(run_job, j): j for j in jobs if j["job"] != "mj"}
        mj = [j for j in jobs if j["job"] == "mj"]
        pending.update({ex.submit(run_job, j): j for j in mj})
        done_designs: set[str] = set()
        while pending:
            for fut in as_completed(list(pending)):
                job = pending.pop(fut)
                r = fut.result()
                rows.append(r)
                log(r)
                again = rerun_job(job, r)
                if again is not None:
                    pending[ex.submit(run_job, again)] = again
                if r.get("job") == "scan":
                    sym = get_jaw(r["design"], r["reading"], r.get("values"), r.get("reach_cap")).symmetric
                    for f in t1_followups(r, sym):
                        pending[ex.submit(run_job, f)] = f
                scans_left = any(p.get("job") == "scan" and p.get("design") == r.get("design") for p in pending.values())
                if r.get("job") == "scan" and not scans_left and r["design"] not in done_designs:
                    done_designs.add(r["design"])
                    for f in nobrake_job(rows, r["block"], r["design"]):
                        pending[ex.submit(run_job, f)] = f
                break
    return rows


CSV_FIRST = ["commit", "block", "test", "design", "reading", "reach_cap", "values", "object", "fill", "case", "kind", "status",
             "depth_mm", "lateral_mm", "friction_scale", "shake_hz", "brake", "verdict", "verdict_basis", "passed",
             "pass_rule", "score", "limiting", "reason", "reset_flag", "substeps", "rerun_of_reset", "finger_torque_Nm",
             "mount_fatigue_flag"]


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = list(CSV_FIRST)
    for r in rows:
        for k in r:
            if k not in keys and k not in ("traceback", "job", "b1_over", "sens"):
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def select(jobs: list[dict[str, Any]], designs: Iterable[str] = (), objects: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Part of a block: its first-stage jobs for these designs and objects (all when empty). Follow-ups inherit both."""
    d, o = set(designs), set(objects)
    return [j for j in jobs if (not d or j["design"] in d) and (not o or j.get("object") in o)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("block", choices=sorted(BLOCKS))
    ap.add_argument("--out", default="runs/batch1")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--designs", default="", help="comma-separated designs to run (default: the whole block)")
    ap.add_argument("--objects", default="", help="comma-separated object ids to run (default: all)")
    ap.add_argument("--name", default="", help="output file stem (default: the block name)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = commit()
    designs, objects = [s for s in a.designs.split(",") if s], [s for s in a.objects.split(",") if s]
    jobs = select(BLOCKS[a.block](), designs, objects)
    stem = a.name or a.block
    t0 = time.time()
    logf = (out / f"{stem}.log").open("w")

    def log(r: dict[str, Any]) -> None:
        msg = (f"{time.time() - t0:7.1f}s {r.get('design', ''):16s} {r.get('test', ''):3s} {r.get('object', ''):24s} "
               f"{r.get('fill', '') or '':8s} {r.get('case', '') or r.get('job', ''):28s} {r.get('kind', ''):7s} "
               f"score {r.get('score') if r.get('score') is not None else float('nan'):6.2f} {r.get('limiting') or r.get('reason') or ''}")
        print(msg, file=logf, flush=True)

    part = "".join(f", {k} {','.join(v)}" for k, v in (("designs", designs), ("objects", objects)) if v)
    print(f"block {a.block}{part}: {len(jobs)} first-stage jobs, commit {sha}", file=logf, flush=True)
    rows = run(jobs, a.workers, log)
    for r in rows:
        r["commit"] = sha
    add_verdicts(rows)
    (out / f"{stem}.json").write_text(json.dumps({"commit": sha, "block": a.block, "designs": designs or None,
                                                  "objects": objects or None, "seconds": round(time.time() - t0, 1),
                                                  "rows": rows}, indent=0, default=str))
    write_rows(rows, out / f"{stem}.csv")
    print(f"done: {len(rows)} rows in {time.time() - t0:.0f} s", file=logf, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
