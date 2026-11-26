"""Batch 1 of the concept review: each design's jaw, the screen and MuJoCo over the batch-1 tests in a
process pool, and the result rows (the review team's batch-1 plan, "Run order").

Run it from a snapshot of the repo (a git worktree), so commits on the branch can't change a run in
flight, and stop it with `pkill -9 -f graspsim2d.batch1`, never Ctrl-C:

    python -m graspsim2d.batch1 refs --out runs/batch1
"""

from __future__ import annotations

import argparse
import csv
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


def ctx() -> dict[str, Any]:
    if not _CTX:
        from .jaw import load_marks
        from .objects import load_objects
        from .scenario import load_scenario
        sc = load_scenario(B1 / "scenario.yaml")
        _CTX.update(sc=sc, b1=sc.raw["batch1"], lib=load_objects(B1 / "objects.yaml"),
                    lay=yaml.safe_load((B1 / "layouts.yaml").read_text()), marks=load_marks(B1 / "marked_faces.yaml"),
                    jaws={})
    return _CTX


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
    from .jaw import jaw_from_design, jaw_from_spec
    if design == "r2-09":
        from .schema import GripperDesign
        d = GripperDesign.from_dict(json.loads((B1 / "r2-09.json").read_text()))
        j = jaw_from_design(d, c["sc"], reading=reading, concept="r2-09")
    elif design == "platform-padded":
        j = jaw_from_design(platform_padded(), c["sc"], reading=reading, concept="platform-padded")
    else:
        from .outline import build_gripper
        from .specs import SpecDesign, load_spec
        sp = load_spec(B1 / "specs" / f"{SPEC_FILES[design]}.yaml", reach_cap=reach_cap)
        dd = SpecDesign.make(sp, values or {}, reading, f"{design} " + (", ".join(f"{k} {v:g}" for k, v in (values or {}).items()) or "nominal"))
        j = jaw_from_spec(build_gripper(sp, dd, c["sc"].keep_out, c["sc"].closed_limit, c["lay"]), c["marks"])
    c["jaws"][key] = j
    return j


def get_obj(oid: str, fill: str = "catalog"):
    from .objects import with_fill
    c = ctx()
    o = c["lib"].by_id(oid)
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
    o = get_obj(job["object"], job["fill"])
    lim = Limits.from_config(c["b1"], "T1")
    r = scan_box(j, o, c["sc"], lim)
    row = {**job, "kind": "screen", "test": "T1", **{k: v for k, v in r.row().items() if k != "object"}}
    row["table_cap_mm"] = o.height * 1e3 - 2 * TABLE_CLEARANCE
    g = r.nominal
    row["status"] = g.status if g is not None else "nogrip"
    row["simulate"] = bool(g is not None and g.status != "nogrip" and math.isfinite(r.depth) and (g.ok or g.status == "mujoco"))
    if not row["simulate"]:
        row["not_simulated"] = (g.reason if g is not None else "") or r.ended
    return {**row, **_torque_flag(j)}


def _torque_flag(j) -> dict[str, Any]:
    lim = float(ctx()["b1"].get("mount_fatigue_Nm_per_finger", 4.0))
    return {"finger_torque_Nm": j.torque, "mount_fatigue_flag": bool(j.torque > lim)}


def job_sheet_screen(job: dict[str, Any]) -> dict[str, Any]:
    from .screen1 import Limits, solve_sheet
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    o = get_obj(job["object"])
    g = solve_sheet(j, o, c["sc"], Limits.from_config(c["b1"], "T2"), job["depth_mm"])
    return {**job, "kind": "screen", "test": "T2", "status": g.status, "ok": g.ok, "score": g.score, "limiting": g.limiting,
            "reason": g.reason, **_flat("m_", g.margins), **_flat("d_", g.demand),
            **{k: v for k, v in g.info.items() if not isinstance(v, (list, dict))}, **_torque_flag(j)}


def job_mj(job: dict[str, Any]) -> dict[str, Any]:
    """One MuJoCo run."""
    from . import mj1
    c = ctx()
    j = get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    o = get_obj(job["object"], job.get("fill", "catalog"))
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


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        f = {"scan": job_scan, "sheet_screen": job_sheet_screen, "mj": job_mj}[job["job"]]
        return _clean(f(job))
    except Exception as e:  # noqa: BLE001 - a failed job is a row, not a dead batch
        import traceback
        return {**job, "kind": "error", "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}


# ---------------------------------------------------------------- blocks

def _base(block: str, design: str, reading: float, **kw: Any) -> dict[str, Any]:
    return {"block": block, "design": design, "reading": reading, **kw}


def t1_jobs(block: str, design: str, reading: float, boxes: Iterable[str], sens: list | None = None,
            **kw: Any) -> list[dict[str, Any]]:
    """`sens`: extra MuJoCo cases at the nominal depth on the catalog fill, as (case, overrides)."""
    base = _base(block, design, reading, **kw)
    if sens:
        base["sens"] = sens
    return [{**base, "job": "scan", "object": b, "fill": f} for b in boxes for f in FILLS]


def t1_followups(scan: dict[str, Any], symmetric: bool = True) -> list[dict[str, Any]]:
    """MuJoCo at the screen's depth, the two +-3 mm corners (deeper and shallower, offset sideways) and,
    at catalog fill, friction x0.8."""
    if not scan.get("simulate"):
        return []
    keep = {k: scan[k] for k in ("block", "design", "reading", "object", "fill") if k in scan}
    keep.update({k: scan[k] for k in ("values", "reach_cap") if scan.get(k) is not None})
    c = ctx()
    place = float(c["b1"].get("placement_mm", 3.0))
    d = float(scan["depth_mm"])
    table = float(scan["table_cap_mm"])
    deep = d + place
    if scan.get("stop_mm") is not None and math.isfinite(scan["stop_mm"] or math.nan):
        deep = min(deep, float(scan["stop_mm"]))
    deep = min(deep, table)
    out = [{**keep, "job": "mj", "test": "T1", "case": "nominal", "depth_mm": d, "lateral_mm": 0.0}]
    sides = (place,) if symmetric else (place, -place)
    for s in sides:
        out.append({**keep, "job": "mj", "test": "T1", "case": f"corner deep{'+' if s > 0 else '-'}", "depth_mm": deep,
                    "lateral_mm": s, "power_s": 1.0})
        out.append({**keep, "job": "mj", "test": "T1", "case": f"corner shallow{'+' if s > 0 else '-'}",
                    "depth_mm": max(d - place, place), "lateral_mm": s, "power_s": 1.0})
    if scan["fill"] == "catalog":
        out.append({**keep, "job": "mj", "test": "T1", "case": "friction x0.8", "depth_mm": d, "lateral_mm": 0.0,
                    "friction_scale": float(c["b1"].get("robust_friction", 0.8)), "power_s": 1.0})
        for case, over in scan.get("sens") or []:
            out.append({**keep, "job": "mj", "test": "T1", "case": case, "depth_mm": d, "lateral_mm": 0.0, "power_s": 1.0,
                        **over})
    return out


def t2_jobs(block: str, design: str, reading: float, sheets: Iterable[str], **kw: Any) -> list[dict[str, Any]]:
    c = ctx()
    hz_sweep = [float(h) for h in c["b1"].get("sheet_shake_hz", [1.5, 2.0, 3.0])]
    base = _base(block, design, reading, **kw)
    out = []
    for s in sheets:
        for p in PINCHES:
            out.append({**base, "job": "sheet_screen", "object": s, "depth_mm": p, "case": f"pinch {p:g}"})
            out.append({**base, "job": "mj", "test": "T2", "object": s, "depth_mm": p, "case": f"pinch {p:g}"})
        out.append({**base, "job": "mj", "test": "T2", "object": s, "depth_mm": 30.0, "case": "friction x0.8",
                    "friction_scale": float(c["b1"].get("robust_friction", 0.8)), "power_s": 1.0})
        if s in BLANKS:
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


BLOCKS = {"refs": block_refs, "s2": block_s2, "s5": block_s5, "s3p0": block_s3_sheets_platform, "s1s6": block_s1_s6}


def nobrake_job(rows: list[dict[str, Any]], block: str, design: str) -> list[dict[str, Any]]:
    """The no-brake power cut (about 1 N m of gear friction), once per design, on its best-held box."""
    c = ctx()
    cands = [r for r in rows if r.get("design") == design and r.get("kind") == "screen" and r.get("test") == "T1"
             and r.get("fill") == "catalog" and r.get("simulate")]
    if not cands:
        return []
    best = max(cands, key=lambda r: r.get("robust") or 0.0)
    keep = {k: best[k] for k in ("design", "reading", "object", "fill") if k in best}
    keep.update({k: best[k] for k in ("values", "reach_cap") if best.get(k) is not None})
    return [{**keep, "block": block, "job": "mj", "test": "T1", "case": "no brake", "depth_mm": best["depth_mm"],
             "brake": float(c["b1"].get("power_cut", {}).get("no_brake_Nm_per_gripper", 1.0)), "stops": []}]


def run(jobs: list[dict[str, Any]], workers: int, log) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(workers) as ex:
        pending = {ex.submit(run_job, j): j for j in jobs if j["job"] != "mj"}
        mj = [j for j in jobs if j["job"] == "mj"]
        pending.update({ex.submit(run_job, j): j for j in mj})
        done_designs: set[str] = set()
        while pending:
            for fut in as_completed(list(pending)):
                pending.pop(fut)
                r = fut.result()
                rows.append(r)
                log(r)
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
             "depth_mm", "lateral_mm", "friction_scale", "shake_hz", "brake", "passed", "score", "limiting", "reason",
             "finger_torque_Nm", "mount_fatigue_flag"]


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("block", choices=sorted(BLOCKS))
    ap.add_argument("--out", default="runs/batch1")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = commit()
    jobs = BLOCKS[a.block]()
    t0 = time.time()
    logf = (out / f"{a.block}.log").open("w")

    def log(r: dict[str, Any]) -> None:
        msg = (f"{time.time() - t0:7.1f}s {r.get('design', ''):16s} {r.get('test', ''):3s} {r.get('object', ''):24s} "
               f"{r.get('fill', '') or '':8s} {r.get('case', '') or r.get('job', ''):28s} {r.get('kind', ''):7s} "
               f"score {r.get('score') if r.get('score') is not None else float('nan'):6.2f} {r.get('limiting') or r.get('reason') or ''}")
        print(msg, file=logf, flush=True)

    print(f"block {a.block}: {len(jobs)} first-stage jobs, commit {sha}", file=logf, flush=True)
    rows = run(jobs, a.workers, log)
    for r in rows:
        r["commit"] = sha
    (out / f"{a.block}.json").write_text(json.dumps({"commit": sha, "block": a.block, "seconds": round(time.time() - t0, 1),
                                                     "rows": rows}, indent=0, default=str))
    write_rows(rows, out / f"{a.block}.csv")
    print(f"done: {len(rows)} rows in {time.time() - t0:.0f} s", file=logf, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
