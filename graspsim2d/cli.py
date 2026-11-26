"""Command line: python -m graspsim2d <command> [options]."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import yaml

from .objects import load_objects, table_path
from .scenario import load_scenario
from .schema import GripperDesign, load_concepts, load_design_space
from .simulator import FIDELITIES

CONFIG_FILES = {"design_space": "design_space.yaml", "concepts": "concepts.yaml", "objects": "objects.yaml",
                "scenario": "scenario.yaml", "search": "search.yaml"}


def _paths(args: argparse.Namespace) -> dict[str, str]:
    base = Path(args.config)
    out = {k: str(base / v) for k, v in CONFIG_FILES.items()}
    for k in CONFIG_FILES:
        v = getattr(args, f"file_{k}", None)
        if v:
            out[k] = v
    return out


def save_inputs(out: Path, paths: dict[str, str]) -> None:
    """Copy the configs, and the box tables the objects file names, into a run directory."""
    out.mkdir(parents=True, exist_ok=True)
    for k in CONFIG_FILES:
        if Path(paths[k]).exists():
            shutil.copy(paths[k], out / Path(paths[k]).name)
    obj = Path(paths["objects"])
    if not obj.exists():
        return
    raw = yaml.safe_load(obj.read_text()) or {}
    moved = False
    for t in raw.get("box_tables") or []:
        src = table_path(t, obj.parent)
        if src.exists():
            shutil.copy(src, out / src.name)
            if str(t["file"]) != src.name:
                t["file"], moved = src.name, True
    if moved:
        (out / obj.name).write_text(f"# Copy of {obj} with box tables pointed at the copies beside it.\n"
                                    + yaml.safe_dump(raw, sort_keys=False))


def _load_design(spec: str, paths: dict[str, str]) -> GripperDesign:
    """`concept` (nominal), `results.json:NAME`, or a design JSON file."""
    space = load_design_space(paths["design_space"])
    concepts = {c.name: c for c in load_concepts(paths["concepts"], space)}
    if spec in concepts:
        return concepts[spec].make(space)
    if ":" in spec:
        f, name = spec.rsplit(":", 1)
        res = json.loads(Path(f).read_text())
        return GripperDesign.from_dict(res["designs"][name]["design"])
    return GripperDesign.from_dict(json.loads(Path(spec).read_text()))


def cmd_describe(args: argparse.Namespace) -> None:
    paths = _paths(args)
    space = load_design_space(paths["design_space"])
    print(f"Design space ({paths['design_space']}):")
    print(f"  {'variable':22s} {'unit':8s} {'nominal':>8s} {'lower':>8s} {'upper':>8s}  placeholder")
    for row in space.describe():
        print(f"  {row['name']:22s} {row['unit']:8s} {row['nominal']:8.3g} {row['lower']:8.3g} {row['upper']:8.3g}  {row['placeholder']}")
    concepts = load_concepts(paths["concepts"], space)
    print("\nConcepts:")
    for c in concepts:
        print(f"  {c.name:14s} {c.label}  ({len(c.active)} variables searched)")
    lib = load_objects(paths["objects"])
    print("\nObjects:")
    for o in lib.objects:
        s = o.summary()
        print(f"  {o.id:18s} {o.kind:5s} {s['grasp_dimension_mm']:7.2f} mm  {s['mass_g']:6.1f} g  mu {s['friction']:.2f}  "
              f"allow {s['allow_pressure_kPa']:.0f} kPa")


def _print_outcome(r: dict) -> None:
    m = r["metrics"]
    head = f"  {r['object_id']:18s} {'PASS' if r['passed'] else 'fail'}  score {r['score']:5.2f}"
    if r["fidelity"] == "analytic":
        body = (f"  N {m['normal_force_N']:5.1f} N  insertion {r['insertion_mm']:5.1f} mm" if "normal_force_N" in m else "")
    else:
        body = (f"  slip {m['slip_mm']:6.2f} mm  rot {m['rotation_deg']:5.2f} deg  p {m['peak_pressure_kPa']:6.1f} kPa"
                if "slip_mm" in m else "")
    margins = " ".join(f"{k}={v:.2f}" for k, v in r["margins"].items())
    tail = f"  ({r['seconds']:.1f} s)" if r["fidelity"] == "mujoco" else ""
    print(f"{head}{body}  {margins or r['reason']}{tail}")


def _save_outcomes(out: Path, paths: dict[str, str], d: GripperDesign, rows: list[dict]) -> None:
    from .sweep import clean, write_outcomes_csv

    save_inputs(out, paths)
    (out / "results.json").write_text(json.dumps(clean({"design": d.to_dict(), "outcomes": rows}), indent=1))
    write_outcomes_csv(out / "results.csv", rows)
    print(f"results -> {out / 'results.json'}, {out / 'results.csv'}")


def cmd_simulate(args: argparse.Namespace) -> None:
    from .simulator import make_simulator, run_batch

    paths = _paths(args)
    d = _load_design(args.design, paths)
    lib = load_objects(paths["objects"])
    sc = load_scenario(paths["scenario"])
    ids = args.object_ids.split(",") if args.object_ids else lib.ids()
    fidelity = getattr(args, "fidelity", "mujoco")
    plot, xml = getattr(args, "plot", None), getattr(args, "xml", None)
    print(f"{'Analytic screen' if fidelity == 'analytic' else 'MuJoCo check'} of {d.label or d.concept}:")
    if fidelity == "mujoco" and (plot or xml):
        # Time histories and models come from runs in this process, one grasp at a time.
        from .geometry import build_finger

        sim = make_simulator(fidelity, record=True)
        f = build_finger(d, sc.keep_out, sc.closed_limit)
        outs = [sim.run(d, lib.by_id(o), sc, finger=f) for o in ids]
        rows = [{"point": 0, **o.to_dict()} for o in outs]
    else:
        outs, rows = [], run_batch(fidelity, [d], ids, paths["objects"], paths["scenario"], args.workers or None)
    for r in rows:
        _print_outcome(r)
    print(f"passes {sum(r['passed'] for r in rows)}/{len(rows)}")
    if outs and plot:
        from .plots import fig_traces

        Path(plot).parent.mkdir(parents=True, exist_ok=True)
        fig_traces(Path(plot), {lib.by_id(o.object_id).label: (d.label or d.concept, o.trace) for o in outs}, sc)
        print(f"traces -> {plot}")
    if outs and xml:
        from .mjsim import build_model

        Path(xml).mkdir(parents=True, exist_ok=True)
        for o in outs:
            if math.isfinite(o.insertion_mm):
                (Path(xml) / f"{o.object_id}.xml").write_text(build_model(d, lib.by_id(o.object_id), sc, o.insertion_mm / 1e3)[0])
    if args.out:
        _save_outcomes(Path(args.out), paths, d, rows)


def cmd_screen(args: argparse.Namespace) -> None:
    args.fidelity = "analytic"
    cmd_simulate(args)


def cmd_sweep(args: argparse.Namespace) -> None:
    from .plots import fig_sweep
    from .sweep import parse_vary, run_sweep, save_sweep

    paths = _paths(args)
    base = _load_design(args.design, paths)
    lib = load_objects(paths["objects"])
    ids = args.object_ids.split(",") if args.object_ids else lib.ids()
    vary = [parse_vary(v) for v in args.vary]
    space = load_design_space(paths["design_space"])
    for name, vals in vary:
        r = space[name]
        if min(vals) < r.lower - 1e-9 or max(vals) > r.upper + 1e-9:
            print(f"note: {name} values go outside the bounding box [{r.lower:g}, {r.upper:g}]")
    out = Path(args.out)
    save_inputs(out, paths)
    n = math.prod(len(v) for _, v in vary)
    print(f"Sweep around {base.label or base.concept}: {n} points x {len(ids)} objects, {args.fidelity} fidelity -> {out}")
    t0 = time.perf_counter()
    res = run_sweep(base, vary, ids, paths, args.fidelity, workers=args.workers or None)
    files = save_sweep(out, res)
    if len(vary) <= 2:
        fig_sweep(out / "figures" / "sweep.png", res, {o.id: o.label for o in lib.objects})
        files.append(out / "figures" / "sweep.png")
    names = [nm for nm, _ in vary]
    print("  " + "  ".join(f"{nm:>14s}" for nm in names) + f"  {'pass':>6s}  {'J':>6s}  {'min':>5s}  worst object")
    for row in res["summary"]:
        print("  " + "  ".join(f"{row[nm]:14.4g}" for nm in names)
              + f"  {row['n_pass']:>2d}/{row['n_objects']:<3d}  {row['J']:6.2f}  {row['min_score']:5.2f}  {row['worst_object']} ({row['worst_reason']})")
    best = max(res["summary"], key=lambda r: (r["n_pass"], r["J"]))
    print(f"best point: {' '.join(f'{nm}={best[nm]:g}' for nm in names)} passes {best['n_pass']}/{best['n_objects']}")
    print(f"done in {time.perf_counter() - t0:.0f} s; wrote " + ", ".join(str(f) for f in files))


def cmd_search(args: argparse.Namespace) -> None:
    from .search import SearchConfig, SearchRun

    paths = _paths(args)
    cfg = SearchConfig.load(paths["search"])
    if args.quick:
        cfg.n_lhs, cfg.de_maxiter, cfg.per_object_maxiter, cfg.finalists_per_concept, cfg.rounds = 24, 2, 2, 1, 1
    if args.rounds:
        cfg.rounds = args.rounds
    if args.workers:
        cfg.workers = args.workers
    out = Path(args.out)
    save_inputs(out, paths)
    log_f = open(out / "search.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    run = SearchRun(paths=paths, out_dir=out, cfg=cfg, log=log)
    result = run.run()
    log(f"done in {result['seconds']:.0f} s; all-rounder {result['all_rounder']}")
    _report(result, run, log)
    make_figures(out, paths, log=log)
    log_f.close()


def _report(result: dict, run, log) -> None:
    ids = result["objects"]
    ar = result["designs"][result["all_rounder"]]
    log(f"\nBest all-rounder: {ar['name']} ({ar['concept']}), passes {ar['sim_pass']}/{len(ids)} in the dynamic check")
    for o in ids:
        r = ar["results"][o]
        log(f"  {o:18s} {'PASS' if r['passed'] else 'FAIL'}  {r['reason']}")
    log("\nBest design per object:")
    for o, p in result["per_object"].items():
        log(f"  {o:18s} {p['design']} ({p['concept']})  {'PASS' if p['passed'] else 'FAIL'}  score {p['score']:.2f}  {p['reason']}")
    for name in result.get("references", []):
        ref = result["designs"][name]
        log(f"\nReference {name} ({ref['why'][0]}): passes {ref['sim_pass']}/{len(ids)} in the dynamic check")
        for o in ids:
            r = ref["results"][o]
            log(f"  {o:18s} {'PASS' if r['passed'] else 'FAIL'}  score {r['score']:.2f}  {r['reason']}")
    rows = [d for d in result["designs"].values() if d.get("hub_clearance_mm") is not None]
    if rows:
        log("\nClearance over the stroke (least gap, and the opening from where the fingers meet at which it occurs):")
        for d in sorted(rows, key=lambda d: d["name"]):
            lig = f"ligament {d['ligament_mm']:.1f} mm" if d.get("ligament_mm") is not None else "no cut needed"
            brk = (f"brackets {d['bracket_clearance_mm']:6.2f} mm at {d['bracket_clearance_phi_deg']:5.1f} deg ({d.get('bracket', '')})"
                   if d.get("bracket_clearance_mm") is not None else "brackets not checked")
            wide = (f"  clear to {d['widest_opening_deg']:5.1f} deg ({d.get('widest_opening_limit', '')})"
                    if d.get("widest_opening_deg") is not None else "")
            log(f"  {d['name']:9s} {d['concept']:16s} hub {d['hub_clearance_mm']:5.2f} mm at {d['hub_clearance_phi_deg']:5.1f} deg  {lig:16s}  {brk}{wide}")


def make_figures(out: Path, paths: dict[str, str], log=print) -> list[Path]:
    from .mjsim import simulate
    from .plots import (fig_best_designs, fig_bounds, fig_bracket_clearance, fig_concepts, fig_design_on_objects, fig_iteration,
                        fig_objects, fig_pass_matrix, fig_traces)

    result = json.loads((out / "results.json").read_text())
    archive = json.loads((out / "archive.json").read_text()) if (out / "archive.json").exists() else []
    space = load_design_space(paths["design_space"])
    concepts = load_concepts(paths["concepts"], space)
    lib = load_objects(paths["objects"])
    sc = load_scenario(paths["scenario"])
    fdir = out / "figures"
    fdir.mkdir(exist_ok=True)
    made = []
    fig_best_designs(fdir / "first-pass-best-designs.png", result, lib, sc)
    made.append(fdir / "first-pass-best-designs.png")
    for name in result.get("references", []):
        p = fdir / ("baseline-as-drawn.png" if name == "baseline" else f"{name}-as-drawn.png")
        fig_design_on_objects(p, result, name, lib, sc)
        made.append(p)
    fig_concepts(fdir / "concepts-nominal.png", concepts, space, lib, sc)
    made.append(fdir / "concepts-nominal.png")
    best = {c: GripperDesign.from_dict(result["designs"][n]["design"]) for c, n in result["best_by_concept"].items()}
    sims = {c: result["designs"][n]["results"] for c, n in result["best_by_concept"].items()}
    fig_concepts(fdir / "concepts-best.png", concepts, space, lib, sc, designs=best, sims=sims)
    made.append(fdir / "concepts-best.png")
    fig_objects(fdir / "object-library.png", lib)
    made.append(fdir / "object-library.png")
    fig_pass_matrix(fdir / "pass-matrix.png", result, lib)
    made.append(fdir / "pass-matrix.png")
    fig_iteration(fdir / "iteration-progress.png", archive, result, len(lib.objects))
    made.append(fdir / "iteration-progress.png")
    fig_bounds(fdir / "designs-in-bounds.png", result, space)
    made.append(fdir / "designs-in-bounds.png")
    if sc.keep_out is not None and sc.keep_out.brackets:
        fig_bracket_clearance(fdir / "bracket-clearance.png", result, sc)
        made.append(fdir / "bracket-clearance.png")
    ar = result["designs"][result["all_rounder"]]
    gd = GripperDesign.from_dict(ar["design"])
    picks = _trace_objects(lib, ar)
    traces = {}
    for o in picks:
        dep = (ar["results"].get(o.id) or {}).get("insertion_mm")
        r = simulate(gd, o, sc, depth=dep / 1e3 if dep and math.isfinite(dep) else None)
        traces[o.label] = (ar["name"], r.trace)
    fig_traces(fdir / "shake-traces.png", traces, sc)
    made.append(fdir / "shake-traces.png")
    for p in made:
        log(f"figure: {p}")
    return made


def _trace_objects(lib, ar) -> list:
    boxes = [o for o in lib.objects if o.kind == "box"]
    sheets = [o for o in lib.objects if o.kind == "sheet"]
    ok = lambda o: (ar["results"].get(o.id) or {}).get("reason", "").startswith("not simulated") is False
    picks = []
    for group in (sorted(boxes, key=lambda o: -o.mass), sorted(sheets, key=lambda o: o.thickness), sorted(sheets, key=lambda o: -o.thickness)):
        for o in group:
            if ok(o) and o not in picks:
                picks.append(o)
                break
    return picks


def cmd_figures(args: argparse.Namespace) -> None:
    out = Path(args.run)
    paths = {k: str(out / v) if (out / v).exists() else str(Path(args.config) / v) for k, v in CONFIG_FILES.items()}
    make_figures(out, paths)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="graspsim2d", description=__doc__)
    p.add_argument("--config", default="config", help="directory with the YAML configs (default: config)")
    for k in CONFIG_FILES:
        p.add_argument(f"--{k.replace('_', '-')}", dest=f"file_{k}", metavar="FILE", help=f"override {CONFIG_FILES[k]}")
    sub = p.add_subparsers(dest="cmd", required=True)
    design_help = "concept name (nominal design), results.json:NAME, or a design JSON file"
    objects_help = "comma-separated object ids (default: all)"
    workers_help = "worker processes (default: all cores)"
    sub.add_parser("describe", help="print the design space, concepts and objects")
    s = sub.add_parser("screen", help="analytic screen of one design on every object (simulate --fidelity analytic)")
    s.add_argument("design", help=design_help)
    s.add_argument("--objects", dest="object_ids", help=objects_help)
    s.add_argument("--out", help="save results.json, results.csv and the inputs in this directory")
    s.add_argument("--workers", type=int, default=0, help=workers_help)
    s = sub.add_parser("simulate", help="one design on every object, at either fidelity (default: MuJoCo)")
    s.add_argument("design", help=design_help)
    s.add_argument("--fidelity", choices=FIDELITIES, default="mujoco")
    s.add_argument("--objects", dest="object_ids", help=objects_help)
    s.add_argument("--out", help="save results.json, results.csv and the inputs in this directory")
    s.add_argument("--plot", help="MuJoCo: write time histories to this PNG")
    s.add_argument("--xml", help="MuJoCo: write the MJCF models to this directory")
    s.add_argument("--workers", type=int, default=0, help=workers_help)
    s = sub.add_parser("sweep", help="grid over design variables around one design, every object, all cores")
    s.add_argument("design", help=design_help)
    s.add_argument("--vary", action="append", required=True, metavar="NAME=VALUES",
                   help="name=a,b,c or name=lo:hi:n (canonical units, see describe); repeat for a grid")
    s.add_argument("--fidelity", choices=FIDELITIES, default="analytic")
    s.add_argument("--objects", dest="object_ids", help=objects_help)
    s.add_argument("--out", default="runs/sweep")
    s.add_argument("--workers", type=int, default=0, help=workers_help)
    s = sub.add_parser("search", help="ideate, simulate, iterate over all concepts")
    s.add_argument("--out", default="runs/latest")
    s.add_argument("--rounds", type=int)
    s.add_argument("--quick", action="store_true", help="tiny budget, for testing the pipeline")
    s.add_argument("--workers", type=int, default=0, help=workers_help)
    s = sub.add_parser("figures", help="redraw the figures of a finished run")
    s.add_argument("run")
    args = p.parse_args(argv)
    {"describe": cmd_describe, "screen": cmd_screen, "simulate": cmd_simulate, "sweep": cmd_sweep, "search": cmd_search,
     "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
