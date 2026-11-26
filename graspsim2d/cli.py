"""Command line: python -m graspsim2d <command> [options]."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

from .analytic import screen_design
from .objects import load_objects
from .scenario import load_scenario
from .schema import GripperDesign, load_concepts, load_design_space

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


def cmd_screen(args: argparse.Namespace) -> None:
    paths = _paths(args)
    d = _load_design(args.design, paths)
    lib = load_objects(paths["objects"])
    sc = load_scenario(paths["scenario"])
    res = screen_design(d, lib.objects, sc)
    print(f"Analytic screen of {d.label or d.concept}:")
    for r in res:
        if r.ok:
            m = " ".join(f"{k}={v:.2f}" for k, v in r.margins.items())
            print(f"  {r.object_id:18s} {'PASS' if r.passed else 'fail'}  score {r.score:5.2f}  N {r.info['normal_force_N']:5.1f} N  "
                  f"insertion {r.info['insertion_mm']:5.1f} mm  {m}")
        else:
            print(f"  {r.object_id:18s} fail  {r.reason}")


def cmd_simulate(args: argparse.Namespace) -> None:
    from .mjsim import build_model, simulate
    from .plots import fig_traces

    paths = _paths(args)
    d = _load_design(args.design, paths)
    lib = load_objects(paths["objects"])
    sc = load_scenario(paths["scenario"])
    ids = args.object_ids.split(",") if args.object_ids else lib.ids()
    traces = {}
    for oid in ids:
        o = lib.by_id(oid)
        t0 = time.perf_counter()
        r = simulate(d, o, sc)
        m = r.metrics
        print(f"  {oid:18s} {'PASS' if r.passed else 'FAIL'} {r.reason:28s} slip {m.get('slip_mm', float('nan')):6.2f} mm  "
              f"rot {m.get('rotation_deg', float('nan')):5.2f} deg  p {m.get('peak_pressure_kPa', float('nan')):6.1f} kPa  "
              f"({time.perf_counter() - t0:.1f} s)")
        traces[o.label] = (d.label or d.concept, r.trace)
        if args.xml:
            xml, _ = build_model(d, o, sc, r.insertion_mm / 1e3)
            Path(args.xml).mkdir(parents=True, exist_ok=True)
            (Path(args.xml) / f"{oid}.xml").write_text(xml)
    if args.plot:
        fig_traces(Path(args.plot), traces, sc)
        print(f"traces -> {args.plot}")


def cmd_search(args: argparse.Namespace) -> None:
    from .search import SearchConfig, SearchRun

    paths = _paths(args)
    cfg = SearchConfig.load(paths["search"])
    if args.quick:
        cfg.n_lhs, cfg.de_maxiter, cfg.per_object_maxiter, cfg.finalists_per_concept, cfg.rounds = 24, 2, 2, 1, 1
    if args.rounds:
        cfg.rounds = args.rounds
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("design_space", "concepts", "objects", "scenario", "search"):
        if Path(paths[k]).exists():
            shutil.copy(paths[k], out / Path(paths[k]).name)
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


def make_figures(out: Path, paths: dict[str, str], log=print) -> list[Path]:
    from .mjsim import simulate
    from .plots import fig_best_designs, fig_bounds, fig_concepts, fig_design_on_objects, fig_iteration, fig_objects, fig_pass_matrix, fig_traces

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
    sub.add_parser("describe", help="print the design space, concepts and objects")
    s = sub.add_parser("screen", help="analytic screen of one design on every object")
    s.add_argument("design", help="concept name (nominal design), results.json:NAME, or a design JSON file")
    s = sub.add_parser("simulate", help="MuJoCo check of one design")
    s.add_argument("design")
    s.add_argument("--objects", dest="object_ids", help="comma-separated object ids (default: all)")
    s.add_argument("--plot", help="write time histories to this PNG")
    s.add_argument("--xml", help="write the MJCF models to this directory")
    s = sub.add_parser("search", help="ideate, simulate, iterate over all concepts")
    s.add_argument("--out", default="runs/latest")
    s.add_argument("--rounds", type=int)
    s.add_argument("--quick", action="store_true", help="tiny budget, for testing the pipeline")
    s = sub.add_parser("figures", help="redraw the figures of a finished run")
    s.add_argument("run")
    args = p.parse_args(argv)
    {"describe": cmd_describe, "screen": cmd_screen, "simulate": cmd_simulate, "search": cmd_search, "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
