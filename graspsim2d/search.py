"""Ideate, simulate, iterate.

Round r:
  1. Screen each concept: Latin hypercube over its bounds, then differential
     evolution on the all-rounder objective, both on the analytic screen
     (corrected by what earlier rounds learned from the dynamic check).
  2. Per-object bests: a short differential evolution per object, started
     from the best concept for that object.
  3. Dynamic check: the top designs of every concept, the best all-rounder
     and the per-object bests, each simulated on every object in MuJoCo.
  4. Learn: for each object and criterion, the ratio of dynamic to analytic
     margin over the checked designs becomes a correction factor on the
     analytic screen for the next round.

All-rounder objective: J = sum_o min(m_o, 1) + w * min_o min(m_o, 2), where
m_o is the smallest pass margin on object o (0 if the grasp fails outright):
first pass as many objects as possible, then raise the weakest margin.
"""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from .analytic import GripResult, solve_grip
from .geometry import build_finger
from .objects import ObjectLibrary, load_objects
from .scenario import Scenario, load_scenario
from .schema import Concept, DesignSpace, GripperDesign, load_concepts, load_design_space

# Analytic criteria standing in for each dynamic criterion when learning corrections.
# Slip itself is a displacement and has no analytic counterpart; the share of the
# available friction the pads use ("friction") is the force ratio the hold margin
# predicts.
CRITERION_MAP = {
    "friction": ("hold",),
    "rotation": ("rotation", "pitch"),
    "pressure": ("pressure",),
    "panel": ("panel",),
    "crease": ("crease",),
}


@dataclass
class SearchConfig:
    n_lhs: int = 256
    de_popsize: int = 3
    de_maxiter: int = 40
    de_tol: float = 1e-3
    per_object_maxiter: int = 30
    finalists_per_concept: int = 3
    rounds: int = 2
    min_margin_weight: float = 0.25
    workers: int = 0
    seed: int = 1
    correction_clip: tuple[float, float] = (0.25, 2.0)

    @classmethod
    def load(cls, path: str | Path | None) -> "SearchConfig":
        if path is None or not Path(path).exists():
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        kw = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if "correction_clip" in kw:
            kw["correction_clip"] = tuple(kw["correction_clip"])
        return cls(**kw)


# ----------------------------------------------------------------- workers

_W: dict[str, Any] = {}


def _init_worker(paths: dict[str, str], corrections: dict[str, dict[str, float]]) -> None:
    _W["space"] = load_design_space(paths["design_space"])
    _W["concepts"] = {c.name: c for c in load_concepts(paths["concepts"], _W["space"])}
    _W["lib"] = load_objects(paths["objects"])
    _W["sc"] = load_scenario(paths["scenario"])
    _W["corr"] = corrections
    _W["weight"] = float(paths.get("weight", 0.25))


def corrected_score(r: GripResult, corr: dict[str, dict[str, float]]) -> float:
    if not r.ok or not r.margins:
        return 0.0
    c = corr.get(r.object_id, {})
    return float(min(v * c.get(k, 1.0) for k, v in r.margins.items()))


def objective_from_scores(scores: list[float], weight: float) -> float:
    s = np.asarray(scores, dtype=float)
    return float(np.sum(np.minimum(s, 1.0)) + weight * np.min(np.minimum(s, 2.0)))


def _evaluate(task: tuple[str, tuple[float, ...], tuple[str, ...] | None]) -> dict[str, Any]:
    concept_name, x, only = task
    c: Concept = _W["concepts"][concept_name]
    d = c.make(_W["space"], list(x))
    f = build_finger(d, _W["sc"].keep_out, _W["sc"].closed_limit)
    objs = _W["lib"].objects if only is None else [_W["lib"].by_id(o) for o in only]
    res = [solve_grip(d, o, _W["sc"], finger=f) for o in objs]
    scores = [corrected_score(r, _W["corr"]) for r in res]
    return {
        "concept": concept_name,
        "x": list(map(float, x)),
        "design": d.to_dict(),
        "objects": [r.object_id for r in res],
        "scores": scores,
        "raw_scores": [r.score for r in res],
        "limiting": [r.limiting for r in res],
        "insertion_mm": [r.info.get("insertion_mm", float("nan")) for r in res],
        "J": objective_from_scores(scores, _W["weight"]),
        "n_pass": int(sum(s >= 1.0 for s in scores)),
    }


def _neg_J(x: np.ndarray, concept_name: str) -> float:
    return -_evaluate((concept_name, tuple(x), None))["J"]


def _neg_obj_score(x: np.ndarray, concept_name: str, oid: str) -> float:
    r = _evaluate((concept_name, tuple(x), (oid,)))
    return -min(r["scores"][0], 5.0)


def _simulate_task(task: tuple[dict[str, Any], str, float | None]) -> dict[str, Any]:
    from .mjsim import simulate

    ddict, oid, depth = task
    d = GripperDesign.from_dict(ddict)
    o = _W["lib"].by_id(oid)
    t0 = time.perf_counter()
    an = solve_grip(d, o, _W["sc"], depth=depth)
    if not an.ok:
        # The grasp cannot be set up (no fit, body or fingertip collision, ...).
        out = {"passed": False, "reason": f"not simulated: {an.reason}", "metrics": {}, "score": 0.0,
               "insertion_mm": an.info.get("insertion_mm", float("nan"))}
    else:
        try:
            r = simulate(d, o, _W["sc"], depth=an.info["insertion_mm"] / 1e3, record=False)
            out = {"passed": r.passed, "reason": r.reason, "metrics": r.metrics, "score": r.score, "insertion_mm": r.insertion_mm}
        except Exception as exc:  # a model that fails to build or run counts as a failed grasp
            out = {"passed": False, "reason": f"simulation error: {exc}", "metrics": {}, "score": 0.0, "insertion_mm": float("nan")}
    out.update(
        {
            "design_label": ddict.get("label", ""),
            "concept": ddict["concept"],
            "object_id": oid,
            "analytic_ok": an.ok,
            "analytic_score": an.score,
            "analytic_margins": an.margins,
            "analytic_limiting": an.limiting,
            "seconds": time.perf_counter() - t0,
        }
    )
    return out


# ------------------------------------------------------------------ driver


@dataclass
class SearchRun:
    paths: dict[str, str]
    out_dir: Path
    cfg: SearchConfig
    space: DesignSpace = field(init=False)
    concepts: list[Concept] = field(init=False)
    lib: ObjectLibrary = field(init=False)
    sc: Scenario = field(init=False)
    archive: list[dict[str, Any]] = field(default_factory=list)
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        self.space = load_design_space(self.paths["design_space"])
        self.concepts = load_concepts(self.paths["concepts"], self.space)
        self.lib = load_objects(self.paths["objects"])
        self.sc = load_scenario(self.paths["scenario"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _pool(self, corrections: dict[str, dict[str, float]]) -> ProcessPoolExecutor:
        n = self.cfg.workers or (os.cpu_count() or 1)
        paths = dict(self.paths, weight=str(self.cfg.min_margin_weight))
        return ProcessPoolExecutor(max_workers=n, initializer=_init_worker, initargs=(paths, corrections))

    def run(self) -> dict[str, Any]:
        corrections: dict[str, dict[str, float]] = {}
        rounds = []
        t_start = time.perf_counter()
        seeds: dict[str, np.ndarray] = {}
        for rnd in range(1, self.cfg.rounds + 1):
            self.log(f"== round {rnd}: analytic search" + (" (with dynamic corrections)" if corrections else ""))
            with self._pool(corrections) as pool:
                screen = self._screen_round(pool, rnd, seeds)
                per_obj = self._per_object(pool, rnd, screen)
                finalists = self._finalists(screen, per_obj, rnd)
                self.log(
                    f"   dynamic check: {len(finalists)} designs x {len(self.lib.objects)} objects"
                    + (f" ({len(self._carried)} earlier designs reselected, not rerun)" if self._carried else "")
                )
                dyn = self._dynamic(pool, finalists)
            corrections = self._learn(dyn, corrections)
            summary = self._summarise_round(rnd, screen, per_obj, finalists, dyn, corrections)
            rounds.append(summary)
            seeds = {c: np.array([e["x"] for e in screen[c]["top"]]) for c in screen}
            (self.out_dir / f"round{rnd}.json").write_text(json.dumps(summary, indent=1, default=_json_default))
        result = self._final(rounds)
        result["seconds"] = time.perf_counter() - t_start
        (self.out_dir / "results.json").write_text(json.dumps(result, indent=1, default=_json_default))
        (self.out_dir / "archive.json").write_text(json.dumps(self.archive, default=_json_default))
        return result

    # -- step 1
    def _screen_round(self, pool: ProcessPoolExecutor, rnd: int, seeds: dict[str, np.ndarray]) -> dict[str, Any]:
        out = {}
        for c in self.concepts:
            t0 = time.perf_counter()
            bounds = c.bounds(self.space)
            dim = len(bounds)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            sampler = qmc.LatinHypercube(d=dim, seed=self.cfg.seed + 97 * rnd + zlib.crc32(c.name.encode()) % 1000)
            pts = lo + sampler.random(self.cfg.n_lhs) * (hi - lo)
            nominal = c.to_vector(c.make(self.space))
            pts = np.vstack([nominal[None, :], pts])
            n_seed = len(seeds.get(c.name, []))
            if n_seed:
                pts = np.vstack([seeds[c.name], pts])
            evals = list(pool.map(_evaluate, [(c.name, tuple(p), None) for p in pts], chunksize=8))
            for e in evals:
                e["stage"] = f"r{rnd}-lhs"
            nominal_eval = evals[n_seed]
            evals.sort(key=lambda e: -e["J"])
            n_pop = max(self.cfg.de_popsize * dim, 10)
            init = np.array([e["x"] for e in evals[:n_pop]])
            if len(init) < n_pop:
                init = np.vstack([init, lo + np.random.default_rng(rnd).random((n_pop - len(init), dim)) * (hi - lo)])
            res = differential_evolution(
                _neg_J,
                bounds,
                args=(c.name,),
                init=init,
                maxiter=self.cfg.de_maxiter,
                tol=self.cfg.de_tol,
                seed=self.cfg.seed + rnd,
                workers=pool.map,
                updating="deferred",
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.7,
            )
            pop = np.asarray(res.population)
            final = list(pool.map(_evaluate, [(c.name, tuple(p), None) for p in pop], chunksize=4))
            for e in final:
                e["stage"] = f"r{rnd}-de"
            allv = evals + final
            self.archive.extend(allv)
            allv.sort(key=lambda e: (-e["J"], -e["n_pass"]))
            top = _distinct(allv, self.cfg.finalists_per_concept, bounds)
            out[c.name] = {
                "best": allv[0],
                "top": top,
                "nominal": nominal_eval,
                "n_evals": len(evals) + int(res.nfev) + len(final),
                "de_nit": int(res.nit),
                "lhs_best_J": evals[0]["J"],
                "seconds": time.perf_counter() - t0,
            }
            self.log(
                f"   {c.name:14s} J={allv[0]['J']:.2f} pass={allv[0]['n_pass']}/{len(self.lib.objects)} "
                f"(LHS best {evals[0]['J']:.2f}) evals={out[c.name]['n_evals']} {out[c.name]['seconds']:.0f}s"
            )
        return out

    # -- step 2
    def _per_object(self, pool: ProcessPoolExecutor, rnd: int, screen: dict[str, Any]) -> dict[str, Any]:
        out = {}
        ids = self.lib.ids()
        for j, oid in enumerate(ids):
            best = None
            for e in self.archive:
                k = e["objects"].index(oid) if oid in e["objects"] else None
                if k is None or len(e["objects"]) != len(ids):
                    continue
                s = e["scores"][k]
                if best is None or s > best[0]:
                    best = (s, e)
            if best is None:
                continue
            e0 = best[1]
            c = next(c for c in self.concepts if c.name == e0["concept"])
            bounds = c.bounds(self.space)
            dim = len(bounds)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            rng = np.random.default_rng(self.cfg.seed + 31 * j + rnd)
            n_pop = max(2 * dim, 10)
            x0 = np.array(e0["x"])
            init = np.clip(x0 + rng.normal(0, 0.15, (n_pop, dim)) * (hi - lo), lo, hi)
            init[0] = x0
            res = differential_evolution(
                _neg_obj_score,
                bounds,
                args=(c.name, oid),
                init=init,
                maxiter=self.cfg.per_object_maxiter,
                seed=self.cfg.seed + j,
                workers=pool.map,
                updating="deferred",
                polish=False,
            )
            full = _evaluate_local(pool, c.name, res.x)
            full["stage"] = f"r{rnd}-object-{oid}"
            self.archive.append(full)
            k = full["objects"].index(oid)
            cand = full if full["scores"][k] >= best[0] else e0
            out[oid] = {"entry": cand, "score": cand["scores"][cand["objects"].index(oid)]}
        return out

    # -- step 3
    def _finalists(self, screen: dict[str, Any], per_obj: dict[str, Any], rnd: int) -> list[dict[str, Any]]:
        """Distinct designs for the dynamic check. A design already checked in an
        earlier round is not run again; its new reasons go to `self._carried`."""
        fin: list[dict[str, Any]] = []
        seen = set()
        checked: dict[tuple, str] = getattr(self, "_checked", {})
        self._carried: dict[str, list[str]] = {}

        def add(e: dict[str, Any], why: str, name: str | None = None) -> None:
            key = tuple(np.round(e["x"], 4)) + (e["concept"],)
            if key in checked:
                self._carried.setdefault(checked[key], []).append(why)
                return
            if key in seen:
                for f in fin:
                    if tuple(np.round(f["x"], 4)) + (f["concept"],) == key:
                        f["why"].append(why)
                return
            seen.add(key)
            g = dict(e)
            g["why"] = [why]
            g["name"] = name or f"r{rnd}-{len(fin) + 1:02d}"
            g["design"] = dict(e["design"], label=g["name"])
            fin.append(g)

        if rnd == 1:
            refs = [c for c in self.concepts if c.reference]
            for c in refs:
                add(screen[c.name]["nominal"], f"{c.label} as drawn (reference)", "baseline" if len(refs) == 1 else f"ref-{c.name}")
        for cname, s in screen.items():
            for i, e in enumerate(s["top"]):
                add(e, f"{cname} top {i + 1}")
        for oid, p in per_obj.items():
            add(p["entry"], f"best for {oid}")
        for f in fin:
            checked[tuple(np.round(f["x"], 4)) + (f["concept"],)] = f["name"]
        self._checked = checked
        return fin

    def _dynamic(self, pool: ProcessPoolExecutor, finalists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = []
        for f in finalists:
            for k, oid in enumerate(self.lib.ids()):
                depth = f["insertion_mm"][k] / 1e3 if k < len(f["insertion_mm"]) and math.isfinite(f["insertion_mm"][k]) else None
                tasks.append((f["design"], oid, depth))
        t0 = time.perf_counter()
        res = list(pool.map(_simulate_task, tasks, chunksize=1))
        self.log(f"   dynamic check done: {len(res)} runs in {time.perf_counter() - t0:.0f}s")
        for r, (dd, _, _) in zip(res, tasks):
            r["design_name"] = dd["label"]
        return res

    # -- step 4
    def _learn(self, dyn: list[dict[str, Any]], prev: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Correction factors (dynamic / analytic margin) per object and criterion, from every
        dynamic run so far. The 25th percentile keeps the screen on the cautious side."""
        self._dyn_all = getattr(self, "_dyn_all", []) + dyn
        ratios: dict[str, dict[str, list[float]]] = {}
        for r in self._dyn_all:
            if not r["analytic_ok"] or not r["metrics"].get("margins"):
                continue
            am = r["analytic_margins"]
            dyn_margins = dict(r["metrics"]["margins"])
            if "friction_margin" in r["metrics"]:
                dyn_margins["friction"] = r["metrics"]["friction_margin"]
            for crit, m_dyn in dyn_margins.items():
                for a_crit in CRITERION_MAP.get(crit, ()):
                    if a_crit in am and math.isfinite(am[a_crit]) and am[a_crit] > 1e-6 and m_dyn < 50:
                        ratios.setdefault(r["object_id"], {}).setdefault(a_crit, []).append(m_dyn / am[a_crit])
                        break
        lo, hi = self.cfg.correction_clip
        out: dict[str, dict[str, float]] = {}
        for oid, crits in ratios.items():
            for crit, vals in crits.items():
                if len(vals) >= 3:
                    out.setdefault(oid, {})[crit] = float(np.clip(np.quantile(vals, 0.25), lo, hi))
        return out

    def _summarise_round(self, rnd, screen, per_obj, finalists, dyn, corrections) -> dict[str, Any]:
        ids = self.lib.ids()
        table: dict[str, dict[str, Any]] = {}
        for r in dyn:
            table.setdefault(r["design_name"], {})[r["object_id"]] = r
        designs = []
        for f in finalists:
            row = table.get(f["name"], {})
            n_pass = sum(1 for o in ids if row.get(o, {}).get("passed"))
            worst = min((min(row[o]["score"], 2.0) if row.get(o, {}).get("passed") else 0.0) for o in ids) if row else 0.0
            sim_scores = [min(row.get(o, {}).get("score", 0.0), 2.0) if row.get(o, {}).get("passed") else min(row.get(o, {}).get("score", 0.0), 0.99) for o in ids]
            designs.append(
                {
                    "name": f["name"],
                    "concept": f["concept"],
                    "why": f["why"],
                    "design": f["design"],
                    "analytic_J": f["J"],
                    "analytic_pass": f["n_pass"],
                    "sim_pass": n_pass,
                    "sim_J": objective_from_scores(sim_scores, self.cfg.min_margin_weight),
                    "sim_worst": worst,
                    "results": {o: _slim(row.get(o)) for o in ids},
                }
            )
        designs.sort(key=lambda d: (-d["sim_pass"], -d["sim_J"], -d["analytic_J"]))
        return {
            "round": rnd,
            "concepts": {
                c: {
                    "best_J": s["best"]["J"],
                    "best_pass": s["best"]["n_pass"],
                    "lhs_best_J": s["lhs_best_J"],
                    "n_evals": s["n_evals"],
                    "seconds": s["seconds"],
                    "best_design": s["best"]["design"],
                }
                for c, s in screen.items()
            },
            "per_object_analytic": {o: {"score": p["score"], "concept": p["entry"]["concept"]} for o, p in per_obj.items()},
            "designs": designs,
            "carried": dict(self._carried),
            "corrections": corrections,
        }

    def _final(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        ids = self.lib.ids()
        all_designs = [d for r in rounds for d in r["designs"]]
        by_name = {d["name"]: d for d in all_designs}
        for r in rounds:
            for name, whys in r.get("carried", {}).items():
                by_name[name]["why"] = by_name[name]["why"] + [f"{w} (round {r['round']}, no better design found)" for w in whys]
        all_designs.sort(key=lambda d: (-d["sim_pass"], -d["sim_J"], -d["analytic_J"]))
        best = all_designs[0]
        per_object = {}
        for o in ids:
            cands = [(d, d["results"][o]) for d in all_designs if d["results"].get(o)]
            cands.sort(key=lambda t: (-(1 if t[1]["passed"] else 0), -t[1]["score"]))
            if cands:
                d, r = cands[0]
                per_object[o] = {"design": d["name"], "concept": d["concept"], "passed": r["passed"], "score": r["score"], "reason": r["reason"]}
        best_by_concept = {}
        for d in all_designs:
            best_by_concept.setdefault(d["concept"], d["name"])
        return {
            "objects": ids,
            "references": [d["name"] for d in all_designs if any(w.endswith("(reference)") for w in d["why"])],
            "all_rounder": best["name"],
            "per_object": per_object,
            "best_by_concept": best_by_concept,
            "designs": {d["name"]: d for d in all_designs},
            "ranking": [d["name"] for d in all_designs],
            "rounds": [{k: v for k, v in r.items() if k != "designs"} | {"ranking": [d["name"] for d in r["designs"]]} for r in rounds],
        }


def _evaluate_local(pool: ProcessPoolExecutor, cname: str, x: np.ndarray) -> dict[str, Any]:
    return next(iter(pool.map(_evaluate, [(cname, tuple(x), None)])))


def _distinct(evals: list[dict[str, Any]], k: int, bounds: list[tuple[float, float]], min_dist: float = 0.08) -> list[dict[str, Any]]:
    """Top-k evaluations that differ by at least min_dist (normalised)."""
    lo = np.array([b[0] for b in bounds])
    span = np.maximum(np.array([b[1] for b in bounds]) - lo, 1e-9)
    picked: list[dict[str, Any]] = []
    for e in evals:
        x = (np.array(e["x"]) - lo) / span
        if all(np.linalg.norm(x - (np.array(p["x"]) - lo) / span) / math.sqrt(len(x)) >= min_dist for p in picked):
            picked.append(e)
        if len(picked) >= k:
            break
    return picked


def _slim(r: dict[str, Any] | None) -> dict[str, Any] | None:
    if r is None:
        return None
    m = r.get("metrics", {})
    return {
        "passed": r["passed"],
        "reason": r["reason"],
        "score": r["score"],
        "insertion_mm": r.get("insertion_mm"),
        "margins": m.get("margins", {}),
        "slip_mm": m.get("slip_mm"),
        "rotation_deg": m.get("rotation_deg"),
        "peak_pressure_kPa": m.get("peak_pressure_kPa"),
        "panel_force_N": m.get("panel_force_N"),
        "crease_moment_Nm": m.get("crease_moment_Nm"),
        "grip_N": m.get("grip_N"),
        "friction_use": m.get("friction_use"),
        "closing_impact_pressure_kPa": m.get("closing_impact_pressure_kPa"),
        "analytic_score": r["analytic_score"],
        "analytic_limiting": r["analytic_limiting"],
        "analytic_margins": r["analytic_margins"],
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))
