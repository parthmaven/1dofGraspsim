"""Parameter sweeps: one base design, a grid over chosen variables, every object.

Each `--vary` spec gives one variable's values (`name=a,b,c` or `name=lo:hi:n`);
the grid is their Cartesian product, applied on top of the base design. Every
point runs on the selected objects at either fidelity on all cores. A sweep
directory holds the inputs, `sweep.json` (spec, points and every outcome),
`sweep.csv` (one row per point and object) and `summary.csv` (one row per
point).
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .schema import SCHEMA_BY_NAME, GripperDesign
from .simulator import run_batch

METRIC_COLUMNS = ("normal_force_N", "grip_N", "peak_pressure_kPa", "slip_mm", "rotation_deg", "panel_force_N",
                  "body_contact_N", "crease_moment_Nm")


def parse_vary(spec: str) -> tuple[str, list[float]]:
    """`name=a,b,c` (listed values) or `name=lo:hi:n` (n evenly spaced values, ends included)."""
    if "=" not in spec:
        raise ValueError(f"--vary {spec!r}: expected name=a,b,c or name=lo:hi:n")
    name, vals = (s.strip() for s in spec.split("=", 1))
    if name not in SCHEMA_BY_NAME:
        raise ValueError(f"--vary {spec!r}: unknown design variable {name!r}")
    if ":" in vals:
        parts = vals.split(":")
        if len(parts) != 3 or int(parts[2]) < 1:
            raise ValueError(f"--vary {spec!r}: a range is lo:hi:n with n >= 1")
        values = np.linspace(float(parts[0]), float(parts[1]), int(parts[2])).tolist()
    else:
        values = [float(v) for v in vals.split(",") if v.strip()]
    if not values:
        raise ValueError(f"--vary {spec!r}: no values")
    return name, [round(v, 6) for v in values]


def grid(base: GripperDesign, vary: Sequence[tuple[str, Sequence[float]]]) -> list[tuple[dict[str, float], GripperDesign]]:
    """Every combination of the varied values, applied to the base design."""
    names = [n for n, _ in vary]
    if len(set(names)) != len(names):
        raise ValueError(f"a variable is varied twice: {names}")
    stem = base.label or base.concept
    points = []
    for combo in itertools.product(*(vals for _, vals in vary)):
        values = dict(zip(names, combo))
        tag = " ".join(f"{n}={v:g}" for n, v in values.items())
        points.append((values, GripperDesign.from_dict({**base.to_dict(), **values, "label": f"{stem} {tag}".strip()})))
    return points


def run_sweep(base: GripperDesign, vary: Sequence[tuple[str, Sequence[float]]], object_ids: Sequence[str],
              paths: dict[str, str], fidelity: str = "analytic", workers: int | None = None, weight: float = 0.25,
              progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    from .search import objective_from_scores

    points = grid(base, vary)
    rows = run_batch(fidelity, [d for _, d in points], object_ids, paths["objects"], paths["scenario"], workers, progress)
    summary = []
    for i, (values, _) in enumerate(points):
        rs = [r for r in rows if r["point"] == i]
        # a run left unscored (MuJoCo blew up at every timestep) is neither a pass nor a fail
        scored = [r for r in rs if r["passed"] is not None]
        worst = min(scored, key=lambda r: r["score"]) if scored else None
        summary.append({
            "point": i,
            **values,
            "n_pass": sum(1 for r in rs if r["passed"]),
            "n_objects": len(rs),
            "n_unscored": len(rs) - len(scored),
            "J": objective_from_scores([r["score"] for r in scored], weight) if scored else math.nan,
            "min_score": worst["score"] if worst else math.nan,
            "worst_object": worst["object_id"] if worst else "",
            "worst_reason": (worst["reason"] or worst["limiting"]) if worst else "",
        })
    return {
        "fidelity": fidelity,
        "base": base.to_dict(),
        "vary": [{"name": n, "values": list(v)} for n, v in vary],
        "objects": list(object_ids),
        "points": [{"point": i, "values": v, "design": d.to_dict()} for i, (v, d) in enumerate(points)],
        "summary": summary,
        "outcomes": rows,
    }


def clean(o: Any) -> Any:
    """JSON-safe copy: non-finite floats become null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_outcomes_csv(path: Path, rows: Sequence[dict[str, Any]], lead: Sequence[str] = ()) -> None:
    """One row per outcome: `lead` columns (e.g. varied variables), result, margins, key metrics."""
    margin_keys = sorted({k for r in rows for k in r["margins"]})
    metric_keys = [k for k in METRIC_COLUMNS if any(k in r["metrics"] for r in rows)]
    cols = ["point", *lead, "object_id", "fidelity", "passed", "score", "limiting", "reason", "insertion_mm",
            *(f"margin_{k}" for k in margin_keys), *metric_keys, "seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **r.get("values", {}), **{f"margin_{k}": v for k, v in r["margins"].items()},
                        **{k: r["metrics"][k] for k in metric_keys if k in r["metrics"]}})


def save_sweep(out: Path, result: dict[str, Any]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    names = [v["name"] for v in result["vary"]]
    values = {p["point"]: p["values"] for p in result["points"]}
    (out / "sweep.json").write_text(json.dumps(clean(result), indent=1))
    write_outcomes_csv(out / "sweep.csv", [{**r, "values": values[r["point"]]} for r in result["outcomes"]], names)
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result["summary"][0]))
        w.writeheader()
        w.writerows(result["summary"])
    return [out / "sweep.json", out / "sweep.csv", out / "summary.csv"]
