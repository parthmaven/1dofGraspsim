"""One interface over both simulation levels.

`make_simulator("analytic")` is the quasi-static screen (analytic.solve_grip);
`make_simulator("mujoco")` is the planar dynamic check (mjsim.simulate). Both
take a design, an object and a scenario and return a GraspOutcome with the same
fields, so the CLI, sweeps and the search handle either level the same way.
`run_batch` spreads designs x objects over worker processes.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .analytic import GripResult, solve_grip
from .geometry import Finger, build_finger
from .objects import GraspObject, load_objects
from .scenario import Scenario, load_scenario
from .schema import GripperDesign

FIDELITIES = ("analytic", "mujoco")


@dataclass
class GraspOutcome:
    """Result of one grasp (design on object) at either level."""

    fidelity: str
    design: str
    object_id: str
    passed: bool
    score: float  # smallest pass-criterion margin; >= 1 passes, 0 when the grasp cannot be set up
    limiting: str  # criterion with the smallest margin, or why the grasp failed
    reason: str  # empty when passed
    margins: dict[str, float]
    metrics: dict[str, float]
    insertion_mm: float
    seconds: float
    screen: dict[str, Any] = field(default_factory=dict)  # mujoco: the analytic screen at the same depth
    trace: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("trace")
        return d


class Simulator(Protocol):
    fidelity: str

    def run(self, design: GripperDesign, obj: GraspObject, sc: Scenario, depth: float | None = None,
            finger: Finger | None = None) -> GraspOutcome: ...


def _numbers(info: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in info.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _screen_summary(r: GripResult) -> dict[str, Any]:
    return {"ok": r.ok, "score": r.score, "margins": dict(r.margins), "limiting": r.limiting,
            "approach_opening_deg": r.info.get("approach_opening_deg"), "approach_stroke_deg": r.info.get("approach_stroke_deg")}


class AnalyticSimulator:
    """Quasi-static screen: fit, contact, hold, pitch, pressure, crease. Milliseconds per grasp."""

    fidelity = "analytic"

    def run(self, design: GripperDesign, obj: GraspObject, sc: Scenario, depth: float | None = None,
            finger: Finger | None = None) -> GraspOutcome:
        t0 = time.perf_counter()
        r = solve_grip(design, obj, sc, depth=depth, finger=finger)
        reason = "" if r.passed else (r.reason if not r.ok else f"{r.limiting} margin {r.score:.2f}")
        return GraspOutcome(self.fidelity, design.label or design.concept, obj.id, r.passed, r.score, r.limiting, reason,
                            dict(r.margins), _numbers(r.info), float(r.info.get("insertion_mm", math.nan)),
                            time.perf_counter() - t0)


class MujocoSimulator:
    """Planar MuJoCo check: close, lift, hold, shake. Seconds per grasp.

    The insertion depth defaults to the one the screen picks; a grasp the screen
    cannot set up (no fit, a collision, an infeasible finger) is not simulated."""

    fidelity = "mujoco"

    def __init__(self, record: bool = False):
        self.record = record

    def run(self, design: GripperDesign, obj: GraspObject, sc: Scenario, depth: float | None = None,
            finger: Finger | None = None) -> GraspOutcome:
        from .mjsim import simulate

        t0 = time.perf_counter()
        label = design.label or design.concept
        f = finger or build_finger(design, sc.keep_out, sc.closed_limit)
        an = solve_grip(design, obj, sc, depth=depth, finger=f)
        screen = _screen_summary(an)
        if not an.ok:
            return GraspOutcome(self.fidelity, label, obj.id, False, 0.0, an.reason, f"not simulated: {an.reason}", {}, {},
                                float(an.info.get("insertion_mm", math.nan)), time.perf_counter() - t0, screen)
        try:
            r = simulate(design, obj, sc, depth=an.info["insertion_mm"] / 1e3, finger=f, record=self.record)
        except Exception as exc:  # a model that fails to build or run counts as a failed grasp
            return GraspOutcome(self.fidelity, label, obj.id, False, 0.0, "simulation error", f"simulation error: {exc}",
                                {}, {}, math.nan, time.perf_counter() - t0, screen)
        margins = dict(r.metrics.get("margins", {}))
        metrics = {k: float(v) for k, v in r.metrics.items() if k != "margins"}
        limiting = min(margins, key=margins.get) if margins else r.reason
        return GraspOutcome(self.fidelity, label, obj.id, r.passed, r.score, limiting, r.reason, margins, metrics,
                            r.insertion_mm, time.perf_counter() - t0, screen, r.trace if self.record else {})


def make_simulator(fidelity: str, **kwargs: Any) -> Simulator:
    if fidelity == "analytic":
        return AnalyticSimulator()
    if fidelity == "mujoco":
        return MujocoSimulator(**kwargs)
    raise ValueError(f"unknown fidelity {fidelity!r}; choose from {FIDELITIES}")


# ------------------------------------------------------------- batches

_B: dict[str, Any] = {}


def _init_batch(fidelity: str, objects_path: str, scenario_path: str) -> None:
    _B["sim"] = make_simulator(fidelity)
    _B["lib"] = load_objects(objects_path)
    _B["sc"] = load_scenario(scenario_path)


def _run_job(job: tuple[int, dict[str, Any], tuple[str, ...]]) -> list[dict[str, Any]]:
    i, ddict, oids = job
    d = GripperDesign.from_dict(ddict)
    f = build_finger(d, _B["sc"].keep_out, _B["sc"].closed_limit)
    return [{"point": i, **_B["sim"].run(d, _B["lib"].by_id(o), _B["sc"], finger=f).to_dict()} for o in oids]


def run_batch(fidelity: str, designs: Sequence[GripperDesign], object_ids: Sequence[str], objects_path: str,
              scenario_path: str, workers: int | None = None,
              progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    """Every design on every object, on `workers` processes (default: all cores).

    Returns one dict per (design, object), tagged with the design's index as
    `point`, in design-then-object order. Analytic jobs go one design per task
    (the finger is built once); MuJoCo jobs go one grasp per task."""
    if fidelity not in FIDELITIES:
        raise ValueError(f"unknown fidelity {fidelity!r}; choose from {FIDELITIES}")
    per_design = fidelity == "analytic"
    jobs = [(i, d.to_dict(), tuple(object_ids) if per_design else (o,))
            for i, d in enumerate(designs) for o in (object_ids[:1] if per_design else object_ids)]
    workers = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    out: list[dict[str, Any]] = []
    if workers == 1:
        _init_batch(fidelity, objects_path, scenario_path)
        for k, job in enumerate(jobs):
            out.extend(_run_job(job))
            if progress:
                progress(k + 1, len(jobs))
        return out
    with ProcessPoolExecutor(workers, initializer=_init_batch, initargs=(fidelity, objects_path, scenario_path)) as ex:
        for k, rows in enumerate(ex.map(_run_job, jobs, chunksize=max(1, len(jobs) // (8 * workers)))):
            out.extend(rows)
            if progress:
                progress(k + 1, len(jobs))
    return out
