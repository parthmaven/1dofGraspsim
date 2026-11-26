"""Figures for a search run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from .analytic import solve_grip  # noqa: E402
from .geometry import build_finger, finger_outline_view  # noqa: E402
from .objects import GraspObject, ObjectLibrary  # noqa: E402
from .scenario import Scenario  # noqa: E402
from .schema import SCHEMA_BY_NAME, Concept, DesignSpace, GripperDesign  # noqa: E402

CONCEPT_COLORS = {
    "baseline_paddle": "#4c72b0",
    "straight_flat": "#4c72b0",
    "cranked_flat": "#dd8452",
    "crowned": "#55a868",
    "dual_zone": "#c44e52",
    "self_aligning": "#8172b3",
}
BOX_COLOR = "#e8c37a"
SHEET_COLOR = "#a8763e"
PAD_COLOR = "#2b2b2b"
PASS, FAIL = "#2e8b57", "#c0392b"
MIN_SHEET_DRAW_MM = 1.5


def _color(concept: str) -> str:
    return CONCEPT_COLORS.get(concept, "#777777")


def draw_keep_out(ax, sc: Scenario | None) -> None:
    """Bracket footprint the fingers and objects must clear."""
    if sc is None or sc.keep_out is None:
        return
    for x, z, r in sc.keep_out.circles:
        ax.add_patch(Circle((x, z), r, fc="#f4c7c3", ec="#c0392b", lw=0.6, ls="--", hatch="////", alpha=0.55, zorder=3))


def draw_gripper(ax, design: GripperDesign, phi: float, color: str, alpha: float = 1.0, lw: float = 0.8,
                 sc: Scenario | None = None) -> None:
    f = build_finger(design, sc.keep_out, sc.closed_limit) if sc is not None else build_finger(design)
    for side in (1, -1):
        parts = finger_outline_view(f, phi, side)
        for P in parts["body"]:
            ax.add_patch(Polygon(P, closed=True, fc=color, ec="k", lw=lw * 0.6, alpha=alpha))
        for P in parts["pads"]:
            ax.add_patch(Polygon(P, closed=True, fc=PAD_COLOR, ec="k", lw=lw * 0.4, alpha=alpha))
    ax.plot([-design.pivot_spacing / 2, design.pivot_spacing / 2], [0, 0], "o", ms=3, color="k", alpha=alpha)
    hub = Rectangle((-design.pivot_spacing / 2 - 8, -2), design.pivot_spacing + 16, 14, fc="#bbbbbb", ec="k", lw=0.5, alpha=alpha)
    ax.add_patch(hub)
    draw_keep_out(ax, sc)


def draw_object(ax, obj: GraspObject, z_top: float, below: float) -> None:
    """Object section in gripper coordinates (mm), clipped `below` mm under the top."""
    B = obj.thickness * 1000
    if obj.kind == "box":
        h = min(obj.height * 1000, below)
        ax.add_patch(Rectangle((-B / 2, z_top - h), B, h, fc=BOX_COLOR, ec="k", lw=0.8, zorder=0))
        zf = z_top - obj.headspace * 1000
        if obj.headspace > 0 and zf > z_top - h:
            ax.plot([-B / 2, B / 2], [zf, zf], ls="--", lw=0.8, color="#7a5c1e", zorder=1)
    else:
        Bd = max(B, MIN_SHEET_DRAW_MM)
        h = min(obj.height * 1000, below)
        ax.add_patch(Rectangle((-Bd / 2, z_top - h), Bd, h, fc=SHEET_COLOR, ec="k", lw=0.5, zorder=0))


def draw_grasp(ax, design: GripperDesign, obj: GraspObject, sc: Scenario, sim: dict[str, Any] | None = None,
               title: str = "", depth_mm: float | None = None) -> None:
    an = solve_grip(design, obj, sc, depth=depth_mm / 1e3 if depth_mm else None)
    d_mm = an.info.get("insertion_mm", depth_mm or sc.for_kind(obj.kind).insertion_depth * 1e3)
    L = design.finger_reach
    z_top = -L + d_mm
    phi = math.radians(an.info["phi_deg"]) if an.ok else math.radians(design.max_opening)
    below = L - d_mm + 70
    draw_object(ax, obj, z_top, below)
    draw_gripper(ax, design, phi, _color(design.concept), sc=sc)
    ax.set_aspect("equal")
    half = max(design.pivot_spacing / 2 + 60, obj.thickness * 500 + 30, 95)
    ax.set_xlim(-half, half)
    ax.set_ylim(-L - 45, 25)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    status, color = _status(sim, an)
    ax.set_title(title, fontsize=8)
    ax.text(0.5, 0.02, status, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))


def _status(sim: dict[str, Any] | None, an) -> tuple[str, str]:
    if sim is not None:
        if sim.get("passed"):
            worst = min(sim["margins"], key=sim["margins"].get) if sim.get("margins") else ""
            return f"PASS  min margin {sim['score']:.2f} ({worst})", PASS
        reason = sim.get("reason", "fail").replace("not simulated: ", "")
        return f"FAIL  {reason}"[:48], FAIL
    if an.ok:
        return (f"screen {an.score:.2f} ({an.limiting})", PASS if an.passed else FAIL)
    return f"FAIL  {an.reason}"[:48], FAIL


def fig_best_designs(path: Path, result: dict[str, Any], lib: ObjectLibrary, sc: Scenario) -> None:
    """All-rounder and per-object best on every object."""
    ids = lib.ids()
    boxes = [o for o in lib.objects if o.kind == "box"]
    sheets = [o for o in lib.objects if o.kind == "sheet"]
    ncol = max(len(boxes), len(sheets))
    fig, axes = plt.subplots(4, ncol, figsize=(2.6 * ncol, 4.2 * 4))
    designs = result["designs"]
    ar = designs[result["all_rounder"]]
    for block, objs in ((0, boxes), (2, sheets)):
        for j in range(ncol):
            for r in (0, 1):
                ax = axes[block + r, j]
                if j >= len(objs):
                    ax.axis("off")
                    continue
                o = objs[j]
                if r == 0:
                    d = ar
                    who = f"all-rounder {d['name']}"
                else:
                    po = result["per_object"].get(o.id)
                    d = designs[po["design"]] if po else ar
                    who = f"best for object: {d['name']}"
                gd = GripperDesign.from_dict(d["design"])
                sim = d["results"].get(o.id)
                draw_grasp(ax, gd, o, sc, sim=sim, title=f"{o.label}\n{who} ({_short(d['concept'])})",
                           depth_mm=(sim or {}).get("insertion_mm"))
    fig.suptitle(
        "First pass: best all-rounder (rows 1, 3) and best design per object (rows 2, 4), closed on each object\n"
        "dynamic check result under each panel; dashed line = settled fill level (panel above it is unsupported); "
        f"sheets drawn at least {MIN_SHEET_DRAW_MM} mm thick\n{KEEP_OUT_NOTE}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=110)
    plt.close(fig)


KEEP_OUT_NOTE = "hatched circle = FT Dynamic bracket (top) keep-out; each finger is cut to clear it over its whole stroke"


def fig_design_on_objects(path: Path, result: dict[str, Any], name: str, lib: ObjectLibrary, sc: Scenario) -> None:
    """One design (e.g. the baseline as drawn) closed on every object, with its dynamic check results."""
    d = result["designs"][name]
    gd = GripperDesign.from_dict(d["design"])
    boxes = [o for o in lib.objects if o.kind == "box"]
    sheets = [o for o in lib.objects if o.kind == "sheet"]
    ncol = max(len(boxes), len(sheets))
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 3.4 * 2 + 1.0))
    for row, objs in enumerate((boxes, sheets)):
        for j in range(ncol):
            ax = axes[row, j]
            if j >= len(objs):
                ax.axis("off")
                continue
            o = objs[j]
            sim = d["results"].get(o.id)
            draw_grasp(ax, gd, o, sc, sim=sim, title=o.label, depth_mm=(sim or {}).get("insertion_mm"))
    fig.suptitle(
        f"{d['why'][0].replace(' (reference)', '')} ({name}), closed on each object; "
        f"passes {d['sim_pass']}/{len(lib.objects)} in the dynamic check\n"
        f"dynamic check result under each panel; dashed line = settled fill level; {KEEP_OUT_NOTE}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _short(concept: str) -> str:
    return {"baseline_paddle": "C1 baseline", "straight_flat": "C1 straight", "cranked_flat": "C2 cranked", "crowned": "C3 crowned",
            "dual_zone": "C4 dual-zone", "self_aligning": "C5 self-aligning"}.get(concept, concept)


def fig_concepts(path: Path, concepts: list[Concept], space: DesignSpace, lib: ObjectLibrary, sc: Scenario,
                 designs: dict[str, GripperDesign] | None = None,
                 sims: dict[str, dict[str, Any]] | None = None) -> None:
    """Each concept: open, and closed on a box and on a sheet. `sims` (concept ->
    object id -> dynamic result) labels the closed panels with the MuJoCo result
    instead of the analytic screen."""
    box = max((o for o in lib.objects if o.kind == "box"), key=lambda o: o.mass)
    sheet = next(o for o in lib.objects if o.kind == "sheet")
    fig, axes = plt.subplots(3, len(concepts), figsize=(3.0 * len(concepts), 13))
    for j, c in enumerate(concepts):
        d = (designs or {}).get(c.name) or c.make(space)
        ax = axes[0, j]
        draw_gripper(ax, d, math.radians(d.max_opening), _color(c.name), sc=sc)
        ax.set_aspect("equal")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-d.finger_reach - 30, 25)
        ax.axis("off")
        ax.set_title(f"{c.label}\nopen {d.max_opening:.0f} deg", fontsize=9)
        ax.text(0.5, -0.02, _wrap(c.idea, 38), transform=ax.transAxes, ha="center", va="top", fontsize=7)
        for row, o in ((1, box), (2, sheet)):
            sim = ((sims or {}).get(c.name) or {}).get(o.id)
            draw_grasp(axes[row, j], d, o, sc, sim=sim, title=o.label, depth_mm=(sim or {}).get("insertion_mm"))
    which = "best first-pass design" if designs else "nominal design; C1 is the baseline as drawn"
    source = "MuJoCo result" if sims else "analytic screen result"
    fig.suptitle(f"Finger concepts ({which} of each); {source} under each closed panel\n{KEEP_OUT_NOTE}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:7])


def fig_objects(path: Path, lib: ObjectLibrary) -> None:
    boxes = [o for o in lib.objects if o.kind == "box"]
    sheets = [o for o in lib.objects if o.kind == "sheet"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    x = 0.0
    for o in boxes:
        B, H = o.thickness * 1000, o.height * 1000
        a1.add_patch(Rectangle((x, 0), B, H, fc=BOX_COLOR, ec="k"))
        zf = H - o.headspace * 1000
        a1.plot([x, x + B], [zf, zf], "--", color="#7a5c1e", lw=0.8)
        a1.text(x + B / 2, 6, f"{o.label}  |  {o.mass * 1000:.0f} g, D {B:.0f} mm, H {H:.0f} mm", ha="center", va="bottom",
                rotation=90, fontsize=7)
        x += B + 25
    a1.set_xlim(-10, x)
    a1.set_ylim(0, 380)
    a1.set_aspect("equal")
    a1.set_title("Cereal boxes: section seen by the jaw (depth x height, to scale); dashed = settled fill line", fontsize=9)
    a1.set_xticks([])
    a1.set_ylabel("mm")
    rows = []
    for o in sheets:
        m = o.material
        rows.append([o.label, f"{o.thickness * 1000:.2f}", f"{m.grammage * 1000:.0f}", f"{o.mass * 1000:.0f}", f"{m.friction:.2f}",
                     f"{m.allow_pressure / 1000:.0f}", f"{m.bending_stiffness:.3g}", f"{m.bending_strength:.3g}"])
    a2.axis("off")
    t = a2.table(cellText=rows, colLabels=["sheet", "mm", "g/m2", "g", "mu", "allow\nkPa", "D\nN*m", "crease\nN*m/m"], loc="center",
                 cellLoc="center", colWidths=[0.36, 0.08, 0.08, 0.07, 0.07, 0.09, 0.09, 0.11])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.8)
    a2.set_title(f"Sheets: {sheets[0].height * 1000:.0f} mm hanging x {sheets[0].width * 1000:.0f} mm wide, pinched near the top edge"
                 if sheets else "Sheets", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_pass_matrix(path: Path, result: dict[str, Any], lib: ObjectLibrary) -> None:
    ids = lib.ids()
    names = result["ranking"]
    designs = result["designs"]
    M = np.zeros((len(names), len(ids)))
    P = np.zeros_like(M, dtype=bool)
    for i, n in enumerate(names):
        for j, o in enumerate(ids):
            r = designs[n]["results"].get(o)
            if r:
                M[i, j] = min(r["score"], 2.0)
                P[i, j] = r["passed"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(ids), 1.4 + 0.3 * len(names)))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    for i in range(len(names)):
        for j in range(len(ids)):
            ax.text(j, i, "P" if P[i, j] else f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="k", fontweight="bold" if P[i, j] else "normal")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([lib.by_id(o).label for o in ids], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} {_short(designs[n]['concept'])} ({designs[n]['sim_pass']}/{len(ids)})" for n in names], fontsize=7)
    ax.set_title("Dynamic check: smallest pass margin per design and object (P = pass, colour capped at 2)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_iteration(path: Path, archive: list[dict[str, Any]], result: dict[str, Any], n_objects: int) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    by_c: dict[str, list[float]] = {}
    for e in archive:
        if len(e["objects"]) == n_objects:
            by_c.setdefault(e["concept"], []).append(e["J"])
    for c, js in by_c.items():
        best = np.maximum.accumulate(np.asarray(js))
        a1.plot(best, color=_color(c), label=_short(c))
    a1.axhline(n_objects, ls=":", color="k", lw=0.8)
    a1.text(0, n_objects + 0.1, "all objects pass (screen)", fontsize=7)
    a1.set_xlabel("archived designs per concept, in order: Latin hypercube, then the evolved population, round by round")
    a1.set_ylabel("best all-rounder objective J (screen)")
    a1.set_title("Analytic search progress (round 2 includes the learned corrections)", fontsize=10)
    a1.legend(fontsize=8)
    rounds = result["rounds"]
    designs = result["designs"]
    w = 0.8 / max(len(rounds), 1)
    concepts = list(CONCEPT_COLORS)
    so_far: list[str] = []
    for k, r in enumerate(rounds):
        so_far += r["ranking"]
        vals = []
        for c in concepts:
            best = max((designs[n]["sim_pass"] for n in so_far if designs[n]["concept"] == c), default=0)
            vals.append(best)
        a2.bar(np.arange(len(concepts)) + k * w, vals, width=w, label=f"round {r['round']}",
               color=[_color(c) for c in concepts], alpha=0.5 + 0.5 * k / max(len(rounds) - 1, 1), ec="k")
    a2.set_xticks(np.arange(len(concepts)) + w * (len(rounds) - 1) / 2)
    a2.set_xticklabels([_short(c) for c in concepts], fontsize=8)
    a2.axhline(n_objects, ls=":", color="k", lw=0.8)
    a2.set_ylabel("objects passed in the dynamic check\n(best design of the concept so far)")
    a2.set_title("Dynamic check by concept, after each round (lighter = earlier round)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_bounds(path: Path, result: dict[str, Any], space: DesignSpace) -> None:
    """Where the leading designs sit inside the bounding box."""
    names = [result["all_rounder"]] + [v["design"] for v in result["per_object"].values()]
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    var = [v for v in SCHEMA_BY_NAME if not space[v].fixed]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for n in uniq:
        d = result["designs"][n]["design"]
        y = [(d[v] - space[v].lower) / (space[v].upper - space[v].lower) for v in var]
        lw = 2.6 if n == result["all_rounder"] else 1.0
        ax.plot(range(len(var)), y, "-o", ms=3, lw=lw, color=_color(result["designs"][n]["concept"]), alpha=0.85,
                label=f"{n} {_short(result['designs'][n]['concept'])}" + (" (all-rounder)" if n == result["all_rounder"] else ""))
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels([f"{v}\n[{space[v].lower:g}, {space[v].upper:g}] {SCHEMA_BY_NAME[v].unit}" for v in var], rotation=70,
                       ha="right", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("position in the placeholder range (0 = lower, 1 = upper)")
    ax.set_title("Leading designs inside the bounding box (variables a concept does not use sit at their nominal)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_traces(path: Path, traces: dict[str, tuple[str, dict[str, np.ndarray]]], sc: Scenario) -> None:
    """Time histories from recorded dynamic runs: {object label: (design name, trace)}."""
    keys = ["grip_N", "rel_z", "rel_th", "friction_use", "p_max", "hinge_M"]
    labels = ["grip force per finger (N)", "slip of grip point (mm)", "object rotation (deg)",
              "share of available\nfriction used (worst pad)", "peak pad pressure (kPa)", "sheet bending moment at pinch (N*m)"]
    scale = {"grip_N": 1, "rel_z": 1e3, "rel_th": 180 / math.pi, "friction_use": 1, "p_max": 1e-3, "hinge_M": 1}
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.1 * len(keys)), sharex=True)
    for lab, (dname, tr) in traces.items():
        if not len(tr.get("t", [])):
            continue
        for ax, k in zip(axes, keys):
            if k not in tr and k != "rel_z":
                continue
            if k == "rel_z":
                y = np.hypot(tr["rel_x"], tr["rel_z"]) * 1e3
            else:
                y = tr[k] * scale[k]
            ax.plot(tr["t"], y, lw=0.9, label=f"{lab} ({dname})")
    t_settle = sc.close_time + sc.settle_time
    t_hold = t_settle + sc.lift_time + sc.hold_time
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab, fontsize=7)
        for t, txt in ((sc.close_time, "close"), (t_settle, "lift"), (t_hold, "shake"), (t_hold + sc.shake_time, "hold")):
            ax.axvline(t, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
    for t, txt in ((0, "close"), (sc.close_time, "settle"), (t_settle, "lift"), (t_hold, "shake"), (t_hold + sc.shake_time, "hold")):
        axes[0].text(t + 0.02, 1.02, txt, transform=axes[0].get_xaxis_transform(), fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Dynamic check time histories (all-rounder)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
