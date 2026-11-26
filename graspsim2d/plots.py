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
from .geometry import bracket_check_solid, build_finger, clearance_profile, finger_outline_view, swept_keep_out, to_view  # noqa: E402
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
    if an.ok:
        phi = math.radians(an.info["phi_deg"])
    elif "approach_opening_deg" in an.info:
        phi = math.radians(an.info["approach_opening_deg"])
    else:
        phi = build_finger(design, sc.keep_out, sc.closed_limit).phi_max
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
        f = build_finger(d, sc.keep_out, sc.closed_limit)
        an = solve_grip(d, box, sc, finger=f)
        phi = math.radians(an.info["approach_opening_deg"]) if "approach_opening_deg" in an.info else f.phi_max
        draw_gripper(ax, d, phi, _color(c.name), sc=sc)
        ax.set_aspect("equal")
        ax.set_xlim(-130, 130)
        ax.set_ylim(-d.finger_reach - 30, 25)
        ax.axis("off")
        ax.set_title(f"{c.label}\nopen {math.degrees(phi - f.phi_meet):.0f} of {d.max_opening:.0f} deg, for the box below", fontsize=9)
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


BRACKET_COLORS = ("#1f77b4", "#2ca02c")


def fig_bracket_clearance(path: Path, result: dict[str, Any], sc: Scenario) -> None:
    """Plan view of the brackets with the all-rounder swung through its stroke, and the least
    gap to the hub and to each bracket against opening, over every first-pass finalist."""
    ko = sc.keep_out
    designs = result["designs"]
    ar = designs[result["all_rounder"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for k, br in enumerate(ko.brackets):
        col = BRACKET_COLORS[k % len(BRACKET_COLORS)]
        if br.outline is not None:
            a1.plot(*br.outline.exterior.xy, color=col, lw=0.8, ls="--")
        for g in getattr(br.region, "geoms", [br.region]):
            a1.add_patch(Polygon(np.asarray(g.exterior.coords), closed=True, fc=col, ec=col, alpha=0.35, lw=0.6,
                                 label=f"{br.name}: part outside the hub and hinges"))
    for x, z, r in ko.circles:
        a1.add_patch(Circle((x, z), r, fc="#f4c7c3", ec="#c0392b", lw=0.6, ls="--", hatch="////", alpha=0.55, label="hub keep-out"))
    for x in (-1, 1):
        a1.add_patch(Circle((x * 19.0, 0.0), ko.hinge_radius, fc="none", ec="k", lw=0.5, ls=":"))
    gd = GripperDesign.from_dict(ar["design"])
    f = build_finger(gd, ko, sc.closed_limit)
    checked = bracket_check_solid(f, ko, swept_keep_out(f, ko, f.phi_meet, f.phi_max))
    openings = tuple(round(x) for x in np.linspace(0.0, gd.max_opening, 4))
    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(openings)))
    for op, shade in zip(openings, shades):
        ph = f.phi_meet + math.radians(op)
        for side in (1, -1):
            for P in finger_outline_view(f, ph, side)["body"]:
                a1.add_patch(Polygon(P, closed=True, fc="none", ec=shade, lw=0.5, alpha=0.8))
            for g in getattr(checked, "geoms", [checked]):
                a1.add_patch(Polygon(to_view(np.asarray(g.exterior.coords), ph, f.pivot_x, side), closed=True, fc=shade, ec="k",
                                     lw=0.3, alpha=0.75, label=f"{ar['name']}, parts checked, {op} deg" if side == 1 else None))
    a1.set_aspect("equal")
    a1.set_xlim(-120, 120)
    a1.set_ylim(-110, 55)
    a1.set_xlabel("x (mm)")
    a1.set_ylabel("z (mm), pivots at z = 0")
    a1.set_title(f"Brackets in plan, with the all-rounder {ar['name']} at {', '.join(map(str, openings[:-1]))} and {openings[-1]} deg "
                 "from where the fingers meet\n"
                 f"outline only = the root (first {ko.root_length:.0f} mm), kept from the platform finger, which clears both "
                 f"brackets in 3D; dotted = hinge, R{ko.hinge_radius:g}", fontsize=8)
    handles, labels = a1.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    a1.legend(keep.values(), keep.keys(), fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    stroke = float(gd.max_opening)
    wide = {name: d.get("widest_opening_deg") for name, d in designs.items()}
    reach = max([w for w in wide.values() if w is not None] + [stroke])
    openings_deg = np.linspace(0.0, 5.0 * math.ceil(reach / 5.0), int(math.ceil(reach / 5.0)) * 2 + 1)
    least: dict[str, np.ndarray] = {}
    base = designs.get("baseline")
    base_prof = None
    for name, d in designs.items():
        g = GripperDesign.from_dict(d["design"])
        fz = build_finger(g, ko, sc.closed_limit)
        top = wide[name] if wide[name] is not None else g.max_opening
        prof = clearance_profile(fz, ko, np.radians(openings_deg))
        for k, v in prof.items():
            v = np.where(openings_deg <= top + 1e-9, v, np.inf)
            least[k] = np.minimum(least.get(k, np.full_like(v, np.inf)), v)
        if d is base:
            base_prof = {k: np.where(openings_deg <= top + 1e-9, v, np.nan) for k, v in prof.items()}
    colors = {"hub": "#c0392b", **{br.name: BRACKET_COLORS[k % len(BRACKET_COLORS)] for k, br in enumerate(ko.brackets)}}
    inside = openings_deg <= stroke + 1e-9
    past = openings_deg >= stroke - 1e-9
    for i, (k, v) in enumerate(least.items()):
        # the brackets share the fillets beside the hub, so their curves can coincide: draw wide to narrow
        lw = max(3.4 - 1.0 * i, 1.4)
        v = np.where(np.isfinite(v), v, np.nan)
        if k == "hub":
            # past the stroke the root's own clearance comes from the 3D check, so only the parts beyond it are measured
            a2.plot(openings_deg[inside], v[inside], lw=lw, color=colors[k], alpha=0.85, label=f"hub, over the stroke: least of {len(designs)} designs")
            a2.plot(openings_deg[past], v[past], lw=1.6, ls="-.", color=colors[k], alpha=0.85, label="hub, past the stroke: parts beyond the root")
        else:
            a2.plot(openings_deg, v, lw=lw, color=colors[k], alpha=0.85, label=f"{k}: least of {len(designs)} designs")
        if base_prof is not None:
            b = base_prof[k]
            if k == "hub":
                b = np.where(inside, b, np.nan)
            a2.plot(openings_deg, b, lw=1.0, ls=":", color=colors[k], label=f"{k}: platform finger as drawn")
    a2.axhline(0.0, color="k", lw=0.8)
    a2.text(1, 0.3, "touching", fontsize=7)
    a2.axvline(stroke, color="k", lw=0.8, ls="--")
    a2.text(stroke - 1, 0.25, f"end of the {stroke:.0f} deg stroke", transform=a2.get_xaxis_transform(), fontsize=7, va="center", ha="right",
            rotation=90, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    lo_w = min(w for w in wide.values() if w is not None) if any(w is not None for w in wide.values()) else None
    if lo_w is not None and lo_w > stroke:
        a2.axvspan(lo_w, openings_deg[-1], color="#999999", alpha=0.15, lw=0)
        a2.text(lo_w + 1, 0.85, f"from {lo_w:.0f} deg the platform root\nof at least one design meets\nthe bottom bracket (3D)",
                transform=a2.get_xaxis_transform(), fontsize=7, va="top")
    a2.set_xlim(0, openings_deg[-1])
    a2.set_ylim(bottom=-1)
    a2.set_xlabel("opening from where the fingers meet (deg)")
    a2.set_ylabel("gap (mm)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7, loc="upper left")
    a2.set_title("Least gap to the hub and to each bracket, over the stroke and on to the widest opening each design\n"
                 f"reaches before its platform root meets a bracket (hub: its circle at R{ko.circles[0][2]:g}; brackets: "
                 "the parts at their heights;\npast the stroke, only the parts beyond the root, which clears the hub and brackets in 3D)"
                 if ko.circles else "", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_traces(path: Path, traces: dict[str, tuple[str, dict[str, np.ndarray]]], sc: Scenario) -> None:
    """Time histories from recorded dynamic runs: {object label: (design name, trace)}."""
    keys = ["grip_N", "rel_z", "rel_th", "friction_use", "p_max", "hinge_M"]
    labels = ["grip force per finger (N)", "slip of grip point (mm)", "object rotation (deg)",
              "share of available\nfriction used (worst pad)", "peak pad pressure (kPa)", "sheet bending moment at pinch (N*m)"]
    scale = {"grip_N": 1, "rel_z": 1e3, "rel_th": 180 / math.pi, "friction_use": 1, "p_max": 1e-3, "hinge_M": 1}
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.1 * len(keys)), sharex=True)
    # a jaw that starts wider closes for longer; line the runs up at the start of the lift
    close = {lab: float(tr["close_s"][0]) if len(tr.get("close_s", [])) else sc.close_time for lab, (_, tr) in traces.items()}
    t_close = max(close.values(), default=sc.close_time)
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
            ax.plot(tr["t"] + t_close - close[lab], y, lw=0.9, label=f"{lab} ({dname})")
    t_settle = t_close + sc.settle_time
    t_hold = t_settle + sc.lift_time + sc.hold_time
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab, fontsize=7)
        for t, txt in ((t_close, "close"), (t_settle, "lift"), (t_hold, "shake"), (t_hold + sc.shake_time, "hold")):
            ax.axvline(t, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
    for t, txt in ((0, "close"), (t_close, "settle"), (t_settle, "lift"), (t_hold, "shake"), (t_hold + sc.shake_time, "hold")):
        axes[0].text(t + 0.02, 1.02, txt, transform=axes[0].get_xaxis_transform(), fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)" if len(set(close.values())) <= 1 else "time (s), runs lined up at the start of the lift")
    fig.suptitle("Dynamic check time histories (all-rounder)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_sweep(path: Path, result: dict[str, Any], labels: dict[str, str] | None = None) -> None:
    """One varied variable: margin per object and pass count against it. Two: pass count over the grid."""
    vary, summary, labels = result["vary"], result["summary"], labels or {}
    n_obj = len(result["objects"])
    base = result["base"].get("label") or result["base"]["concept"]
    title = f"Sweep around {base}, {result['fidelity']} fidelity"
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(vary) == 1:
        name = vary[0]["name"]
        xs = np.array([s[name] for s in summary])
        score = {(r["point"], r["object_id"]): r["score"] for r in result["outcomes"]}
        panel = {(r["point"], r["object_id"]): (r["margins"].get("panel"), r["reason"] == "dropped") for r in result["outcomes"]}
        boxes = [o for o in result["objects"] if any(panel[(s["point"], o)][0] is not None for s in summary)]
        fig, axes = plt.subplots(3 if boxes else 2, 1, figsize=(9, 9.5 if boxes else 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2, 1] if boxes else [3, 1]})
        a1, a2 = axes[0], axes[-1]
        for k, oid in enumerate(result["objects"]):
            a1.plot(xs, [min(score[(s["point"], oid)], 3.0) for s in summary], marker="os^D"[k // 20 % 4], ms=3,
                    color=plt.cm.tab20(k % 20), ls="-" if k % 2 == 0 else "--", label=labels.get(oid, oid))
        a1.axhline(1.0, color="k", ls=":", lw=0.8)
        a1.set_ylabel("smallest pass margin (>= 1 passes,\ncapped at 3; 0 = dropped)")
        a1.set_title(title, fontsize=10)
        a1.legend(fontsize=7, ncol=2)
        if boxes:
            # a dropped run keeps the margins measured up to the drop
            a3 = axes[1]
            for k, oid in enumerate(result["objects"]):
                if oid not in boxes:
                    continue
                c = plt.cm.tab20(k % 20)
                pts = [(x, min(m, 3.0), dropped) for x, s in zip(xs, summary) for m, dropped in [panel[(s["point"], oid)]] if m is not None]
                a3.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=1, ls="-" if k % 2 == 0 else "--")
                for dropped, style in ((False, dict(mfc=c)), (True, dict(mfc="white"))):
                    sel = [p for p in pts if p[2] == dropped]
                    if sel:
                        a3.plot([p[0] for p in sel], [p[1] for p in sel], ls="none", marker="o", ms=4, mec=c, **style)
            a3.axhline(1.0, color="k", ls=":", lw=0.8)
            a3.set_ylabel("panel margin on each box\n(open marker: the box dropped)")
        width = 0.8 * float(np.min(np.diff(np.unique(xs)))) if len(np.unique(xs)) > 1 else 0.8
        a2.bar(xs, [s["n_pass"] for s in summary], width=width, color="tab:green", ec="k")
        a2.set_ylim(0, n_obj)
        a2.set_ylabel("objects passed")
        a2.set_xlabel(f"{name} ({SCHEMA_BY_NAME[name].unit})")
    else:
        (n1, v1), (n2, v2) = ((v["name"], v["values"]) for v in vary[:2])
        M = np.full((len(v2), len(v1)), np.nan)
        for s in summary:
            M[v2.index(s[n2]), v1.index(s[n1])] = s["n_pass"]
        fig, ax = plt.subplots(figsize=(2.2 + 0.7 * len(v1), 1.6 + 0.45 * len(v2)))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=n_obj, origin="lower", aspect="auto")
        for i in range(len(v2)):
            for j in range(len(v1)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(v1)))
        ax.set_xticklabels([f"{v:g}" for v in v1], fontsize=7)
        ax.set_yticks(range(len(v2)))
        ax.set_yticklabels([f"{v:g}" for v in v2], fontsize=7)
        ax.set_xlabel(f"{n1} ({SCHEMA_BY_NAME[n1].unit})")
        ax.set_ylabel(f"{n2} ({SCHEMA_BY_NAME[n2].unit})")
        ax.set_title(f"{title}\nobjects passed (of {n_obj})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
