#!/usr/bin/env python3
"""Short review clips from real batch-1 runs, for watching on a phone.

A stand-in for `mujoco` inside graspsim2d.t5 and graspsim2d.mj1 records the bodies and the case's contact loads while
the batch-1 harness (batch1.job_t5, job_scan, job_mj) runs each job unchanged, so every clip is the scored run itself.

    python tools/render_review_clips.py run                  # the sims, in parallel, into runs/review-clips
    python tools/render_review_clips.py render --dest DIR    # the MP4s and filmstrips from those recordings
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "runs" / "review-clips"
DESIGNS = ("P0", "S1", "S2", "S3", "S4", "S5", "S6")
NAMES = {"P0": "Today's finger", "S1": "Platform-plus", "S2": "Deep reach", "S3": "Top-edge clamp",
         "S4": "Asymmetric pair", "S5": "L-cradle crank", "S6": "Crowned face"}
BOX = "gm_cheerios_family_18oz"
REC_DT = 5e-4  # s between recorded poses; loads are kept at the harness's own sampling


# ----------------------------------------------------------------------------------------------- recording

class Recorder:
    """Forwards every attribute to the real mujoco module; after each mj_step keeps the drawn bodies' poses every
    REC_DT and, at the harness's sampling (every 4 base steps), the normal load on the case or box per finger."""

    def __init__(self, mj: Any, kind: str):
        self._mj, self.kind = mj, kind
        self.model = self.data = None
        self.t_pose: list[float] = []
        self.pose: list[np.ndarray] = []
        self.q: list[np.ndarray] = []
        self.t_load: list[float] = []
        self.N: list[np.ndarray] = []
        self.X: list[np.ndarray] = []
        self.Z: list[np.ndarray] = []
        self.static: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mj, name)

    def _index(self, m: Any, d: Any) -> None:
        mj = self._mj
        self.model, self.data = m, d
        gname = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        if self.kind == "t5":
            self.bodies = ["carriage", "case", "gripA", "gripB", "fingerAR", "fingerAL", "fingerBR", "fingerBL"]
            self.joints = ["jAR", "jAL", "jBR", "jBL", "cx", "cz", "cth", "kx", "kz"]
            self.parts = ["A_out", "A_thumb", "A_hub", "B_out", "B_thumb", "B_hub", "deck"]
            obj = "case"

            def part(n: str) -> int:
                if n.startswith(("A_hub", "B_hub")):
                    return self.parts.index(n[0] + "_hub")
                tag = n.split("_", 1)[0]
                if tag in ("AR", "BL"):
                    return self.parts.index(tag[0] + "_out")
                if tag in ("AL", "BR"):
                    return self.parts.index(tag[0] + "_thumb")
                return self.parts.index("deck") if n == "deck" else -1
        else:
            self.bodies = ["base", "fingerR", "fingerL", "obj"]
            self.joints = ["jR", "jL", "ox", "oz", "oth", "base_x", "base_z"]
            self.parts = ["R", "L", "table"]
            obj = "obj"

            def part(n: str) -> int:
                tag = n.split("_", 1)[0]
                if tag in ("R", "L"):
                    return self.parts.index(tag)
                return self.parts.index("table") if n == "table" else -1

        self.P = len(self.parts)
        self.part_of = np.array([part(n) for n in gname], dtype=int)
        self.obj_geom = gname.index(obj)
        self.bids = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b) for b in self.bodies], dtype=int)
        jids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        self.joints = [j for j, i in zip(self.joints, jids) if i >= 0]
        self.qadr = np.array([m.jnt_qposadr[i] for i in jids if i >= 0], dtype=int)
        self.every = 4 * max(1, int(round(1e-4 / m.opt.timestep)))
        self.n = 0
        self.next_pose = 0.0
        st: dict[str, Any] = {"timestep": float(m.opt.timestep), "every": self.every}
        for g in (obj, "deck", "table"):
            if g in gname:
                i = gname.index(g)
                st[g] = {"pos": [float(d.geom_xpos[i][0]), float(d.geom_xpos[i][2])],
                         "half": [float(m.geom_size[i][0]), float(m.geom_size[i][2])]}
        if self.kind == "t5":
            st["hubs"] = {g: self._hub_outlines(m, d, gname, g) for g in "AB"}
        self.static = st

    def _hub_outlines(self, m: Any, d: Any, gname: list[str], g: str) -> list[list[list[float]]]:
        """Each hub piece as a convex outline (mm) in its gripper's frame."""
        from shapely.geometry import MultiPoint
        b = self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_BODY, f"grip{g}")
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        out = []
        for i, n in enumerate(gname):
            if not n.startswith(f"{g}_hub") or m.geom_type[i] != self._mj.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[i]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[i] + v @ d.geom_xmat[i].reshape(3, 3).T
            loc = (w - pb) @ Rb
            hull = MultiPoint([(float(x) * 1e3, float(z) * 1e3) for x, _, z in loc]).convex_hull
            if hull.geom_type == "Polygon":
                out.append([list(p) for p in hull.exterior.coords])
        return out

    def mj_step(self, m: Any, d: Any) -> None:
        self._mj.mj_step(m, d)
        if m is not self.model or d is not self.data:
            self._index(m, d)
        t = float(d.time)
        if self.t_pose and t < self.t_pose[-1] - 1e-9:
            self._rewind(t)
        if self.n % self.every == 0:
            self._loads(d, t)
        self.n += 1
        if t >= self.next_pose - 1e-12:
            xm = d.xmat[self.bids]
            xp = d.xpos[self.bids]
            self.t_pose.append(t)
            self.pose.append(np.stack([xp[:, 0], xp[:, 2], np.arctan2(xm[:, 2], xm[:, 0])], axis=1))
            self.q.append(d.qpos[self.qadr].copy())
            self.next_pose = (math.floor(t / REC_DT + 1e-9) + 1) * REC_DT

    def _rewind(self, t: float) -> None:
        """mj1 restores a snapshot when a stop drops the box: keep only what came before it."""
        k = int(np.searchsorted(self.t_pose, t, side="left"))
        del self.t_pose[k:], self.pose[k:], self.q[k:]
        k = int(np.searchsorted(self.t_load, t, side="left"))
        del self.t_load[k:], self.N[k:], self.X[k:], self.Z[k:]
        self.next_pose = t

    def _loads(self, d: Any, t: float) -> None:
        P = self.P
        N, X, Z = np.zeros(P), np.zeros(P), np.zeros(P)
        n = d.ncon
        if n:
            con = d.contact
            g1, g2, adr = con.geom1[:n], con.geom2[:n], con.efc_address[:n]
            sel = ((g1 == self.obj_geom) | (g2 == self.obj_geom)) & (adr >= 0)
            if sel.any():
                other = np.where(g1[sel] == self.obj_geom, g2[sel], g1[sel])
                p = self.part_of[other]
                ok = p >= 0
                if ok.any():
                    fn = np.abs(d.efc_force[adr[sel][ok]])
                    pos = con.pos[:n][sel][ok]
                    N = np.bincount(p[ok], weights=fn, minlength=P)
                    X = np.bincount(p[ok], weights=fn * pos[:, 0], minlength=P)
                    Z = np.bincount(p[ok], weights=fn * pos[:, 2], minlength=P)
        self.t_load.append(t)
        self.N.append(N)
        self.X.append(X)
        self.Z.append(Z)


def _finger_outlines(jaw: Any, kind: str) -> dict[str, dict[str, list]]:
    """Each finger's drawing outline and pads (mm) in its MuJoCo body frame: (X, Z) = (-side v, -u). Gripper B in
    T5 carries the mirrored jaw: its side s is the spec's side -s."""
    def polys(geom: Any) -> list[list[list[float]]]:
        gs = getattr(geom, "geoms", [geom])
        return [[[float(u), float(v)] for u, v in g.exterior.coords] for g in gs if g.geom_type == "Polygon"]

    def place(pts: list[list[float]], side: int) -> list[list[float]]:
        return [[-side * v, -u] for u, v in pts]

    fingers = {"AR": (1, 1), "AL": (-1, -1), "BR": (-1, 1), "BL": (1, -1)} if kind == "t5" else {"R": (1, 1), "L": (-1, -1)}
    out = {}
    for f, (spec_side, side) in fingers.items():
        js = jaw.sides[spec_side]
        out[f] = {"body": [place(p, side) for p in polys(js.outline)],
                  "pads": [place([[float(a), float(b)] for a, b in face.pad.outline()], side)
                           for face in js.faces if face.kind == "pad"]}
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int, bool, np.bool_)) or v is None or isinstance(v, str):
        return v.item() if isinstance(v, (np.generic,)) else v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _timeline_dict(tl: Any) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in vars(tl).items() if k != "sc"}


def t5_job(design: str, case: str, mass: float, **kw: Any) -> dict[str, Any]:
    """batch1.t5_jobs' tips-only hold job, centred, with S4's and S5's stop-seated approach at 45 deg."""
    from graspsim2d import batch1
    blk = batch1.ctx()["b1"]["t5"]["block"]
    seated = 45.0 in [float(r) for r in blk["stop_seated"].get(design, [])]
    return batch1._base("t5tips", design, 45.0, job="t5", test="T5", hub_contact=False, object=case, mass_kg=mass,
                        com_offset=0.0, case=f"hold, {mass:g} kg, tips only (review clip)", rated=True,
                        t5_path="tips only", **({"approach": "seated"} if seated else {}), **kw)


def run_list() -> dict[str, tuple[str, dict[str, Any]]]:
    runs: dict[str, tuple[str, dict[str, Any]]] = {}
    for d in DESIGNS:
        # batch1.t5_jobs applies the 1 g stop only at the 23 kg ceiling; these clips set it explicitly.
        runs[f"lift12_{d}"] = ("t5", t5_job(d, "2ft", 12.0, stop_g=1.0))
    runs["lift23_P0_cut"] = ("t5", t5_job("P0", "3ft", 23.0, stop_g=1.0, cut_before_stop=True))
    for d in ("P0", "S2"):
        runs[f"box_{d}"] = ("t1", {"block": "review", "design": d, "reading": 45.0, "job": "scan", "object": BOX,
                                   "fill": "catalog"})
    return runs


def run_one(name: str, kind: str, job: dict[str, Any], out_dir: str) -> str:
    import mujoco
    from graspsim2d import batch1, mj1, t5
    t0 = time.time()
    extra: dict[str, Any] = {}
    if kind == "t5":
        mod, tl_name = t5, "T5Timeline"
        call = lambda: batch1.job_t5(job)
    else:
        scan = batch1.job_scan(job)
        if not scan.get("simulate"):
            raise RuntimeError(f"{name}: the screen didn't simulate ({scan.get('not_simulated')})")
        mjob = batch1.t1_followups(scan)[0]
        extra = {"scan": {k: scan.get(k) for k in ("depth_mm", "status", "stop_mm", "table_cap_mm")}, "mj_job": mjob}
        mod, tl_name = mj1, "Timeline"
        call = lambda: batch1.job_mj(mjob)
    rec = Recorder(mujoco, kind)
    timelines: list[Any] = []
    base_tl = getattr(mod, tl_name)

    class _Timeline(base_tl):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any):
            super().__init__(*a, **k)
            timelines.append(self)

    mod.mujoco, prev = rec, mod.mujoco
    setattr(mod, tl_name, _Timeline)
    try:
        row = call()
    finally:
        mod.mujoco = prev
        setattr(mod, tl_name, base_tl)
    jaw = batch1.get_jaw(job["design"], job["reading"], job.get("values"), job.get("reach_cap"))
    meta = {"name": name, "kind": kind, "job": _jsonable(job), **_jsonable(extra), "row": _jsonable(row),
            "bodies": rec.bodies, "joints": rec.joints, "parts": rec.parts, "static": _jsonable(rec.static),
            "timelines": [_timeline_dict(tl) for tl in timelines], "outlines": _finger_outlines(jaw, kind),
            "wall_s": round(time.time() - t0, 1)}
    out = Path(out_dir)
    np.savez_compressed(out / f"{name}.npz", t_pose=np.array(rec.t_pose), pose=np.array(rec.pose, dtype=np.float64),
                        q=np.array(rec.q), t_load=np.array(rec.t_load), N=np.array(rec.N), X=np.array(rec.X),
                        Z=np.array(rec.Z))
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
    return f"{name}: {meta['wall_s']} s, dropped_in={row.get('dropped_in')!r}, passed={row.get('passed')}"


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = run_list()
    names = [n for n in runs if not args.only or n in args.only]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, n, *runs[n], str(out)): n for n in names}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{futs[f]}: FAILED {e!r}", flush=True)


# ----------------------------------------------------------------------------------------------- the board rule

# SIM's footprint rule for batch 1 (the T5 recount, revision 2.26): a contact within 1 deg of the wall is flush and
# reads its force over the face, against 105 kPa; anything else is an edge and reads its force over the contact
# width, against 0.84 N/mm. S4's floor is read whole, about 5,300 mm^2, as SIM's round 13 rules until the corner
# fix; the recount read its 600 mm^2 tip land (S4_LAND_MM2).
FOOTPRINT = {"P0": {"thumb": ("edge", 6.0), "out": ("edge", 6.0)},
             "S1": {"thumb": ("edge", 45.0), "out": ("edge", 45.0)},
             "S2": {"thumb": ("edge", 100.0), "out": ("edge", 100.0)},
             "S3": {"thumb": ("edge", 60.0), "out": ("edge", 60.0)},
             "S4": {"thumb": ("face", 1813.0), "out": ("face", 5300.0)},
             "S5": {"thumb": ("face", 1943.0), "out": ("face", 1943.0)},
             "S6": {"thumb": ("edge", 70.0), "out": ("edge", 70.0)}}
S4_LAND_MM2 = 600.0
LIMIT = {"edge": 0.84, "face": 105.0}
HUMID = 0.6  # 85 % RH on C-flute


def board(design: str, role: str, N: Any) -> tuple[Any, Any]:
    kind, size = FOOTPRINT[design][role]
    reading = N / size * (1e3 if kind == "face" else 1.0)
    return reading, reading / LIMIT[kind]


def t5_verdict(row: dict[str, Any], design: str) -> dict[str, Any]:
    """The rated reading from the row's single-sample peaks over both grippers, as the recount reads it."""
    out: dict[str, Any] = {}
    for role in ("thumb", "out"):
        N = max(row[f"A_{role}_peak_N"], row[f"B_{role}_peak_N"])
        reading, ratio = board(design, role, N)
        out[role] = {"N": N, "reading": reading, "ratio": ratio, "kind": FOOTPRINT[design][role][0]}
    gov = max(("thumb", "out"), key=lambda r: out[r]["ratio"])
    g = out[gov]
    if row.get("dropped_in"):
        word, ok = "DROPS", False
    elif g["ratio"] > 1.0:
        word, ok = "CRUSHES THE BOARD", False
    elif g["ratio"] > HUMID:
        word, ok = "HOLDS, DRY CASE ONLY", True
    else:
        word, ok = "HOLDS", True
    slip = float(row.get("slip_mm") or 0.0)
    if ok and slip >= 2.0:
        word += f", SLIPS {slip:.0f} MM"
    who = "thumb" if gov == "thumb" else "outside finger"
    lim = LIMIT[g["kind"]]
    detail = (f"{who} {g['reading']:.3g} N/mm, limit {lim:g}" if g["kind"] == "edge"
              else f"{who} {g['reading']:.0f} kPa, limit {lim:.0f} ({lim * HUMID:.0f} humid)")
    return {"word": word, "ok": ok, "ratio": g["ratio"], "gov": gov, "detail": detail, **out}


# ----------------------------------------------------------------------------------------------- drawing

W_PX, H_PX, FPS = 720, 1280, 30
C = {"bg": "#ffffff", "ink": "#16202c", "sub": "#566170", "rule": "#d5dae1", "case": "#dcc097", "case_edge": "#86642f",
     "finger": "#86a0c4", "finger_edge": "#2b3f5c", "pad": "#262a30", "hub": "#cfd3d8", "hub_edge": "#8d949d",
     "housing": "#aab2bc", "arm": "#dde1e6", "deck": "#c4c9cf", "pass": "#1e8449", "fail": "#c0392b",
     "warn": "#d68910", "cereal": "#e8c37a", "empty": "#f7ecd2", "fill_line": "#7a5c1e", "table": "#a7adb4",
     "stop": "#b9461f", "calm": "#2f5d8a", "ghost": "#2b3f5c"}


def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    return plt


def fs(n: float) -> float:
    """Points for n pixels at 100 dpi."""
    return n * 0.72


class Canvas:
    """A figure laid out in pixels from the top-left."""

    def __init__(self, w: int = W_PX, h: int = H_PX):
        plt = _plt()
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.fig.patch.set_facecolor(C["bg"])

    def axes(self, x: float, y: float, w: float, h: float) -> Any:
        ax = self.fig.add_axes((x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h))
        ax.set_axis_off()
        return ax

    def text(self, x: float, y: float, s: str, size: float, color: str = C["ink"], weight: str = "normal",
             ha: str = "left", va: str = "top", **kw: Any) -> Any:
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=fs(size), color=color, weight=weight, ha=ha,
                             va=va, **kw)

    def box(self, x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 0.0,
            round_px: float = 0.0) -> Any:
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x / self.w, 1 - (y + h) / self.h), w / self.w, h / self.h,
                           boxstyle=f"round,pad=0,rounding_size={round_px / self.h}" if round_px else "square,pad=0",
                           transform=self.fig.transFigure, fc=fc, ec=ec, lw=lw)
        self.fig.patches.append(p)
        return p

    def line(self, x0: float, y0: float, x1: float, y1: float, color: str = C["rule"], lw: float = 1.5) -> Any:
        from matplotlib.lines import Line2D
        ln = Line2D([x0 / self.w, x1 / self.w], [1 - y0 / self.h, 1 - y1 / self.h], transform=self.fig.transFigure,
                    color=color, lw=lw)
        self.fig.lines.append(ln)
        return ln

    def grab(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


class Rec:
    """One recorded run."""

    def __init__(self, name: str, run_dir: Path = RUN_DIR):
        self.name = name
        self.meta = json.loads((run_dir / f"{name}.json").read_text())
        z = np.load(run_dir / f"{name}.npz")
        self.t, self.pose, self.q = z["t_pose"], z["pose"], z["q"]
        self.tL, self.N, self.X, self.Z = z["t_load"], z["N"], z["X"], z["Z"]
        self.b = {n: i for i, n in enumerate(self.meta["bodies"])}
        self.j = {n: i for i, n in enumerate(self.meta["joints"])}
        self.p = {n: i for i, n in enumerate(self.meta["parts"])}
        self.tl = self.meta["timelines"][-1]
        self.row = self.meta["row"]
        self.design = self.meta["job"]["design"]
        self.t_last = float(self.t[-1])

    def run_time(self, v: float) -> float:
        """Virtual time v: before the lift, -1..0 spans the start to the end of the settle; after it, seconds."""
        ts = float(self.tl["t_settle"])
        return ts * (1.0 + v) if v < 0 else ts + v

    def k(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def body(self, name: str, t: float) -> np.ndarray:
        return self.pose[self.k(t), self.b[name]]

    def joint(self, name: str, t: float) -> float:
        return float(self.q[self.k(t), self.j[name]])

    def loads(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per part: the largest sampled normal load in (t0, t1] and where it acted (load-weighted, mm)."""
        a = int(np.searchsorted(self.tL, t0, side="right"))
        b = int(np.searchsorted(self.tL, t1, side="right"))
        if b <= a:
            a, b = max(b - 1, 0), max(b, 1)
        cols = np.arange(self.N.shape[1])
        kk = a + np.argmax(self.N[a:b], axis=0)
        N = self.N[kk, cols]
        Nn = np.maximum(N, 1e-9)
        return N, self.X[kk, cols] / Nn * 1e3, self.Z[kk, cols] / Nn * 1e3


def t1_slip(rec: Rec) -> Any:
    """mj1's slip: a grip point fixed in the box (its centreline, at the load-weighted contact height when the
    settle ends) against the gripper base, from that sample on."""
    t_ref = float(rec.tL[np.searchsorted(rec.tL, float(rec.tl["t_settle"]) - 1e-12)])
    N, _, Z = rec.loads(t_ref - 1e-9, t_ref)
    parts = [rec.p["R"], rec.p["L"]]
    w = N[parts]
    box = rec.body("obj", t_ref)
    zg = float(np.average(Z[parts], weights=w)) if w.sum() > 1e-6 else box[1] * 1e3
    c, s = math.cos(box[2]), math.sin(box[2])
    dx, dz = 0.0, zg - box[1] * 1e3
    local = np.array([[c * dx - s * dz, s * dx + c * dz]])

    def grip(t: float) -> np.ndarray:
        return tf(rec.body("obj", t), local)[0] - rec.body("base", t)[:2] * 1e3

    g0 = grip(t_ref)
    return lambda t: float(np.hypot(*(grip(t) - g0))) if t >= t_ref else 0.0


def tf(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Body-frame points (mm) to world (mm) for a pose (x m, z m, rotation about y)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack([pose[0] * 1e3 + c * pts[:, 0] + s * pts[:, 1], pose[1] * 1e3 - s * pts[:, 0] + c * pts[:, 1]])


def rect(x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], dtype=float)


def ratio_color(r: float) -> str:
    return C["fail"] if r > 1.0 else (C["warn"] if r > HUMID else C["pass"])


def speed_text(sim_s: float, video_s: float) -> str:
    sp = sim_s / video_s
    if sp < 0.7:
        return f"slowed {1 / sp:.0f}×"
    if sp <= 1.4:
        return "real time"
    return f"{sp:.0f}× speed"


class Scene:
    """A run's bodies as polygons on `ax`, in the frame of `follow` (the carriage, or the gripper base)."""

    def __init__(self, ax: Any, rec: Rec, window: tuple[float, float, float, float], follow: str, lw: float = 1.0):
        from matplotlib.patches import Polygon
        self._Polygon = Polygon
        self.ax, self.rec, self.follow, self.lw = ax, rec, follow, lw
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        self.items: list[tuple[Any, np.ndarray, str]] = []
        self.world: list[tuple[Any, np.ndarray]] = []

    def add(self, local: np.ndarray, body: str, fc: str, ec: str, z: float, lw: float | None = None, **kw: Any) -> Any:
        p = self._Polygon(np.asarray(local, dtype=float), closed=True, fc=fc, ec=ec, lw=self.lw if lw is None else lw,
                          zorder=z, joinstyle="round", **kw)
        self.ax.add_patch(p)
        self.items.append((p, np.asarray(local, dtype=float), body))
        return p

    def add_world(self, pts: np.ndarray, fc: str, ec: str, z: float) -> Any:
        p = self._Polygon(pts, closed=True, fc=fc, ec=ec, lw=self.lw, zorder=z)
        self.ax.add_patch(p)
        self.world.append((p, np.asarray(pts, dtype=float)))
        return p

    def add_gripper(self, grip: str, fingers: tuple[str, ...], hubs: list | None = None) -> None:
        """Schematic arm and housing behind the pivots, the hub pieces, then each finger's outline and pads."""
        self.add(rect(-20, 20, 50, 2000), grip, C["arm"], C["hub_edge"], 1)
        self.add(rect(-38, 38, 4, 52), grip, C["housing"], C["hub_edge"], 2)
        for poly in hubs or []:
            self.add(np.array(poly), grip, C["hub"], C["hub_edge"], 2)
        outl = self.rec.meta["outlines"]
        for f in fingers:
            for poly in outl[f]["body"]:
                self.add(np.array(poly), f"finger{f}", C["finger"], C["finger_edge"], 4)
            for poly in outl[f]["pads"]:
                self.add(np.array(poly), f"finger{f}", C["pad"], C["pad"], 5)

    def offset(self, t: float) -> np.ndarray:
        p = self.rec.body(self.follow, t)
        return np.array([p[0], p[1]]) * 1e3

    def update(self, t: float) -> np.ndarray:
        off = self.offset(t)
        k = self.rec.k(t)
        for patch, local, body in self.items:
            patch.set_xy(tf(self.rec.pose[k, self.rec.b[body]], local) - off)
        for patch, pts in self.world:
            patch.set_xy(pts - off)
        return off


def t5_scene(ax: Any, rec: Rec, window: tuple[float, float, float, float], lw: float = 1.0) -> Scene:
    st = rec.meta["static"]
    sc = Scene(ax, rec, window, "carriage", lw)
    hx, hz = (v * 1e3 for v in st["case"]["half"])
    if "deck" in st:
        dx, dz = (v * 1e3 for v in st["deck"]["half"])
        px_, pz_ = (v * 1e3 for v in st["deck"]["pos"])
        sc.deck = sc.add_world(rect(px_ - dx, px_ + dx, pz_ - dz, pz_ + dz), C["deck"], C["hub_edge"], 0.5)
    sc.add_gripper("gripA", ("AR", "AL"), st["hubs"]["A"])
    sc.add_gripper("gripB", ("BR", "BL"), st["hubs"]["B"])
    sc.add(rect(-hx, hx, -hz, hz), "case", C["case"], C["case_edge"], 3, lw=lw * 1.4)
    return sc


def finger_bbox(rec: Rec, fingers: tuple[str, ...], t: float, follow: str) -> tuple[float, float, float, float]:
    off = np.array(rec.body(follow, t)[:2]) * 1e3
    pts = np.vstack([tf(rec.body(f"finger{f}", t), np.array(p)) - off for f in fingers
                     for p in rec.meta["outlines"][f]["body"]])
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def fit(cx: float, cz: float, hx: float, hz: float, aspect: float) -> tuple[float, float, float, float]:
    """A window centred on (cx, cz) covering +-hx, +-hz, widened to the axes' aspect (width / height)."""
    if hx / hz < aspect:
        hx = hz * aspect
    else:
        hz = hx / aspect
    return cx - hx, cx + hx, cz - hz, cz + hz


class Gauge:
    """A vertical bar: the live board reading over its limit (0 - 2x), the peak so far, and the limit line."""

    def __init__(self, cv: Canvas, x: float, y: float, w: float, h: float):
        from matplotlib.patches import Rectangle
        self.ax = cv.axes(x, y, w, h)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 2)
        self.ax.add_patch(Rectangle((0, 0), 1, 2, fc="#eef0f3", ec="#b8bec6", lw=1))
        self.fill = Rectangle((0, 0), 1, 0, fc=C["pass"], ec="none")
        self.ax.add_patch(self.fill)
        self.peak = self.ax.plot([-0.15, 1.15], [0, 0], color=C["ink"], lw=2.5, clip_on=False)[0]
        self.ax.plot([-0.35, 1.35], [1, 1], color=C["ink"], lw=1.6, ls=(0, (3, 2)), clip_on=False)

    def set(self, live: float, peak: float) -> None:
        self.fill.set_height(min(live, 2.0))
        self.fill.set_facecolor(ratio_color(live))
        self.peak.set_ydata([min(peak, 2.0)] * 2)
        self.peak.set_visible(peak > 0)


class Banner:
    """The phase banner: a coloured pill with the phase and the playback speed, and the sim clock."""

    def __init__(self, cv: Canvas, y: float):
        self.pill = cv.box(20, y, 680, 46, C["calm"], round_px=10)
        self.txt = cv.text(36, y + 23, "", 23, "white", "bold", va="center")
        self.clock = cv.text(686, y + 23, "", 19, "white", ha="right", va="center")

    def set(self, phase: str, speed: str, color: str, t: float | None) -> None:
        self.pill.set_facecolor(color)
        self.txt.set_text(f"{phase}  ·  {speed}" if speed else phase)
        self.clock.set_text("" if t is None else f"sim {t:5.2f} s")


def plan(segs: list[tuple[float, float, float, str, str]], freeze: float, freeze_label: str,
         freeze_color: str) -> list[tuple[float, float, str, str, str]]:
    """Frames as (v, v at the previous frame, phase, speed, colour) from (v0, v1, video seconds, phase, colour)."""
    out = []
    v_prev = segs[0][0]
    for v0, v1, dur, phase, color in segs:
        n = max(1, int(round(dur * FPS)))
        sim = (v1 - v0) if v0 >= 0 else None
        sp = speed_text(v1 - v0, dur) if sim is not None else ""
        for i in range(n):
            v = v0 + (v1 - v0) * (i + 1) / n
            out.append((v, v_prev, phase, sp, color))
            v_prev = v
    for _ in range(int(round(freeze * FPS))):
        out.append((v_prev, v_prev, freeze_label, "", freeze_color))
    return out


class Encoder:
    def __init__(self, path: Path, w: int = W_PX, h: int = H_PX, crf: int = 26):
        import subprocess
        self.path = path
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-tune", "animation",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "3.1",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self.p.stdin is not None
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path}")


def filmstrip(tiles: list[tuple[np.ndarray, str]], title: str, path: Path, scale: float = 0.75,
              cols: int = 2) -> None:
    """Labelled frames in a grid, palettised so the PNG stays small."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 27)
    big = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 34)
    h0, w0 = tiles[0][0].shape[:2]
    tw, th = int(w0 * scale), int(h0 * scale)
    cap, pad, head = 58, 14, 72
    rows = math.ceil(len(tiles) / cols)
    W = cols * tw + (cols + 1) * pad
    H = head + rows * (cap + th + pad)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 18), title, fill=C["ink"], font=big)
    for i, (frame, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = head + r * (cap + th + pad)
        d.rounded_rectangle((x, y + 6, x + tw, y + cap - 6), radius=8, fill="#2f5d8a")
        d.text((x + 14, y + cap / 2), f"{i + 1}. {label}", fill="white", font=bold, anchor="lm")
        img.paste(Image.fromarray(frame).resize((tw, th), Image.LANCZOS), (x, y + cap))
        d.rectangle((x, y + cap, x + tw - 1, y + cap + th - 1), outline="#b8bec6", width=2)
    img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


# ----------------------------------------------------------------------------------------------- clip 1

def clip_lift12(run_dir: Path, dest: Path) -> dict[str, Any]:
    """All seven designs in the 12 kg two-arm lift on the fingertips, through the 1 g stop."""
    recs = {d: Rec(f"lift12_{d}", run_dir) for d in DESIGNS}
    verdict = {d: t5_verdict(recs[d].row, d) for d in DESIGNS}
    cv = Canvas()
    cv.text(20, 14, "12 kg case, lifted on the fingertips", 31, weight="bold")
    sub = cv.text(20, 58, "Each panel: gripper A, the one the 1 g stop loads", 21, C["sub"])
    banner = Banner(cv, 96)
    cv.line(0, 160, 720, 160)
    cells = [(0, 0)] + [(i % 2, i // 2) for i in range(1, 8)]
    order = ["setup", *DESIGNS]
    CW, CH, Y0 = 360, 279, 162
    aspect = 232 / 170
    t_ref = {d: recs[d].run_time(5.4) for d in DESIGNS}
    boxes = {d: finger_bbox(recs[d], ("AR", "AL"), t_ref[d], "carriage") for d in DESIGNS}
    hx = max(b[1] - b[0] for b in boxes.values()) / 2 + 16
    hz = max(b[3] - b[2] for b in boxes.values()) / 2 + 16
    scenes, gauges, ratio_txt, peaks = {}, {}, {}, {d: 0.0 for d in DESIGNS}
    markers: dict[str, dict[str, Any]] = {}
    mm_per_px: dict[str, float] = {}
    from matplotlib.patches import Circle
    for name, (cx_, cy_) in zip(order, cells):
        x, y = cx_ * CW, Y0 + cy_ * CH
        if cx_ == 1:
            cv.line(x, y + 8, x, y + CH - 8)
        if cy_ > 0:
            cv.line(x + 8, y, x + CW - 8, y)
        if name == "setup":
            rec = recs["P0"]
            st = rec.meta["static"]
            W2 = st["case"]["half"][0] * 1e3
            case_z = (rec.body("case", t_ref["P0"])[1] - rec.body("carriage", t_ref["P0"])[1]) * 1e3
            ax = cv.axes(x + 6, y + 40, 348, 160)
            win = fit(0.0, case_z - 60, W2 + 175, 260, 348 / 160)
            scenes[name] = t5_scene(ax, rec, win, lw=0.6)
            b = boxes["P0"]
            wx = fit((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, hx, hz, aspect)
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((wx[0], wx[2]), wx[1] - wx[0], wx[3] - wx[2], fill=False, ec=C["stop"], lw=1.6,
                                   ls=(0, (3, 2)), zorder=8))
            ax.text(wx[1], wx[3] + 8, "close-ups", color=C["stop"], fontsize=fs(16), ha="right", va="bottom",
                    weight="bold", zorder=9)
            ax.text(-W2 - 20, win[2] + 6, "B", fontsize=fs(22), weight="bold", color=C["ink"], ha="right", zorder=9)
            ax.text(W2 + 20, win[2] + 6, "A", fontsize=fs(22), weight="bold", color=C["ink"], ha="left", zorder=9)
            scenes["arrow"] = ax.annotate("", xy=(W2 * 0.55, case_z), xytext=(-W2 * 0.35, case_z), zorder=9,
                                          arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.8",
                                                      "lw": 4, "color": C["stop"]})
            cv.text(x + 14, y + 10, "Setup: 2 ft case, 12 kg", 22, weight="bold")
            land = recs["S4"].row["A_out_peak_N"] / S4_LAND_MM2 * 1e3
            for k, s in enumerate(("Bar: board load ÷ the board's limit;", "red above the line crushes the board.",
                                   "*S4: floor read whole, as SIM rules;",
                                   f"its tip land alone reads {land:.0f} kPa.")):
                cv.text(x + 14, y + 203 + 18 * k, s, 15, C["sub"])
            continue
        d = name
        cv.text(x + 14, y + 8, d, 27, weight="bold")
        cv.text(x + 62, y + 13, NAMES[d], 19, C["sub"])
        b = boxes[d]
        win = fit((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, hx, hz, aspect)
        ax = cv.axes(x + 4, y + 42, 232, 170)
        scenes[d] = t5_scene(ax, recs[d], win, lw=0.9)
        mm_per_px[d] = (win[1] - win[0]) / 232
        markers[d] = {}
        for role, col in (("thumb", C["fail"]), ("out", C["fail"])):
            c = Circle((0, 0), 1.0, fc=col, ec="white", lw=1.2, alpha=0.85, zorder=7, visible=False)
            ax.add_patch(c)
            markers[d][role] = c
        gauges[d] = Gauge(cv, x + 248, y + 52, 20, 150)
        cv.text(x + 278, y + 116, "limit", 15, C["sub"], va="center")
        ratio_txt[d] = cv.text(x + 352, y + 58, "", 27, C["ink"], "bold", ha="right")
        vd = verdict[d]
        cv.text(x + 14, y + 218, vd["word"] + ("*" if d == "S4" else ""), 20, C["pass"] if vd["ok"] else C["fail"], "bold")
        cv.text(x + 14, y + 246, vd["detail"], 15.5, C["sub"])
    segs = [(-1.0, 0.0, 1.0, "Close on the case", C["calm"]),
            (0.0, 1.0, 0.9, "Lift 80 mm", C["calm"]),
            (1.0, 5.46, 0.9, "Shake 0.25 g, hold", C["calm"]),
            (5.46, 5.80, 3.4, "1 g STOP", C["stop"]),
            (5.80, 11.7, 0.8, "Power cut, brakes hold", C["calm"])]
    holds = [d for d in DESIGNS if verdict[d]["ok"]]
    frames = plan(segs, 1.7, "Result", C["ink"])
    t_scored = {d: float(recs[d].tl["t_lift"]) for d in DESIGNS}
    stills = {"lifted": 5.40, "stop": 5.52, "late": 5.70, "end": frames[-1][0]}
    tiles: dict[str, np.ndarray] = {}

    def draw(v: float, v_prev: float, phase: str, speed: str, color: str, final: bool) -> None:
        banner.set(phase, speed, color, None if v < 0 else recs["P0"].run_time(v))
        if final:
            names = " and ".join(d + ("*" if d == "S4" else "") for d in sorted(holds, key=lambda d: d != "S5"))
            sub.set_text(f"Hold on a dry case: {names or 'none'}. The rest crush the board.")
            sub.set_color(C["ink"])
        for name, sc in scenes.items():
            if name == "arrow":
                sc.set_visible(5.5 <= v < 5.704)
                continue
            rec = recs["P0"] if name == "setup" else recs[name]
            t = min(rec.run_time(v), rec.t_last)
            off = sc.update(t)
            if hasattr(sc, "deck"):
                sc.deck.set_visible(v < 5.5)
            if name == "setup":
                continue
            d = name
            t0 = min(rec.run_time(v_prev), t)
            N, X, Z = rec.loads(t0, t)
            live = 0.0
            for role in ("thumb", "out"):
                i = rec.p[f"A_{role}"]
                _, r = board(d, role, N[i])
                live = max(live, float(r))
                m = markers[d][role]
                m.set_visible(N[i] > 2.0 and v >= 0)
                m.center = (X[i] - off[0], Z[i] - off[1])
                m.set_radius((3.0 + 5.0 * min(float(r), 2.0)) * mm_per_px[d])
                m.set_facecolor(ratio_color(float(r)))
            if t >= t_scored[d]:
                peaks[d] = max(peaks[d], live)
            gauges[d].set(live, peaks[d])
            pk = peaks[d]
            ratio_txt[d].set_text(f"{pk:.0f}×" if pk >= 10 else f"{pk:.1f}×" if pk >= 1 else f"{pk:.2f}×")
            ratio_txt[d].set_color(ratio_color(pk) if pk > 0 else C["ink"])
            ratio_txt[d].set_visible(pk > 0)

    enc = Encoder(dest / "clip1-case-lift-12kg-all-seven.mp4")
    for i, (v, vp, phase, speed, color) in enumerate(frames):
        final = i >= len(frames) - int(round(1.7 * FPS))
        draw(v, vp, phase, speed, color, final)
        frame = cv.grab()
        enc.write(frame)
        for key, vs in stills.items():
            if key not in tiles and (v >= vs - 1e-9):
                tiles[key] = frame
    enc.close()
    filmstrip([(tiles["lifted"], "Lifted, before the stop"), (tiles["stop"], "1 g stop, +20 ms"),
               (tiles["late"], "1 g stop, +0.2 s"), (tiles["end"], "After 5 s on the brakes")],
              "12 kg case lift, all seven, through the 1 g stop", dest / "clip1-case-lift-12kg-all-seven-filmstrip.png")
    check = {d: {"gauge_peak": peaks[d], "row_ratio_A": max(board(d, r, recs[d].row[f"A_{r}_peak_N"])[1]
                                                             for r in ("thumb", "out"))} for d in DESIGNS}
    return {"verdict": verdict, "check": check}


# ----------------------------------------------------------------------------------------------- clip 2

def clip_cut23(run_dir: Path, dest: Path) -> dict[str, Any]:
    """Today's finger with the 23 kg case when the power cuts before the 1 g stop."""
    rec = Rec("lift23_P0_cut", run_dir)
    tl = rec.tl
    v_cut = float(tl["t_power"]) - float(tl["t_settle"])
    v_stop = float(tl["stop"][0]) - float(tl["t_settle"])
    v_end = rec.t_last - float(tl["t_settle"])
    cv = Canvas()
    cv.text(20, 14, "Today's finger (P0), 23 kg case", 31, weight="bold")
    sub = cv.text(20, 58, "The power cuts first, then the 1 g stop", 22, C["sub"])
    banner = Banner(cv, 96)
    cv.line(0, 160, 720, 160)
    t_ref = rec.run_time(v_cut)
    b = finger_bbox(rec, ("AR", "AL"), t_ref, "carriage")
    win = fit((b[0] + b[1]) / 2 - 4, (b[2] + b[3]) / 2 + 6, (b[1] - b[0]) / 2 + 24, (b[3] - b[2]) / 2 + 24, 720 / 660)
    ax = cv.axes(0, 166, 720, 660)
    main = t5_scene(ax, rec, win, lw=1.6)
    cv.text(24, 176, "Gripper A, close up", 22, C["sub"], "bold")
    off0 = main.offset(t_ref)
    k0 = rec.k(t_ref)
    for f in ("AR", "AL"):
        for poly in rec.meta["outlines"][f]["body"]:
            pts = tf(rec.pose[k0, rec.b[f"finger{f}"]], np.array(poly)) - off0
            ax.add_patch(main._Polygon(pts, closed=True, fill=False, ec=C["stop"], lw=1.8, ls=(0, (4, 3)), zorder=6))
    ax.text(win[1] - 14, win[2] + 12, "dashed: the fingers when the power cut", color=C["stop"], fontsize=fs(18),
            ha="right", va="bottom", zorder=9)
    cv.line(0, 832, 720, 832)
    cv.text(24, 846, "Jaw A opens", 22, C["sub"], "bold")
    jaw_txt = cv.text(24, 878, "", 46, C["ink"], "bold")
    jawB = cv.text(24, 940, "", 18, C["sub"])
    cv.text(380, 846, "Case slip on the fingers", 22, C["sub"], "bold")
    slip_txt = cv.text(380, 878, "", 46, C["ink"], "bold")
    slip_cut = cv.text(380, 978, "", 15, C["sub"])
    from matplotlib.patches import Rectangle
    sax = cv.axes(380, 944, 316, 26)
    sax.set_xlim(0, 22)
    sax.set_ylim(0, 1)
    sax.add_patch(Rectangle((0, 0), 22, 1, fc="#eef0f3", ec="#b8bec6", lw=1))
    sbar = Rectangle((0, 0), 0, 1, fc=C["warn"], ec="none")
    sax.add_patch(sbar)
    sax.plot([20, 20], [-0.4, 1.4], color=C["ink"], lw=2, clip_on=False)
    cv.text(696, 938, "drop at 20 mm", 15, C["ink"], ha="right", va="bottom")
    cv.line(0, 1004, 720, 1004)
    cv.text(24, 1014, "Whole case: 3 ft, 23 kg", 20, C["sub"], "bold")
    st = rec.meta["static"]
    W2 = st["case"]["half"][0] * 1e3
    case_z = (rec.body("case", t_ref)[1] - rec.body("carriage", t_ref)[1]) * 1e3
    cax = cv.axes(10, 1040, 700, 236)
    cwin = fit(0.0, case_z - 70, W2 + 170, 330, 700 / 236)
    ctx_scene = t5_scene(cax, rec, cwin, lw=0.6)
    cax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0], win[3] - win[2], fill=False, ec=C["stop"], lw=1.4,
                            ls=(0, (3, 2)), zorder=8))
    cax.text(-W2 - 30, cwin[2] + 10, "B", fontsize=fs(22), weight="bold", ha="right", zorder=9)
    cax.text(W2 + 30, cwin[2] + 10, "A", fontsize=fs(22), weight="bold", ha="left", zorder=9)
    arrow = cax.annotate("", xy=(W2 * 0.55, case_z), xytext=(-W2 * 0.3, case_z), zorder=9,
                         arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.8", "lw": 4, "color": C["stop"]})
    segs = [(-1.0, 0.0, 0.8, "Close on the case", C["calm"]),
            (0.0, v_cut - 0.15, 1.5, "Lift, shake, hold", C["calm"]),
            (v_cut - 0.15, v_cut, 0.6, "Drives on, holding", C["calm"]),
            (v_cut, v_stop, 0.8, "POWER CUT, brakes on", C["warn"]),
            (v_stop, v_end, 2.6, "1 g STOP", C["stop"])]
    frames = plan(segs, 2.2, "DROPS: slip passes 20 mm", C["fail"])
    q_cut = {j: rec.joint(j, rec.run_time(v_cut)) for j in ("jAR", "jAL", "jBR", "jBL")}

    def rel_at(t: float) -> np.ndarray:
        return (rec.body("case", t)[:2] - rec.body("carriage", t)[:2]) * 1e3

    # The harness's slip: the case against the carriage from the first sample of the hold (t_lift on).
    rel_hold = rel_at(float(tl["t_lift"]) + 1e-6)
    rel_cut = rel_at(float(tl["t_power"]) + 1e-6)
    stills = {"hold": v_cut - 0.05, "cut": v_stop, "mid": v_stop + 0.05, "end": frames[-1][0]}
    tiles: dict[str, np.ndarray] = {}
    enc = Encoder(dest / "clip2-p0-23kg-power-cut-drop.mp4")
    for i, (v, vp, phase, speed, color) in enumerate(frames):
        t = min(rec.run_time(v), rec.t_last)
        banner.set(phase, speed, color, None if v < 0 else t)
        main.update(t)
        ctx_scene.update(t)
        for s_ in (main, ctx_scene):
            s_.deck.set_visible(v < v_cut - 0.3)
        arrow.set_visible(v >= v_stop)
        cut = v >= v_cut
        openA = math.degrees(max(rec.joint("jAR", t) - q_cut["jAR"], 0.0)) if cut else 0.0
        openB = math.degrees(max(rec.joint("jBR", t) - q_cut["jBR"], 0.0)) if cut else 0.0
        jaw_txt.set_text(f"{openA:.1f}°")
        jaw_txt.set_color(C["fail"] if openA > 3 else C["ink"])
        jawB.set_text(f"per finger · gripper B opens {openB:.2f}°")
        rel = rel_at(t)
        ended = rec.run_time(v) >= rec.t_last
        slip = (float(rec.row["slip_all_mm"]) if ended else
                float(np.hypot(*(rel - rel_hold))) if rec.run_time(v) >= float(tl["t_lift"]) else 0.0)
        after = float(rec.row["power_slip_mm"]) if ended else float(np.hypot(*(rel - rel_cut)))
        slip_txt.set_text(f"{slip:.1f} mm")
        slip_txt.set_color(C["fail"] if ended else C["ink"])
        slip_cut.set_text(f"{after:.1f} mm of it after the power cut" if cut else "")
        sbar.set_width(min(slip, 22))
        sbar.set_facecolor(C["fail"] if ended else C["warn"])
        if i >= len(frames) - int(round(2.2 * FPS)):
            sub.set_text("The brakes saturate, the jaw back-drives, the case drops")
            sub.set_color(C["fail"])
        frame = cv.grab()
        enc.write(frame)
        for key, vs in stills.items():
            if key not in tiles and v >= vs - 1e-9:
                tiles[key] = frame
    enc.close()
    filmstrip([(tiles["hold"], "Holding, drives on"), (tiles["cut"], "Power cut: brakes only"),
               (tiles["mid"], "1 g stop +50 ms: jaw opens"), (tiles["end"], "+0.1 s: slip hits 20 mm")],
              "Today's finger, 23 kg case, power cut first", dest / "clip2-p0-23kg-power-cut-drop-filmstrip.png")
    return {"open_A_deg": math.degrees(rec.joint("jAR", rec.t_last) - q_cut["jAR"]),
            "open_B_deg": math.degrees(rec.joint("jBR", rec.t_last) - q_cut["jBR"]),
            "power_slip_mm": rec.row["power_slip_mm"], "dropped_in": rec.row["dropped_in"]}


# ----------------------------------------------------------------------------------------------- clip 3

def clip_box(run_dir: Path, dest: Path) -> dict[str, Any]:
    """Family Size at catalog fill: today's finger beside S2, where each grips against the cereal line."""
    from graspsim2d import batch1
    obj = batch1.get_obj(BOX, "catalog")
    H, Wd = obj.height * 1e3, obj.thickness * 1e3
    fill_top = obj.fill_height * 1e3
    recs = {d: Rec(f"box_{d}", run_dir) for d in ("P0", "S2")}
    cv = Canvas()
    cv.text(20, 14, "Cheerios Family Size, catalog fill", 31, weight="bold")
    sub = cv.text(20, 58, "Where each grip sits, and the 0.5 g shake", 22, C["sub"])
    banner = Banner(cv, 96)
    cv.line(0, 160, 720, 160)
    cv.line(360, 172, 360, 1150)
    scenes, slip_txt, verdict_txt, info = {}, {}, {}, {}
    from matplotlib.patches import Circle
    for col, d in enumerate(("P0", "S2")):
        rec = recs[d]
        x = col * 360
        cv.text(x + 16, 172, d, 27, weight="bold")
        cv.text(x + 64, 177, NAMES[d], 20, C["sub"])
        t_hold = float(rec.tl["t_lift"])
        base = rec.body("base", t_hold)
        boxp = rec.body("obj", t_hold)
        top_z = (boxp[1] * 1e3 + H / 2) - base[1] * 1e3
        win = fit(0.0, top_z - 205, 118, 300, 356 / 820)
        ax = cv.axes(x + 2, 214, 356, 820)
        sc = Scene(ax, rec, win, "base", 1.2)
        st = rec.meta["static"]
        tx, tz = (v * 1e3 for v in st["table"]["half"])
        px_, pz_ = (v * 1e3 for v in st["table"]["pos"])
        sc.add_world(rect(px_ - tx, px_ + tx, pz_ - tz, pz_ + tz), C["table"], C["hub_edge"], 0.5)
        sc.add_gripper("base", ("R", "L"))
        hx, hz = Wd / 2, H / 2
        sc.add(rect(-hx, hx, -hz, hz), "obj", C["empty"], C["case_edge"], 3, lw=1.8)
        sc.add(rect(-hx, hx, -hz, -hz + fill_top), "obj", C["cereal"], "none", 3.1, lw=0)
        sc.add(rect(-hx, hx, -hz + fill_top - 0.9, -hz + fill_top + 0.9), "obj", C["fill_line"], "none", 3.2, lw=0)
        sc.add(rect(-hx, hx, -hz, hz), "obj", "none", C["case_edge"], 3.3, lw=1.8)
        sc.labels = [(ax.text(0, 0, f"no cereal\nin the top\n{H - fill_top:.0f} mm", fontsize=fs(16),
                              color=C["fill_line"], ha="center", va="center", zorder=9, linespacing=1.15),
                      np.array([0.0, hz - (H - fill_top) * 0.5])),
                     (ax.text(0, 0, "cereal", fontsize=fs(20), color=C["fill_line"], ha="center", va="center",
                              weight="bold", zorder=9), np.array([0.0, -hz + fill_top * 0.5]))]
        N, X, Z = rec.loads(t_hold - 0.02, t_hold)
        grip_low = []
        sc.dots = []
        for part in ("R", "L"):
            c = Circle((0, 0), 4.5, fc=C["fail"], ec="white", lw=1.2, zorder=8)
            ax.add_patch(c)
            sc.dots.append((c, rec.p[part]))
            if N[rec.p[part]] > 0.5:
                grip_low.append((boxp[1] * 1e3 + H / 2) - Z[rec.p[part]])
        depth = float(rec.meta["mj_job"]["depth_mm"])
        info[d] = {"depth_mm": depth, "contact_below_top_mm": [round(float(g), 1) for g in grip_low],
                   "grip_N": rec.row["grip_N"], "dropped_in": rec.row["dropped_in"],
                   "slip_mm": rec.row["slip_mm"], "t_drop_after_shake_start_s": rec.t_last - float(rec.tl["t_hold"])}
        sc.labels.append((ax.text(0, 0, "grip on\nthe top\nedge", fontsize=fs(16), color=C["fail"],
                                  ha="left", va="top", weight="bold", zorder=9, linespacing=1.1),
                          np.array([hx + 5, hz - depth - 8])))
        sc.slip = t1_slip(rec)
        info[d]["drawn_final_slip_mm"] = sc.slip(rec.t_last)
        scenes[d] = sc
        slip_txt[d] = cv.text(x + 16, 1040, "", 24, C["ink"], "bold")
        verdict_txt[d] = cv.text(x + 16, 1078, "", 22, C["fail"], "bold")
        info[d]["verdict"] = ("SLIDES OUT" if rec.row["dropped_in"] == "main" else
                              "DROPS" if rec.row["dropped_in"] else "HOLDS")
    cv.line(0, 1150, 720, 1150)
    cv.text(20, 1164, f"The fingertips go {info['P0']['depth_mm']:.0f} mm in, but both grip the box at its top",
            19, C["ink"])
    cv.text(20, 1192, f"edge, {H - fill_top:.0f} mm above the cereal, and slide out in the shake.", 19, C["ink"])
    cv.text(20, 1232, "The harness counts a slip past 20 mm as a drop and ends the run there.", 16, C["sub"])
    vd = {d: float(recs[d].t_last - recs[d].tl["t_settle"]) for d in recs}
    segs = [(-1.0, 0.0, 1.0, "Close on the box", C["calm"]),
            (0.0, 0.8, 0.8, "Lift 80 mm, hold", C["calm"]),
            (0.8, vd["P0"] - 0.14, 0.9, "Shake 0.5 g", C["stop"]),
            (vd["P0"] - 0.14, vd["P0"] + 0.02, 1.6, "Shake 0.5 g", C["stop"]),
            (vd["P0"] + 0.02, vd["S2"] - 0.14, 1.0, "Shake 0.5 g", C["stop"]),
            (vd["S2"] - 0.14, vd["S2"] + 0.01, 1.5, "Shake 0.5 g", C["stop"])]
    frames = plan(segs, 1.6, "Neither holds", C["fail"])
    stills = {"close": -0.01, "lift": 0.7, "p0": vd["P0"] + 0.01, "end": frames[-1][0]}
    tiles: dict[str, np.ndarray] = {}
    enc = Encoder(dest / "clip3-family-size-p0-vs-s2.mp4")
    for i, (v, vp, phase, speed, color) in enumerate(frames):
        banner.set(phase, speed, color, None if v < 0 else recs["P0"].run_time(v))
        for d, sc in scenes.items():
            rec = recs[d]
            t = min(rec.run_time(v), rec.t_last)
            off = sc.update(t)
            boxp = rec.body("obj", t)
            for txt, loc in sc.labels:
                txt.set_position(tuple(tf(boxp, loc[None, :])[0] - off))
            N, X, Z = rec.loads(t - 0.002, t)
            for c, pi in sc.dots:
                c.set_visible(N[pi] > 0.5)
                c.center = (X[pi] - off[0], Z[pi] - off[1])
            done = rec.run_time(v) >= rec.t_last
            slip = sc.slip(t)
            slip_txt[d].set_text(f"slip {slip:.1f} mm")
            slip_txt[d].set_color(C["fail"] if done else C["ink"])
            verdict_txt[d].set_text(f"SLIDES OUT, {info[d]['t_drop_after_shake_start_s']:.1f} s\ninto the shake"
                                    if done and info[d]["verdict"] == "SLIDES OUT" else (info[d]["verdict"] if done else ""))
        frame = cv.grab()
        enc.write(frame)
        for key, vs in stills.items():
            if key not in tiles and v >= vs - 1e-9:
                tiles[key] = frame
    enc.close()
    filmstrip([(tiles["close"], "Closed on the box"), (tiles["lift"], "Lifted: grip vs cereal line"),
               (tiles["p0"], "P0 slides out in the shake"), (tiles["end"], "S2 slides out too")],
              "Family Size pick: today's finger beside S2", dest / "clip3-family-size-p0-vs-s2-filmstrip.png")
    return info


def cmd_render(args: argparse.Namespace) -> None:
    run_dir, dest = Path(args.out), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    fns = {"1": clip_lift12, "2": clip_cut23, "3": clip_box}
    report = {}
    for k in args.clips:
        t0 = time.time()
        report[k] = fns[k](run_dir, dest)
        print(f"clip {k}: {time.time() - t0:.0f} s", flush=True)
    (run_dir / "render-report.json").write_text(json.dumps(_jsonable(report), indent=1))
    print(json.dumps(_jsonable(report), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(RUN_DIR))
    r.add_argument("--only", nargs="*")
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    r = sub.add_parser("render")
    r.add_argument("--out", default=str(RUN_DIR))
    r.add_argument("--dest", default=str(RUN_DIR / "clips"))
    r.add_argument("--clips", nargs="*", default=["1", "2", "3"])
    r.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
