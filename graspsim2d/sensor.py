"""The wrist force/torque sensor's reading (test conditions revision 2.34).

The finger assembly, the drive and brake, and the hub shell all ride on the sensor's moving side, so the sensor reads
the sum of every external contact on the jaw assembly, fingers and hub. The pinch between the fingers cancels in that
sum. The sensor's axis is the hub's and the tool's, square to the jaw plane (MuJoCo's y), and its centre sits on the
hub's axis (`sensor.centre_mm`, gripper coordinates: x across, z along the centreline with the tips at -z), `depth_mm`
along that axis below the top datum. The model extrudes every part about y = 0, so a contact's place along the axis
comes from its part's depth, not from MuJoCo.

Per gripper, at every sample:
* `sensor_F_N`: the in-plane force;
* `sensor_T_Nm`: the jaw-plane moment about the sensor's centre, the torsion about its axis;
* `sensor_Mb_Nm`: the bending, each contact's in-plane force times its offset along the tool axis from the sensor's
  centre, the offset taken from the contact's layer.
Each is read as its largest 5 ms average over the whole window and as the final hold's mean. The components are
averaged first, as a filtered sensor would, and the magnitude taken after.

The specs don't say where a finger's width sits along the tool axis, so `finger_span_mm` sets it: a full-width part,
and comb layer a, hang from its top; comb layer b sits at its bottom.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Mapping

import numpy as np

KEYS = ("F", "T", "Mb")
UNITS = {"F": "N", "T": "Nm", "Mb": "Nm"}


def config(b1: Mapping[str, Any]) -> dict[str, Any]:
    c = dict(b1.get("sensor") or {})
    c.setdefault("centre_mm", (0.0, -31.0))
    c.setdefault("depth_mm", 24.0)
    c.setdefault("finger_span_mm", (1.0, 156.0))
    c.setdefault("hub_span_mm", (14.5, 67.3))
    c.setdefault("average_s", 0.005)
    return c


def layer_depth(b1: Mapping[str, Any], js, u: float, v: float, width_mm: float) -> float:
    """The middle of a finger's width at a finger-local point, in mm along the tool axis below the top datum."""
    top, bottom = (float(x) for x in config(b1)["finger_span_mm"])
    layers = js.layers_at(u, v) if hasattr(js, "layers_at") else None
    if layers is not None and set(layers) == {"b"}:
        return bottom - 0.5 * width_mm
    return top + 0.5 * width_mm


def hub_depth(b1: Mapping[str, Any]) -> float:
    lo, hi = (float(x) for x in config(b1)["hub_span_mm"])
    return 0.5 * (lo + hi)


def contact_forces(model, data, own: np.ndarray):
    """Contacts between this gripper's geoms and anything else: (own geom, force on it, world point)."""
    n = data.ncon
    if n == 0:
        return None
    con = data.contact
    g1, g2 = con.geom1[:n], con.geom2[:n]
    adr = con.efc_address[:n]
    m1, m2 = own[g1], own[g2]
    sel = (m1 != m2) & (adr >= 0)
    if not sel.any():
        return None
    idx = np.nonzero(sel)[0]
    a = adr[idx]
    fr = con.frame[idx]
    dim = con.dim[idx]
    ef = data.efc_force
    t1 = np.where(dim > 1, ef[np.minimum(a + 1, len(ef) - 1)], 0.0)
    t2 = np.where(dim > 2, ef[np.minimum(a + 2, len(ef) - 1)], 0.0)
    f_on2 = ef[a][:, None] * fr[:, 0:3] + t1[:, None] * fr[:, 3:6] + t2[:, None] * fr[:, 6:9]
    mine2 = m2[idx]
    return np.where(mine2, g2[idx], g1[idx]), np.where(mine2[:, None], f_on2, -f_on2), con.pos[idx]


class Wrench:
    """One gripper's sensor. `geoms` maps each geom of the jaw assembly to its depth (mm below the top datum);
    `body` is the gripper's frame, in which the sensor's centre is fixed; a mirrored gripper flips its x."""

    def __init__(self, model, b1: Mapping[str, Any], body: str, geoms: Mapping[str, float], sample_s: float,
                 mirror: bool = False):
        c = config(b1)
        cx, cz = (float(x) * 1e-3 for x in c["centre_mm"])
        self.body = int(model.body(body).id)
        self.local = np.array([-cx if mirror else cx, 0.0, cz])
        self.depth_mm = float(c["depth_mm"])
        self.own = np.zeros(model.ngeom, bool)
        self.arm = np.zeros(model.ngeom)
        for name, d in geoms.items():
            gid = int(model.geom(name).id)
            self.own[gid] = True
            self.arm[gid] = (float(d) - self.depth_mm) * 1e-3
        self.n = max(1, int(round(float(c["average_s"]) / sample_s)))
        self.average_s = self.n * sample_s
        self.buf: deque[np.ndarray] = deque()
        self.acc = np.zeros(5)
        self.peak = {k: [0.0, math.nan, ""] for k in KEYS}
        self.by_phase: dict[str, dict[str, float]] = {k: {} for k in KEYS}
        self.sample_peak = {k: 0.0 for k in KEYS}
        self.hold_acc = np.zeros(5)
        self.hold_n = 0

    def read(self, model, data) -> np.ndarray:
        """(Fx, Fz, T, Mb_x, Mb_z) now: the in-plane force, the moment about the centre and the bending vector's
        components (the bending is about the in-plane axis square to each force, so its size is |sum d F|)."""
        got = contact_forces(model, data, self.own)
        if got is None:
            return np.zeros(5)
        gid, f, pos = got
        centre = data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ self.local
        r = pos - centre
        fx, fz = f[:, 0], f[:, 2]
        d = self.arm[gid]
        return np.array([fx.sum(), fz.sum(), float((r[:, 2] * fx - r[:, 0] * fz).sum()),
                         float((d * fx).sum()), float((d * fz).sum())])

    @staticmethod
    def sizes(w: np.ndarray) -> dict[str, float]:
        return {"F": math.hypot(w[0], w[1]), "T": abs(float(w[2])), "Mb": math.hypot(w[3], w[4])}

    def sample(self, model, data, t: float, phase: str, window: bool, hold: bool) -> None:
        """Call at every sample from the end of the close: each phase keeps its own 5 ms peak; `window` marks the
        rated window, which gives the run's peak, and `hold` the final hold's span for its mean."""
        w = self.read(model, data)
        if hold:
            self.hold_acc += w
            self.hold_n += 1
        sizes = self.sizes(w)
        if window:
            for k, v in sizes.items():
                self.sample_peak[k] = max(self.sample_peak[k], v)
        self.buf.append(w)
        self.acc += w
        if len(self.buf) > self.n:
            self.acc -= self.buf.popleft()
        if len(self.buf) < self.n:
            return
        for k, v in self.sizes(self.acc / self.n).items():
            if v > self.by_phase[k].get(phase, -1.0):
                self.by_phase[k][phase] = v
            if window and v > self.peak[k][0]:
                self.peak[k] = [v, t, phase]

    def readings(self, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        hold = self.sizes(self.hold_acc / self.hold_n) if self.hold_n else {k: math.nan for k in KEYS}
        for k in KEYS:
            key = f"{prefix}sensor_{k}_{UNITS[k]}"
            v, t, ph = self.peak[k]
            out[key] = v
            out[f"{key}_at_s"] = t
            out[f"{key}_phase"] = ph
            out[f"{key}_hold"] = hold[k]
            out[f"{key}_sample_peak"] = self.sample_peak[k]
            out[f"{key}_by_phase"] = {p: round(x, 3) for p, x in self.by_phase[k].items()}
        return out


class Rolling:
    """A scalar's largest n-sample average in each phase."""

    def __init__(self, n: int):
        self.n = max(1, int(n))
        self.buf: deque[float] = deque()
        self.acc = 0.0
        self.by_phase: dict[str, float] = {}

    def add(self, v: float, phase: str) -> None:
        self.buf.append(v)
        self.acc += v
        if len(self.buf) > self.n:
            self.acc -= self.buf.popleft()
        if len(self.buf) == self.n:
            a = self.acc / self.n
            if a > self.by_phase.get(phase, -1.0):
                self.by_phase[phase] = a

    def phases(self, digits: int = 2) -> dict[str, float]:
        return {p: round(v, digits) for p, v in self.by_phase.items()}


def worst(rows: Mapping[str, Any], grippers: str) -> dict[str, Any]:
    """The larger gripper's reading of each quantity, for a two-gripper run."""
    out: dict[str, Any] = {}
    for k in KEYS:
        key = f"sensor_{k}_{UNITS[k]}"
        g = max(grippers, key=lambda g: rows.get(f"{g}_{key}", 0.0) or 0.0)
        out[key] = rows.get(f"{g}_{key}")
        out[f"{key}_hold"] = max((rows.get(f"{x}_{key}_hold") or 0.0) for x in grippers)
        out[f"{key}_gripper"] = g
    return out
