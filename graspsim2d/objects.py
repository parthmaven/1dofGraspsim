"""Predefined object library: filled cereal boxes and cardboard sheets.

Everything is described in the grasp plane. A box stands upright and the jaw
closes across its depth (front to back), so the 2D section is depth x height
and the box width is out of plane. A sheet hangs from a pinch near its free
edge, so the section is thickness x hanging length and the sheet width is
out of plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

G = 9.81


@dataclass(frozen=True)
class Material:
    """Surface and board properties seen by the pad. SI units."""

    name: str
    friction: float  # pad on this surface, design value
    k_surface: float  # Pa/m: through-thickness stiffness per area at the contact
    allow_pressure: float  # Pa: crush or dent allowable under a pad
    grammage: float = 0.0  # kg/m^2
    bending_stiffness: float = 0.0  # N*m (EI per metre of width), sheets
    bending_strength: float = 0.0  # N*m/m: bending moment per width that creases the board
    shear_friction_note: str = ""
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspObject:
    """One object in the grasp plane. Lengths in m, mass in kg."""

    id: str
    label: str
    kind: str  # "box" or "sheet"
    thickness: float  # m: box depth or sheet caliper (the jaw closes across it)
    height: float  # m: in-plane vertical extent (box height, sheet hanging length)
    width: float  # m: out of plane
    mass: float  # kg, gross
    com_below_top: float  # m
    inertia: float  # kg m^2 about the COM, out of plane
    material: Material
    headspace: float = 0.0  # m of unsupported panel below the top of a box
    unsupported_force: float = float("inf")  # N per pad allowed on an unsupported panel
    net_mass: float = 0.0
    packaging_mass: float = 0.0
    notes: str = ""
    sources: tuple[str, ...] = ()

    @property
    def weight(self) -> float:
        return self.mass * G

    def summary(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "grasp_dimension_mm": round(self.thickness * 1000, 2),
            "height_mm": round(self.height * 1000, 1),
            "width_mm": round(self.width * 1000, 1),
            "mass_g": round(self.mass * 1000, 1),
            "friction": self.material.friction,
            "allow_pressure_kPa": round(self.material.allow_pressure / 1000, 1),
            "material": self.material.name,
        }
        if self.kind == "box":
            d["headspace_mm"] = round(self.headspace * 1000, 1)
            d["unsupported_force_N"] = self.unsupported_force
        else:
            d["bending_stiffness_Nm"] = self.material.bending_stiffness
            d["bending_strength_Nm_per_m"] = self.material.bending_strength
        return d


@dataclass
class ObjectLibrary:
    objects: list[GraspObject]
    materials: dict[str, Material]
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def by_id(self, oid: str) -> GraspObject:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(f"no object {oid!r}; have {[o.id for o in self.objects]}")

    def ids(self) -> list[str]:
        return [o.id for o in self.objects]


def _material(name: str, e: Mapping[str, Any]) -> Material:
    if "contact_modulus_MPa" in e:
        # Box wall: panel backed by product, an elastic foundation of the given depth.
        k = float(e["contact_modulus_MPa"]) * 1e6 / (float(e["contact_depth_mm"]) * 1e-3)
    else:
        # Sheet pinched from both sides: each pad compresses half the caliper.
        k = float(e["zd_modulus_MPa"]) * 1e6 / (0.5 * float(e["caliper_mm"]) * 1e-3)
    return Material(
        name=name,
        friction=float(e["friction"]),
        k_surface=k,
        allow_pressure=float(e["allow_pressure_kPa"]) * 1e3,
        grammage=float(e.get("grammage_gsm", 0.0)) / 1000.0,
        bending_stiffness=float(e.get("bending_stiffness_Nm", 0.0)),
        bending_strength=float(e.get("bending_strength_Nm_per_m", 0.0)),
        sources=tuple(e.get("sources", []) or []),
    )


def _box(e: Mapping[str, Any], mat: Material) -> GraspObject:
    """Filled cereal box: carton shell plus settled contents."""
    B, H, W = (float(e[k]) / 1000.0 for k in ("depth_mm", "height_mm", "width_mm"))
    net = float(e["net_mass_g"]) / 1000.0
    fill = float(e.get("fill_fraction", 0.8))
    if "packaging_mass_g" in e:
        pack = float(e["packaging_mass_g"]) / 1000.0
    else:
        # Blank area of a tuck-end carton with glue flap, times board grammage.
        pack = 1.15 * 2 * (W * H + B * H + W * B) * mat.grammage
    # Shell: front/back panels at x = +-B/2, side panels across the section,
    # top/bottom at the ends. Contents fill the bottom `fill` of the height.
    a_fb, a_s, a_tb = 2 * W * H, 2 * B * H, 2 * W * B
    a = a_fb + a_s + a_tb
    m_fb, m_s, m_tb = pack * a_fb / a, pack * a_s / a, pack * a_tb / a
    i_shell = m_fb * (B**2 / 4 + H**2 / 12) + m_s * (B**2 + H**2) / 12 + m_tb * (H**2 / 4 + B**2 / 12)
    hf = fill * H
    z_shell, z_cont = H / 2, hf / 2  # above the bottom
    m = pack + net
    z_com = (pack * z_shell + net * z_cont) / m
    i_cont = net * (B**2 + hf**2) / 12
    inertia = i_shell + pack * (z_shell - z_com) ** 2 + i_cont + net * (z_cont - z_com) ** 2
    return GraspObject(
        id=str(e["id"]),
        label=str(e["label"]),
        kind="box",
        thickness=B,
        height=H,
        width=W,
        mass=m,
        com_below_top=H - z_com,
        inertia=inertia,
        material=mat,
        headspace=(1 - fill) * H,
        unsupported_force=float(e.get("unsupported_panel_force_N", "inf")),
        net_mass=net,
        packaging_mass=pack,
        notes=str(e.get("notes", "")),
        sources=tuple(e.get("sources", []) or []),
    )


def _sheet(e: Mapping[str, Any], mat: Material, caliper_mm: float) -> GraspObject:
    t = caliper_mm / 1000.0
    H = float(e["hang_length_mm"]) / 1000.0
    W = float(e["width_mm"]) / 1000.0
    m = mat.grammage * H * W
    return GraspObject(
        id=str(e["id"]),
        label=str(e["label"]),
        kind="sheet",
        thickness=t,
        height=H,
        width=W,
        mass=m,
        com_below_top=H / 2,
        inertia=m * (H**2 + t**2) / 12,
        material=mat,
        notes=str(e.get("notes", "")),
        sources=tuple(e.get("sources", []) or []),
    )


def load_objects(path: str | Path) -> ObjectLibrary:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    mats_raw = raw.get("materials", {}) or {}
    materials = {name: _material(name, e) for name, e in mats_raw.items()}
    objects: list[GraspObject] = []
    sheet_defaults = raw.get("sheet_defaults", {}) or {}
    box_defaults = raw.get("box_defaults", {}) or {}
    for e in raw.get("boxes", []) or []:
        e = {**box_defaults, **e}
        if e.get("enabled", True):
            objects.append(_box(e, materials[e["material"]]))
    for e in raw.get("sheets", []) or []:
        e = {**sheet_defaults, **e}
        if e.get("enabled", True):
            mname = e["material"]
            objects.append(_sheet(e, materials[mname], float(mats_raw[mname]["caliper_mm"])))
    enabled = raw.get("enabled")
    if enabled:
        objects = [o for o in objects if o.id in set(enabled)]
    ids = [o.id for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate object ids")
    return ObjectLibrary(objects=objects, materials=materials, source=str(path), meta=raw.get("meta", {}) or {})
