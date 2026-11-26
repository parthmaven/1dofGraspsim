"""Predefined object library: filled cereal boxes and cardboard sheets.

Everything is described in the grasp plane. A box stands upright, the gripper
comes down on its top end and the jaw closes across the depth (front to back),
so the 2D section is the depth x height side panel and the box width is out of
plane (`grasp_axis: width` closes across the width instead). A
sheet hangs from a pinch near its free edge, so the section is thickness x
hanging length and the sheet width is out of plane.

Boxes come from the `boxes` list in the YAML file and from CSV box tables
(`box_tables`), one box per row with the columns in BOX_TABLE_COLUMNS.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

G = 9.81
GRASP_AXES = ("depth", "width")
BOX_TABLE_COLUMNS = ("product_title", "size_label", "upc", "net_mass_g", "box_mass_g", "height_mm", "width_mm",
                     "depth_mm", "fill_height_mm", "com_height_mm", "bulk_density_g_cm3", "source", "confidence")
_BOX_TABLE_REQUIRED = ("product_title", "net_mass_g", "height_mm", "width_mm", "depth_mm")


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
    bare_friction: float = 0.3  # bare printed finger on this surface


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
    fill_height: float = 0.0  # m of settled contents above the bottom (boxes)
    grasp_axis: str = ""  # boxes: the box dimension the jaw closes across
    confidence: str = ""
    notes: str = ""
    sources: tuple[str, ...] = ()

    @property
    def com_height(self) -> float:
        """Centre of mass above the bottom (boxes) or below the pinch line (sheets), m."""
        return self.height - self.com_below_top

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
            d["grasp_axis"] = self.grasp_axis
            d["fill_height_mm"] = round(self.fill_height * 1000, 1)
            d["com_height_mm"] = round(self.com_height * 1000, 1)
            d["headspace_mm"] = round(self.headspace * 1000, 1)
            d["unsupported_force_N"] = self.unsupported_force
            if self.confidence:
                d["confidence"] = self.confidence
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
        bare_friction=float(e.get("bare_friction", 0.3)),
    )


def _given(e: Mapping[str, Any], key: str) -> float | None:
    v = e.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return float(v)


def _box(e: Mapping[str, Any], mat: Material) -> GraspObject:
    """Filled cereal box: carton shell plus contents settled at the bottom.

    Fill height: `fill_height_mm`, else net mass over `bulk_density_g_cm3` and
    the box footprint, else `fill_fraction` of the height. Centre of mass:
    `com_height_mm` above the bottom, else the shell-plus-contents estimate."""
    D, H, W = (float(e[k]) / 1000.0 for k in ("depth_mm", "height_mm", "width_mm"))
    axis = str(e.get("grasp_axis", "depth"))
    if axis not in GRASP_AXES:
        raise ValueError(f"box {e.get('id')}: grasp_axis must be one of {GRASP_AXES}, not {axis!r}")
    B, Wo = (D, W) if axis == "depth" else (W, D)  # across the jaw, out of plane
    net = float(e["net_mass_g"]) / 1000.0
    pack_g = _given(e, "packaging_mass_g")
    if pack_g is not None:
        pack = pack_g / 1000.0
    else:
        # Blank area of a tuck-end carton with glue flap, times board grammage.
        pack = 1.15 * 2 * (W * H + D * H + W * D) * mat.grammage
    hf_mm, rho = _given(e, "fill_height_mm"), _given(e, "bulk_density_g_cm3")
    if hf_mm is not None:
        hf = hf_mm / 1000.0
    elif rho:
        hf = net / (rho * 1000.0 * W * D)
    else:
        hf = float(e.get("fill_fraction", 0.8)) * H
    if not 0.0 < hf <= H * (1 + 1e-9):
        raise ValueError(f"box {e.get('id')}: fill height {hf * 1000:.1f} mm is outside the box height {H * 1000:.1f} mm")
    # Shell: front/back panels at x = +-B/2, side panels across the section,
    # top/bottom at the ends. Contents fill the bottom hf of the height.
    a_fb, a_s, a_tb = 2 * Wo * H, 2 * B * H, 2 * Wo * B
    a = a_fb + a_s + a_tb
    m_fb, m_s, m_tb = pack * a_fb / a, pack * a_s / a, pack * a_tb / a
    i_shell = m_fb * (B**2 / 4 + H**2 / 12) + m_s * (B**2 + H**2) / 12 + m_tb * (H**2 / 4 + B**2 / 12)
    z_shell, z_cont = H / 2, hf / 2  # above the bottom
    m = pack + net
    z_est = (pack * z_shell + net * z_cont) / m
    i_cont = net * (B**2 + hf**2) / 12
    inertia = i_shell + pack * (z_shell - z_est) ** 2 + i_cont + net * (z_cont - z_est) ** 2
    com_mm = _given(e, "com_height_mm")
    z_com = com_mm / 1000.0 if com_mm is not None else z_est
    return GraspObject(
        id=str(e["id"]),
        label=str(e["label"]),
        kind="box",
        thickness=B,
        height=H,
        width=Wo,
        mass=m,
        com_below_top=H - z_com,
        inertia=inertia,
        material=mat,
        headspace=H - hf,
        unsupported_force=float(e.get("unsupported_panel_force_N", "inf")),
        net_mass=net,
        packaging_mass=pack,
        fill_height=hf,
        grasp_axis=axis,
        confidence=str(e.get("confidence", "")),
        notes=str(e.get("notes", "")),
        sources=tuple(e.get("sources", []) or []),
    )


def with_fill(o: GraspObject, factor: float, tag: str) -> GraspObject:
    """The box with its fill height scaled by `factor` (contents still settled at the bottom):
    headspace, centre of mass and inertia follow. A measured centre of mass moves with the
    contents' own centroid."""
    if o.kind != "box" or factor == 1.0:
        return replace(o, id=f"{o.id}@{tag}", label=f"{o.label} ({tag} fill)") if o.kind == "box" else o
    H, B, Wo = o.height, o.thickness, o.width
    hf0 = o.fill_height
    hf = min(max(hf0 * factor, 1e-3), H)
    net, pack, m = o.net_mass, o.packaging_mass, o.mass
    z_com = o.com_height + net * (hf - hf0) / 2 / m
    a_fb, a_s, a_tb = 2 * Wo * H, 2 * B * H, 2 * Wo * B
    a = a_fb + a_s + a_tb
    m_fb, m_s, m_tb = pack * a_fb / a, pack * a_s / a, pack * a_tb / a
    i_shell = m_fb * (B**2 / 4 + H**2 / 12) + m_s * (B**2 + H**2) / 12 + m_tb * (H**2 / 4 + B**2 / 12)
    inertia = i_shell + pack * (H / 2 - z_com) ** 2 + net * (B**2 + hf**2) / 12 + net * (hf / 2 - z_com) ** 2
    return replace(o, id=f"{o.id}@{tag}", label=f"{o.label} ({tag} fill)", fill_height=hf, headspace=H - hf,
                   com_below_top=H - z_com, inertia=inertia)


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


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def table_path(entry: Mapping[str, Any], base: Path) -> Path:
    """A box table's file, resolved against the directory of the objects YAML."""
    p = Path(str(entry["file"])).expanduser()
    return p if p.is_absolute() else base / p


def read_box_table(path: str | Path) -> list[dict[str, str]]:
    """Rows of a box table CSV (see BOX_TABLE_COLUMNS), blank rows skipped."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in _BOX_TABLE_REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"{path}: box table is missing columns {missing}; expected {list(BOX_TABLE_COLUMNS)}")
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in reader]
    return [r for r in rows if r.get("product_title")]


def boxes_from_table(path: str | Path, defaults: Mapping[str, Any], materials: Mapping[str, Material]) -> list[GraspObject]:
    """One box per CSV row, on top of `defaults` (box_defaults plus the table entry).

    A row's id is `id_prefix` plus its UPC (or its title when there is none);
    `names`, keyed by UPC, gives rows a short id and label instead."""
    out = []
    prefix = str(defaults.get("id_prefix", ""))
    names = {str(k): v for k, v in (defaults.get("names") or {}).items()}
    for row in read_box_table(path):
        title = " ".join(x for x in (row["product_title"], row.get("size_label", "")) if x)
        e: dict[str, Any] = {k: v for k, v in defaults.items() if k not in ("file", "id_prefix", "enabled", "names")}
        name = names.get(row.get("upc", ""), {})
        e["id"] = prefix + str(name.get("id") or row.get("upc") or _slug(title))
        conf = row.get("confidence", "")
        e["label"] = str(name.get("label") or title) + (" (placeholder)" if conf.lower() == "placeholder" else "")
        for col, key in (("net_mass_g", "net_mass_g"), ("box_mass_g", "packaging_mass_g"), ("height_mm", "height_mm"),
                         ("width_mm", "width_mm"), ("depth_mm", "depth_mm"), ("fill_height_mm", "fill_height_mm"),
                         ("com_height_mm", "com_height_mm"), ("bulk_density_g_cm3", "bulk_density_g_cm3")):
            if row.get(col):
                e[key] = float(row[col])
        e["confidence"] = conf
        e["notes"] = f"UPC {row['upc']}" if row.get("upc") else ""
        e["sources"] = [row["source"]] if row.get("source") else []
        out.append(_box(e, materials[e["material"]]))
    return out


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
    for t in raw.get("box_tables", []) or []:
        if t.get("enabled", True):
            objects.extend(boxes_from_table(table_path(t, path.parent), {**box_defaults, **t}, materials))
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
