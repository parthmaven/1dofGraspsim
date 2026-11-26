"""Elastomer pad stiffness (lit review section 3.1).

Shore A to Young's modulus: Gent (1958), doi:10.5254/1.3542351.
Bonded-pad compression modulus: Gent & Lindley (1959),
doi:10.1243/PIME_PROC_1959_173_022_02, E_c = E (1 + 2 S^2) with shape factor
S = loaded area / free (bulging) area, capped by the bulk modulus.
"""

from __future__ import annotations

BULK_MODULUS = 2.0e9  # Pa, rubber; caps E_c for very thin pads


def shore_a_to_E(shore: float) -> float:
    """Young's modulus (Pa) from Shore A hardness, Gent's empirical relation."""
    s = min(max(shore, 5.0), 95.0)
    return 1e6 * 0.0981 * (56.0 + 7.62336 * s) / (0.137505 * (254.0 - 2.54 * s))


def shape_factor(length: float, width: float, thickness: float) -> float:
    return (length * width) / (2.0 * (length + width) * thickness)


def compression_modulus(shore: float, length: float, width: float, thickness: float) -> float:
    """Apparent compression modulus (Pa) of a pad bonded on one face to the finger."""
    E = shore_a_to_E(shore)
    S = shape_factor(length, width, thickness)
    ec = E * (1.0 + 2.0 * S * S)
    return 1.0 / (1.0 / ec + 1.0 / BULK_MODULUS)


def foundation_stiffness(shore: float, length: float, width: float, thickness: float) -> float:
    """Normal stiffness per area k = E_c / t (Pa/m), Winkler foundation."""
    return compression_modulus(shore, length, width, thickness) / thickness


def shear_stiffness(shore: float, thickness: float) -> float:
    """Tangential stiffness per area G / t (Pa/m), with G = E/3 for rubber."""
    return shore_a_to_E(shore) / 3.0 / thickness


def series(*k: float) -> float:
    return 1.0 / sum(1.0 / max(x, 1e-12) for x in k)


def pad_k_area(shore: float, length_m: float, width_m: float, thickness_m: float, k_surface: float) -> float:
    """Pad and object surface in series (Pa/m)."""
    return series(foundation_stiffness(shore, length_m, width_m, thickness_m), k_surface)


def elastomer_note(shore: float) -> str:
    return f"Shore A{shore:.0f}: E = {shore_a_to_E(shore) / 1e6:.2f} MPa"
