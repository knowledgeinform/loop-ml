"""Charge-feasibility filter for the prototype tournament (scoring-time).

Relaxation is done once for every (composition, prototype); plausibility
gates are applied at training so the corpus never needs re-relaxing.

Rule: with O fixed at -2, a prototype host demands a fixed mean cation
charge (e.g. MO rocksalt +2, AO2 fluorite +4).  A cation mixture is
feasible on that host iff the host target lies within [min, max] of
sum(x_e * os_e) over each element's common solid-state oxidation states
(the convex bound of charge-neutral assignments).
"""
from __future__ import annotations

OXIDATION_STATES = {
    "Li": [1],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2], "Cd": [2], "Zn": [2],
    "Cu": [1, 2], "Ag": [1], "Au": [1, 3],
    "Al": [3], "Ga": [3], "In": [3], "B": [3],
    "Sc": [3], "Y": [3], "La": [3], "Ce": [3, 4], "Pr": [3, 4], "Nd": [3],
    "Sm": [2, 3], "Eu": [2, 3], "Gd": [3], "Tb": [3, 4], "Dy": [3], "Ho": [3],
    "Er": [3], "Tm": [3], "Yb": [2, 3], "Lu": [3],
    "Ti": [3, 4], "Zr": [4], "Hf": [4], "V": [3, 4, 5], "Nb": [5], "Ta": [5],
    "Cr": [2, 3], "Mo": [4, 5, 6], "W": [4, 5, 6], "Ru": [3, 4],
    "Mn": [2, 3, 4], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3],
    "Si": [4], "Ge": [4], "Sn": [2, 4], "Pb": [2, 4], "Bi": [3], "Sb": [3, 5],
}

# mean cation charge per cation required by each host (O = -2)
HOST_TARGET_CHARGE = {
    "rocksalt": 2.0,   # MO
    "cscl": 2.0,       # MO (CsCl-ordered)
    "fluorite": 4.0,   # AO2
    "perovskite": 3.0, # ABO3 -> (A+B)/2
    "spinel": 8.0 / 3.0,  # AB2O4 -> (A+2B)/3
    "pyrochlore": 7.0 / 2.0,  # A2B2O7 kept for offline analyses
}


def charge_window(cat_fracs: dict[str, float]) -> tuple[float, float]:
    """[min, max] attainable mean cation charge for a cation mixture."""
    total = sum(cat_fracs.values())
    if total <= 0:
        raise ValueError("empty cation mixture")
    lo = hi = 0.0
    for el, x in cat_fracs.items():
        states = OXIDATION_STATES[el]
        w = x / total
        lo += w * min(states)
        hi += w * max(states)
    return lo, hi


def charge_feasible(prototype: str, cat_fracs: dict[str, float],
                    band: float = 0.25) -> bool:
    """band: per-cation slack for oxygen vacancy/interstitial compensation
    (|delta O| up to ~0.12-0.25 per cation is common in oxides)."""
    target = HOST_TARGET_CHARGE[prototype]
    lo, hi = charge_window(cat_fracs)
    return lo - band <= target <= hi + band


def feasible_prototypes(cat_fracs: dict[str, float],
                        prototypes: list[str]) -> list[str]:
    return [p for p in prototypes
            if all(e in OXIDATION_STATES for e in cat_fracs)
            and charge_feasible(p, cat_fracs)]
