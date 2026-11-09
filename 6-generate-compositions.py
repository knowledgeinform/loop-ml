#!/usr/bin/env python3
"""Step 6: enumerate the screening composition space under constraints.

Screening means walking EVERY combination subject to the chosen
constraints -- not just the labeled materials.  Default constraint (the
one used in the deck): anion fixed to O, equimolar mixtures of 3--5
cations from a candidate pool, giving C(10,3)+C(10,4)+C(10,5) = 582
compositions.  Output is rows-compatible with step 1's format, so steps
2-4 treat screening compositions exactly like labeled ones (same corpus,
relaxations keyed by task id and shared, resumable).

Usage:
    uv run python 6-generate-compositions.py [--pool Al,Ca,Co,Cu,Fe,Mg,Mn,Ni,Sn,Zn]
        [--sizes 3,4,5] [--o-fraction 0.5] [--out data/compositions.jsonl]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loopml import protogen

DEFAULT_POOL = "Al,Ca,Co,Cu,Fe,Mg,Mn,Ni,Sn,Zn"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default=DEFAULT_POOL,
                    help="candidate cations (comma list)")
    ap.add_argument("--sizes", default="3,4,5",
                    help="comma list of cation counts per composition")
    ap.add_argument("--o-fraction", type=float, default=0.5,
                    help="mole fraction reserved for oxygen; cations share the rest "
                         "equimolar (protogen re-normalizes cations per sublattice, "
                         "so this only affects the canonical key)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "compositions.jsonl")
    args = ap.parse_args(argv)

    pool = sorted({e.strip() for e in args.pool.split(",") if e.strip()})
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if any(n < 2 or n > len(pool) for n in sizes):
        raise SystemExit(f"sizes {sizes} incompatible with pool of {len(pool)}")

    out, seen = [], set()
    for n in sizes:
        for combo in itertools.combinations(pool, n):
            frac = {e: (1.0 - args.o_fraction) / n for e in combo}
            frac["O"] = args.o_fraction
            key = protogen.canonical_key(frac)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "row_id": f"S:{key}",
                "chem_key": key,
                "material_auid": f"S:{key}",
                "recipe_auid": "",
                "source": "generated",
                "source_id": "enumerated",
                "elements": frac,
                "label": None,
                "split": "screen",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    per_size = {}
    for r in out:
        k = len([e for e in r["elements"] if e != "O"])
        per_size[k] = per_size.get(k, 0) + 1
    print(f"{len(out)} compositions -> {args.out}  ({per_size})")
    print("next: uv run python 2-build-prototypes.py --rows "
          f"{args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
