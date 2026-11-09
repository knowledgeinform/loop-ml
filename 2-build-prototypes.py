#!/usr/bin/env python3
"""Step 2: build POCC prototype ensembles from a rows jsonl.

For every distinct composition in the rows file, generate one ensemble per
requested prototype under <out>/<chem_key>__<prototype>/ (manifest.json +
POSCARs sampled with AFLOW POCC).  Resumable: existing manifests are
skipped ("cached"); POCC hangs and ordered-parent failures are SKIPPED AND
LOGGED to gen_summary.json -- never insist on a hung enumerator.

Sampled POCC is deterministic; no RNG seed needed.  Site partitioning
splits the cation mixture over the prototype's metal sublattices
(heaviest share first), so every family gets a fully occupied, chemically
complete cell.  This step deliberately applies NO chemistry filters --
plausibility rules belong to scoring/training so the corpus never needs
re-generating (see loopml/filters.py).

Usage:
    uv run python 2-build-prototypes.py --rows data/phase_rows.v3.jsonl
        --out data/pocc [--prototypes rocksalt,cscl,fluorite,spinel,perovskite]
        [--samples 24] [--timeout 90] [--workers 20]
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loopml import protogen

# startup lattice constants; relaxation absorbs the error
LATTICE_A = {"rocksalt": 4.2, "cscl": 3.2, "fluorite": 5.4, "spinel": 8.1,
             "perovskite": 4.0, "pyrochlore": 10.5, "bixbyite": 10.6}
DEFAULT_PROTOS = "rocksalt,cscl,fluorite,spinel,perovskite"


def compositions_from_rows(rows_path: Path) -> dict[str, dict[str, float]]:
    """canonical chem_key -> normalized mole fractions, one per composition."""
    out: dict[str, dict[str, float]] = {}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        els = r["elements"] if isinstance(r["elements"], dict) else ast.literal_eval(str(r["elements"]))
        total = sum(els.values())
        frac = {e: v / total for e, v in els.items()}
        out[protogen.canonical_key(frac)] = frac
    return out


def gen_one(job: tuple[str, str, dict[str, float], Path, int, int]) -> dict:
    key, proto, frac, out_root, samples, timeout = job
    outdir = out_root / f"{key}__{proto}"
    if (outdir / "manifest.json").is_file():
        return {"chem_key": key, "proto": proto, "status": "cached"}
    t0 = time.time()
    try:
        manifest = protogen.generate_prototypes(
            composition=frac, outdir=outdir, prototype=proto,
            lattice_a=LATTICE_A[proto], timeout=timeout, split_sites=True,
            sample_number=samples, force=True)
    except Exception as exc:
        return {"chem_key": key, "proto": proto, "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:150]}
    manifest["chem_key"] = key
    manifest["max_atoms"] = max(sum(s["counts"].values()) for s in manifest["structures"])
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"chem_key": key, "proto": proto, "status": "ok",
            "n_structures": len(manifest["structures"]), "max_atoms": manifest["max_atoms"],
            "secs": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=ROOT / "data" / "phase_rows.v3.jsonl",
                    help="rows jsonl from 1-collect-data.py or 6-generate-compositions.py")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pocc",
                    help="ensemble folder (shared corpus: labeled + screening)")
    ap.add_argument("--prototypes", default=DEFAULT_PROTOS)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=90, help="per-call POCC timeout, s")
    ap.add_argument("--workers", type=int, default=6,
                    help="POCC subprocesses; oversubscribe freely (2x cores is fine)")
    args = ap.parse_args(argv)

    protos = [p.strip() for p in args.prototypes.split(",") if p.strip()]
    unknown = [p for p in protos if p not in LATTICE_A]
    if unknown:
        raise SystemExit(f"unknown prototypes {unknown}; have {sorted(LATTICE_A)}")
    comps = compositions_from_rows(args.rows)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(key, proto, frac, args.out, args.samples, args.timeout)
            for key, frac in sorted(comps.items()) for proto in protos]
    print(f"{len(comps)} compositions x {len(protos)} prototypes -> {len(jobs)} jobs", flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(gen_one, j) for j in jobs]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            if rec["status"] == "error" or done % 50 == 0:
                print(f"  [{done}/{len(jobs)}] {rec}", flush=True)
    (args.out / "gen_summary.json").write_text(
        json.dumps(sorted(results, key=lambda r: (r["chem_key"], r["proto"])), indent=2) + "\n")
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    print(f"ok={len(ok)} errors={sum(1 for r in results if r['status'] == 'error')} "
          f"structures={sum(r.get('n_structures', 0) for r in ok)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
