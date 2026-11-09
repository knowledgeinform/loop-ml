#!/usr/bin/env python3
"""Step 3.5: join relax results to ensembles -> per-composition EFA file.

For every ensemble dir (<chem>__<proto>/manifest.json): join relaxed
energies by task id (<ensemble dir>/<POSCAR file>), require coverage
(>= max(3, min-frac * sampled) converged configs), compute EFA = 1/sigma (AFLOW getEFA port,
loopml.mlip.efa_from_energies) and the degeneracy-weighted Boltzmann-mean
energy.  Per composition, the output row keeps ALL candidate EFAs plus the
lowest-Boltzmann-energy pick, so downstream policy changes (charge
filtering, per-family features) are pure scoring, never re-relaxation.

Outputs:
    --out       jsonl: {chem_key, efa, sigma_ev_atom, selected_proto,
                        temperature, n_prototypes, candidates}
    --coverage  json: per-(chem, proto) coverage audit + status

Usage:
    uv run python 4-score-efa.py [--ensembles data/pocc]
        [--work data/mlip_work]
        [--out data/efa.jsonl] [--coverage data/efa_coverage.json]
        [--min-frac 0.8] [--temperature 298.15] [--exclude-protos a,b]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loopml.mlip import (boltzmann_mean_energy, efa_from_energies,
                         efa_pick_lowest_energy, task_key)


def load_results_best(work_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(work_dir.glob("*.jsonl")):
        if ".in." in path.name or path.is_dir():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            k = row.get("tid")
            if k is None:  # pre-tid (hashed) ledger rows are never joined
                continue
            good = row.get("energy_atom") is not None and str(row.get("converged")) == "True"
            if k not in out or (good and (out[k].get("energy_atom") is None
                                          or str(out[k].get("converged")) != "True")):
                out[k] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ensembles", type=Path, default=ROOT / "data" / "pocc")
    ap.add_argument("--work", type=Path, default=ROOT / "data" / "mlip_work")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "efa.jsonl")
    ap.add_argument("--coverage", type=Path, default=ROOT / "data" / "efa_coverage.json")
    ap.add_argument("--min-frac", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--exclude-protos", default="",
                    help="comma list of prototypes to drop before the Boltzmann pick")
    args = ap.parse_args(argv)
    excluded = {p.strip() for p in args.exclude_protos.split(",") if p.strip()}

    results = load_results_best(args.work)
    print(f"{len(results)} relax results on file")

    by_chem: dict[str, list[dict]] = {}
    coverage = []
    for man in sorted(args.ensembles.glob("*/manifest.json")):
        m = json.loads(man.read_text())
        chem, proto = m["chem_key"], m["hypothesis"]
        if proto in excluded:
            continue
        e, g, missing, unconverged = [], [], 0, 0
        for st in m["structures"]:
            rec = results.get(f"{man.parent.name}/{st['file']}")
            if rec is None or rec.get("energy_atom") is None:
                rec = None
            if rec is None:
                missing += 1
            elif str(rec.get("converged")) != "True":
                unconverged += 1
            else:
                e.append(float(rec["energy_atom"]))
                g.append(st["degeneracy"])
        need = max(3, int(args.min_frac * len(m["structures"]) + 0.5))
        cov = {"chem_key": chem, "proto": proto, "sampled": len(m["structures"]),
               "scored": len(e), "missing": missing, "unconverged": unconverged}
        if len(e) < need:
            cov["status"] = "skipped"
            coverage.append(cov)
            continue
        try:
            efa = efa_from_energies(e, g)
        except ValueError:
            cov["status"] = "sigma<=0"
            coverage.append(cov)
            continue
        cov["status"] = "ok"
        coverage.append(cov)
        by_chem.setdefault(chem, []).append({"name": proto, "energies": e, "dgs": g})

    rows = []
    for chem, ensembles in sorted(by_chem.items()):
        pick = efa_pick_lowest_energy(ensembles, temperature=args.temperature)
        rows.append({"chem_key": chem, "efa": pick["efa"],
                     "sigma_ev_atom": 1.0 / pick["efa"],
                     "selected_proto": pick["selected"], "temperature": args.temperature,
                     "n_prototypes": len(ensembles),
                     "candidates": pick["candidates"]})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args.coverage.write_text(json.dumps(coverage, indent=2) + "\n")
    per_proto = collections.Counter(c["proto"] for c in coverage if c["status"] == "ok")
    print(f"compositions with EFA: {len(rows)} | ensembles scored: {dict(per_proto)}")
    print(f"selected_proto histogram: "
          f"{dict(collections.Counter(r['selected_proto'] for r in rows))}")
    print(f"wrote {args.out.name} and {args.coverage.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
