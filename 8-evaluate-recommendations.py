#!/usr/bin/env python3
"""Step 8: evaluate the classifier on the screening space -> recommendations.

Trains the final classifier on ALL labeled rows (the model-selection
protocol is the one validated in step 5: grouped OOF CV picks the model
family from the train-side CV table), then scores every enumerated
composition (step 6) that has a computed EFA (steps 2-4) and writes a
ranked recommendation list.  Screening rows missing an EFA are reported
and skipped: run the corpus steps first.

Note: unlike the A/B in step 5 (which held out a test split), this final
model trains on every labeled material -- the point is prediction on
compositions that were never in the corpus, and the quality gate is
step 5's held-out metrics, not this script's.

Usage:
    uv run python 8-evaluate-recommendations.py [--rows data/phase_rows.v3.jsonl]
        [--compositions data/compositions.jsonl] [--efa data/efa.jsonl]
        [--out data/recommendations.jsonl] [--top 20]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("trainer", ROOT / "5-train-classifier.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

import numpy as np

from loopml import config, protogen


def formula_of(elements: dict[str, float]) -> str:
    metals = sorted((e for e in elements if e != "O"),
                    key=lambda e: -elements[e])
    body = "".join(f"{e}{elements[e]:.3g}" for e in metals)
    return f"({body})O{elements.get('O', 0):.3g}" if metals else "O"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=ROOT / "data" / "phase_rows.v3.jsonl",
                    help="labeled rows used for training")
    ap.add_argument("--compositions", type=Path, default=ROOT / "data" / "compositions.jsonl",
                    help="enumerated screening rows from step 6")
    ap.add_argument("--efa", type=Path, default=ROOT / "data" / "efa.jsonl",
                    help="EFA file from step 4 (covers labeled AND screening compositions)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "recommendations.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "recommendations_report.json")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    # --- training set: labeled rows with a charge-feasible EFA (step-5 protocol)
    rows, table = tr.load_dataset(refresh=False)
    rows = tr.attach_efa(rows, table, args.efa)
    if len(rows) < 30:
        raise SystemExit(f"only {len(rows)} labeled rows with EFA; run steps 1-4 first")
    best = None
    for kind in tr.KINDS:
        X, y, _ = tr.make_matrix(rows, table, ("comp", "efa"))
        proba = tr.oof_probabilities(kind, rows, X, y)
        ok = ~np.isnan(proba)
        auc = tr.metrics(y[ok], proba[ok])["auc"]
        print(f"CV AUC {kind:14s} {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, kind)
    _, kind = best
    X, y, names = tr.make_matrix(rows, table, ("comp", "efa"))
    model = tr.make_model(kind)
    model.fit(X, y)
    print(f"final model: {kind} trained on all {len(rows)} labeled rows with EFA "
          f"(model choice from CV, gate on step-5 held-out metrics)")

    # --- screening rows: features + EFA, then predict
    screen = []
    for line in args.compositions.read_text().splitlines():
        if line.strip():
            screen.append(json.loads(line))
    for row in screen:
        table[row["row_id"]] = tr.features_for_row(config.RECORDS_DIR, row)
    scored = tr.attach_efa(screen, table, args.efa)
    missing = len(screen) - len(scored)
    if not scored:
        raise SystemExit("no screening composition has a computed EFA; run steps 2-4 "
                         f"with --rows {args.compositions} first")
    for row in scored:
        row["label"] = 0  # dummy: prediction only, y is never used below
    Xs, _, _ = tr.make_matrix(scored, table, ("comp", "efa"))
    p = tr.predict_positive(model, Xs)

    out_rows = []
    for row, prob in sorted(zip(scored, p), key=lambda t: -t[1]):
        els = row["elements"] if isinstance(row["elements"], dict) else ast.literal_eval(str(row["elements"]))
        out_rows.append({"chem_key": protogen.canonical_key(els),
                         "formula": formula_of(els),
                         "elements": els,
                         "p_single_phase": round(float(prob), 4)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows))
    report = {"model": kind, "cv_auc": round(best[0], 4),
              "n_labeled_train_rows": len(rows),
              "n_screening_compositions": len(screen),
              "n_scored": len(scored), "n_missing_efa_skipped": missing,
              "efa": str(args.efa), "rows": str(args.rows)}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\nranked {len(out_rows)} recommendations -> {args.out.name} "
          f"({missing} skipped: no EFA)")
    print(f"{'rank':>4}  {'p(single)':>9}  formula")
    for i, r in enumerate(out_rows[:args.top], 1):
        print(f"{i:>4}  {r['p_single_phase']:>9.3f}  {r['formula']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
