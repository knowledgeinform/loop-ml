#!/usr/bin/env python3
"""

from __future__ import annotations
Step 4 (training): train the classifier from rows (+ optional EFA file)
and A/B it.  Self-contained: composition feature table, model zoo,
grouped out-of-fold CV, metrics, held-out evaluation.
"""


# ==================== composition feature table (was loopml/features.py) ====================


import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_VERSION = "phase-feat-v1"

ASSET_PATH = Path(__file__).resolve().parent / "loopml" / "assets" / "element_properties.v1.json"

ELEMENT_FIELDS = (
    "atomic_number",
    "atomic_mass",
    "electronegativity_Ghosh",
    "hardness_chemical_Ghosh",
    "chemical_scale_Pettifor",
    "radii_Ghosh08",
    "radii_Pyykko",
    "radius_covalent",
    "polarizability",
    "density",
    "temperature_melting",
    "enthalpy_vaporization",
    "valence_iupac",
    "valence_s",
    "valence_p",
    "valence_d",
    "period",
)

_R = 8.314462618  # J/mol/K

_number_cache: dict[str, dict[str, float]] | None = None


def _element_table() -> dict[str, dict[str, float]]:
    global _number_cache
    if _number_cache is None:
        raw = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
        table = {}
        for symbol, payload in raw.items():
            if symbol == "AAA_notes" or not isinstance(payload, dict):
                continue
            row: dict[str, float] = {}
            for field in ELEMENT_FIELDS:
                try:
                    value = float(payload.get(field))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    row[field] = value
            table[symbol] = row
        _number_cache = table
    return _number_cache


def _pairwise_mean(
    values: np.ndarray, weights: np.ndarray, property_values: np.ndarray
) -> float:
    # sum_i sum_j w_i w_j |p_i - p_j| (Hume-Rothery style mismatch)
    diffs = np.abs(property_values[:, None] - property_values[None, :])
    return float(np.sum(weights[:, None] * weights[None, :] * diffs))


def composition_features(elements: dict[str, float]) -> dict[str, float]:
    table = _element_table()
    amounts = {
        symbol: float(value)
        for symbol, value in elements.items()
        if float(value) > 0
    }
    if not amounts:
        raise ValueError("empty composition")
    total = sum(amounts.values())
    symbols = sorted(amounts)
    x = np.array([amounts[s] / total for s in symbols], dtype=float)

    features: dict[str, float] = {}
    features["comp_n_elements"] = float(len(symbols))
    features["comp_n_cations"] = float(sum(s != "O" for s in symbols))
    cation_total = sum(v for s, v in amounts.items() if s != "O")
    features["comp_oxygen_to_cation"] = (
        amounts.get("O", 0.0) / cation_total if cation_total else 0.0
    )
    features["comp_entropy_mixing"] = -float(np.sum(x * np.log(x))) * _R
    features["comp_is_equimolar"] = float(np.allclose(x, x[0]))

    for field in ELEMENT_FIELDS:
        vals = np.array([table.get(s, {}).get(field, np.nan) for s in symbols])
        mask = ~np.isnan(vals)
        prefix = f"comp_{field}"
        if not np.any(mask):
            features[f"{prefix}_wmean"] = np.nan
            features[f"{prefix}_min"] = np.nan
            features[f"{prefix}_max"] = np.nan
            features[f"{prefix}_range"] = np.nan
            features[f"{prefix}_mismatch"] = np.nan
            continue
        w = x[mask]
        w = w / w.sum()
        v = vals[mask]
        wmean = float(np.sum(w * v))
        features[f"{prefix}_wmean"] = wmean
        features[f"{prefix}_min"] = float(v.min())
        features[f"{prefix}_max"] = float(v.max())
        features[f"{prefix}_range"] = float(v.max() - v.min())
        features[f"{prefix}_mismatch"] = _pairwise_mean(x, w, v)
    return features


TEMP_KEYS = ("temperature_c", "max_temp_c", "temp_c", "temperature")
ATMOSPHERE_TOKENS = (
    "air",
    "oxygen",
    "argon",
    "nitrogen",
    "vacuum",
    "reducing",
    "forming_gas",
)


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _step_temperature(step: dict) -> float | None:
    for key in TEMP_KEYS:
        value = _to_float(step.get(key))
        if value is not None:
            return value
    return None


def _step_hold_hours(step: dict) -> float | None:
    hours = _to_float(step.get("hold_time_hours"))
    if hours is not None:
        return hours
    minutes = _to_float(step.get("hold_time_min"))
    if minutes is not None:
        return minutes / 60.0
    seconds = _to_float(step.get("hold_time_s"))
    if seconds is not None:
        return seconds / 3600.0
    return None


def process_features(steps: list[dict]) -> dict[str, float]:
    features: dict[str, float] = {
        "proc_n_steps": float(len(steps)),
        "proc_temp_max": np.nan,
        "proc_temp_mean": np.nan,
        "proc_hold_total_hr": np.nan,
        "proc_has_milling": 0.0,
        "proc_has_pressing": 0.0,
        "proc_has_combustion_or_calcination": 0.0,
        "proc_precursor_count": 0.0,
        "proc_uses_nitrate": 0.0,
        "proc_uses_carbonate": 0.0,
        "proc_uses_hydroxide": 0.0,
    }
    for token in ATMOSPHERE_TOKENS:
        features[f"proc_atm_{token}"] = 0.0
    if not steps:
        features["proc_has_steps"] = 0.0
        return features
    features["proc_has_steps"] = 1.0

    temps: list[float] = []
    holds: list[float] = []
    formulas: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        temp = _step_temperature(step)
        if temp is not None:
            temps.append(temp)
        hold = _step_hold_hours(step)
        if hold is not None:
            holds.append(hold)

        kind = " ".join(
            str(step.get(key) or "")
            for key in ("step_type", "method", "atmosphere")
        ).lower()
        if "mill" in kind or "grind" in kind:
            features["proc_has_milling"] = 1.0
        if "press" in kind or "pellet" in kind:
            features["proc_has_pressing"] = 1.0
        if "calcine" in kind or "combust" in kind:
            features["proc_has_combustion_or_calcination"] = 1.0

        atmosphere = str(step.get("atmosphere") or "").lower()
        for token in ATMOSPHERE_TOKENS:
            if token.replace("_", " ") in atmosphere:
                features[f"proc_atm_{token}"] = 1.0

        for precursor in step.get("precursors_list") or []:
            if isinstance(precursor, dict) and precursor.get("formula"):
                formulas.append(str(precursor["formula"]))
        if isinstance(step.get("precursors"), str):
            formulas.extend(step["precursors"].split(","))

    if temps:
        features["proc_temp_max"] = max(temps)
        features["proc_temp_mean"] = sum(temps) / len(temps)
    if holds:
        features["proc_hold_total_hr"] = sum(holds)

    blob = " ".join(formulas)
    precursor_set = {f.strip() for f in formulas if f.strip()}
    features["proc_precursor_count"] = float(len(precursor_set))
    features["proc_uses_nitrate"] = float("NO3" in blob)
    features["proc_uses_carbonate"] = float("CO3" in blob)
    features["proc_uses_hydroxide"] = float("OH" in blob and "O" not in blob[:2])
    return features


STRUCTURE_FAMILIES = (
    "rocksalt",
    "spinel",
    "perovskite",
    "fluorite",
    "pyrochlore",
    "other",
)


def structure_features(family: str | None) -> dict[str, float]:
    normalized = str(family or "").strip().lower()
    features = {
        f"struct_family_{name}": float(normalized == name)
        for name in STRUCTURE_FAMILIES
    }
    features["struct_family_known"] = float(bool(normalized and normalized != "unknown"))
    return features


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _recipe_dir(records_dir: Path, material_auid: str, recipe_auid: str) -> Path | None:
    if not recipe_auid or ":R:" not in recipe_auid:
        return None
    mat = material_auid.replace(":", "-")
    rec = recipe_auid.split(":R:", 1)[1].replace(":", "-")
    path = records_dir / "materials" / mat / "recipes" / rec
    return path if path.is_dir() else None


def features_for_row(records_dir: str | Path, row: dict) -> dict[str, float]:
    """Assemble all at-screening-time features for one labeled attempt row."""
    records_dir = Path(records_dir)
    features = dict(composition_features(row["elements"]))

    recipe_path = None
    recipe = None
    directory = _recipe_dir(records_dir, row["material_auid"], row.get("recipe_auid", ""))
    if directory is not None:
        recipe_path = directory / "recipe.json"
        recipe = _load_json(recipe_path)

    steps: list = []
    if recipe and isinstance(recipe.get("synthesis_steps"), list):
        steps = recipe["synthesis_steps"]
    if not steps:
        if row["source"] == "literature":
            source_id = row["source_id"].split(":")[-1]
            mat = row["material_auid"].replace(":", "-")
            lit = None
            if directory is not None:
                lit = _load_json(directory / "literature" / f"{source_id}.json")
            if lit is None:
                matches = sorted(
                    (records_dir / "materials" / mat / "recipes").glob(
                        f"*/literature/{source_id}.json"
                    )
                )
                lit = _load_json(matches[0]) if matches else None
            steps = (
                ((lit or {}).get("exp_condition") or {})
                .get("additional_params", {})
                .get("synthesis_steps", [])
            )
        else:
            steps = []
    features.update(process_features(steps if isinstance(steps, list) else []))

    family = None
    if recipe and recipe.get("structure_family"):
        family = recipe["structure_family"]
    else:
        material = _load_json(
            records_dir / "materials" / row["material_auid"].replace(":", "-") / "material.json"
        )
        if material:
            family = material.get("structure_family")
    features.update(structure_features(family))
    return features


def build_feature_table(
    records_dir: str | Path, rows: list[dict]
) -> dict[str, dict[str, float]]:
    return {row["row_id"]: features_for_row(records_dir, row) for row in rows}


# ==================== models, CV, metrics (was loopml/run_phase_baselines.py) ====================


import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from loopml.config import BASELINE_RESULTS_PATH, DATA_DIR, PHASE_FEATURES_PATH, PHASE_ROWS_PATH, RECORDS_DIR

ROWS_PATH = PHASE_ROWS_PATH
FEATURES_PATH = PHASE_FEATURES_PATH

BLOCKS = {
    "comp": ("comp_",),
    "proc": ("proc_",),
    "struct": ("struct_",),
}


def load_dataset(refresh: bool):
    rows = [json.loads(line) for line in ROWS_PATH.read_text().splitlines()]
    if refresh or not FEATURES_PATH.exists():
        table = build_feature_table(RECORDS_DIR, rows)
        with FEATURES_PATH.open("w") as handle:
            for row in rows:
                record = dict(row)
                record["features"] = table[row["row_id"]]
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    else:
        table = {}
        rows = []
        for line in FEATURES_PATH.read_text().splitlines():
            record = json.loads(line)
            rows.append(record)
            table[record["row_id"]] = record.pop("features")
    return rows, table


def make_matrix(rows, table, blocks):
    prefixes = tuple(p for block in blocks for p in BLOCKS[block])
    names = sorted(
        {
            name
            for row in rows
            for name in table[row["row_id"]]
            if name.startswith(prefixes)
        }
    )
    X = np.array(
        [[table[row["row_id"]].get(name, np.nan) for name in names] for row in rows],
        dtype=float,
    )
    y = np.array([row["label"] for row in rows], dtype=int)
    return X, y, names


def make_model(kind: str):
    if kind == "dummy":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="prior")
    if kind.startswith("logreg"):
        C = float(kind.split("-")[1])
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=C, max_iter=5000)),
            ]
        )
    if kind.startswith("rf"):
        leaf = int(kind.split("-")[1])
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=leaf,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=21,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind.startswith("et"):
        leaf = int(kind.split("-")[1])
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=leaf,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=21,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind.startswith("hgb"):
        parts = kind.split("-")
        lr = float(parts[1])
        depth = None if parts[2] == "None" else int(parts[2])
        return HistGradientBoostingClassifier(
            learning_rate=float(lr),
            max_depth=depth,
            max_iter=300,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=21,
        )
    raise ValueError(kind)


def metrics(y, p) -> dict[str, float]:
    out = {
        "auc": float("nan"),
        "ap": float("nan"),
        "brier": brier_score_loss(y, p),
        "bal_acc": float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, p)
        out["ap"] = average_precision_score(y, p)
        out["bal_acc"] = balanced_accuracy_score(y, (np.asarray(p) >= 0.5).astype(int))
    return out


def oof_probabilities(kind, rows, X, y):
    proba = np.full(len(rows), np.nan)
    for fold in range(4):
        train_idx = np.where(
            np.array(
                [
                    row["split"] == "train" and row.get("cv_fold") != fold
                    for row in rows
                ]
            )
        )[0]
        val_idx = np.where(
            np.array(
                [
                    row["split"] == "train" and row.get("cv_fold") == fold
                    for row in rows
                ]
            )
        )[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model = make_model(kind)
        model.fit(X[train_idx], y[train_idx])
        proba[val_idx] = predict_positive(model, X[val_idx])
    return proba


def predict_positive(model, X):
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        p = model.predict_proba(X)
        return p[:, classes.index(1)] if 1 in classes else p[:, 0]
    return np.asarray(model.predict(X), dtype=float)



# ==================== script ====================

import argparse
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from loopml import filters, protogen

BLOCKS["efa"] = ("efa_",)
KINDS = ("logreg-0.01", "logreg-1", "rf-3", "et-1", "hgb-0.06-None")


def attach_efa(rows, table, efa_path: Path) -> list[dict]:
    """Keep rows whose composition has a charge-feasible EFA; add features."""
    tourney = {}
    for line in efa_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tourney[r["chem_key"]] = r
    keep, n_infeasible, n_missing = [], 0, 0
    for row in rows:
        els = row["elements"] if isinstance(row["elements"], dict) else ast.literal_eval(str(row["elements"]))
        total = sum(els.values())
        frac = {e: v / total for e, v in els.items()}
        key = protogen.canonical_key(frac)
        if key not in tourney:
            n_missing += 1
            continue
        cats = {e: v for e, v in frac.items() if e != "O"}
        rec = tourney[key]
        feas = set(filters.feasible_prototypes(
            cats, [c["name"] for c in rec["candidates"]]))
        cands = [c for c in rec["candidates"] if c["name"] in feas]
        if not cands:
            n_infeasible += 1
            continue
        best = min(cands, key=lambda c: c["mean_energy"])
        v = best["efa"]
        feats = table[row["row_id"]]
        feats["efa_value"] = v
        feats["efa_log"] = math.log(v)
        feats["efa_sigma"] = 1.0 / v
        feats["efa_n_feasible"] = float(len(feas))
        keep.append(row)
    print(f"rows: {len(rows)} total -> {len(keep)} with charge-feasible EFA "
          f"({n_missing} without computed EFA, {n_infeasible} with no feasible prototype)")
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=None,
                    help="rows jsonl from step 1 (default: dataset config path)")
    ap.add_argument("--efa", type=Path, default=None,
                    help="EFA jsonl from 4-score-efa.py (default: data/efa.jsonl if "
                         "it exists; pass an explicit path or --no-efa to override)")
    ap.add_argument("--no-efa", action="store_true",
                    help="force a composition-only run")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "classifier_results.json")
    args = ap.parse_args(argv)

    if args.rows is not None:
        global ROWS_PATH, FEATURES_PATH
        ROWS_PATH = args.rows
        FEATURES_PATH = args.rows.with_suffix(".features.json")
        rows, table = load_dataset(refresh=not FEATURES_PATH.exists())
    else:
        rows, table = load_dataset(refresh=False)

    if args.efa is None and not args.no_efa:
        candidate = ROOT / "data" / "efa.jsonl"
        args.efa = candidate if candidate.is_file() else None
    with_efa = args.efa is not None and not args.no_efa
    if with_efa:
        rows = attach_efa(rows, table, args.efa)
    train_mask = np.array([r["split"] == "train" for r in rows])
    test_mask = np.array([r["split"] == "test" for r in rows])
    arms = (("A comp", ("comp",)), ("B comp+efa", ("comp", "efa"))) if with_efa \
        else (("A comp", ("comp",)),)

    print(f"\nCV on train (pooled OOF, grouped folds){'; A = comp, B = comp+EFA' if with_efa else ''}\n")
    print(f"{'model':14s} {'arm':22s} {'AUC':>6s} {'AP':>6s} {'Brier':>6s} {'balAcc':>6s}")
    cv = {}
    best = None
    for kind in KINDS:
        for arm, blocks in arms:
            X, y, _ = make_matrix(rows, table, blocks)
            proba = oof_probabilities(kind, rows, X, y)
            ok = ~np.isnan(proba)
            m = metrics(y[ok], proba[ok])
            cv[f"{kind}|{arm}"] = m
            print(f"{kind:14s} {arm:22s} {m['auc']:6.3f} {m['ap']:6.3f} "
                  f"{m['brier']:6.3f} {m['bal_acc']:6.3f}")
            if best is None or m["auc"] > best[0]:
                best = (m["auc"], kind)

    kind = best[1]
    matrices = {}
    for arm, blocks in arms:
        matrices[arm] = make_matrix(rows, table, blocks)[0]
    y = make_matrix(rows, table, ("comp",))[1]
    delta = None
    if with_efa:
        pa = oof_probabilities(kind, rows, matrices["A comp"], y)
        pb = oof_probabilities(kind, rows, matrices["B comp+efa"], y)
        ok = ~np.isnan(pa)
        delta = metrics(y[ok], pb[ok])["auc"] - metrics(y[ok], pa[ok])["auc"]

    print(f"\nHELD-OUT TEST (model: {kind}, {int(test_mask.sum())} rows, "
          f"labels {np.bincount(y[test_mask], minlength=2).tolist()} neg/pos)")
    test = {}
    for arm, X in matrices.items():
        model = make_model(kind)
        model.fit(X[train_mask], y[train_mask])
        p = predict_positive(model, X[test_mask])
        m = metrics(y[test_mask], p)
        test[arm] = m
        print(f"  {arm:12s} AUC {m['auc']:.3f}  AP {m['ap']:.3f}  "
              f"Brier {m['brier']:.3f}  balAcc {m['bal_acc']:.3f}")

    out = {"n_rows": len(rows), "model": kind, "efa": str(args.efa) if args.efa else None,
           "cv_auc_delta_B_minus_A": round(delta, 4) if delta is not None else None,
           "cv_table": cv, "test": test}
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())