#!/usr/bin/env python3
"""

from __future__ import annotations
Step 1: build the labeled dataset (rows jsonl + deterministic splits)
from the synthesis-records archive (materials/<M-xxx>/recipes/<R-xxx>/
trials|literature).  Self-contained: record parsing, prime-remainder
grouped splits, dataset assembly.

Outputs (default --out-dir data/):
    phase_rows.v3.jsonl    one row per labeled synthesis attempt
                           (elements, label, split, cv_fold, provenance)
    phase_split.v3.json    split manifest

Usage:
    uv run python 1-collect-data.py [--records PATH] [--out-dir data]
        [--test-fraction 0.2] [--dry-run]
"""


# ==================== records archive parsing (was loopml/data.py) ====================


import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal

LabelSource = Literal["experiment", "literature"]

PHASE_STATUS_LABELS = {"single_phase": 1, "multi_phase": 0}


@dataclass(frozen=True)
class AttemptRow:
    material_auid: str
    elements: dict[str, float]
    source: LabelSource
    source_id: str
    recipe_auid: str
    label: int
    year: int | None = None
    notes: str | None = None

    def row_id(self) -> str:
        return f"{self.source}:{self.material_auid}:{self.source_id}"

    def to_dict(self) -> dict:
        return asdict(self)


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _material_auid(material_dir: Path, material: dict) -> str:
    auid = material.get("_id") or material.get("material_auid")
    if auid:
        return str(auid)
    return material_dir.name.replace("M-", "M:", 1)


def iter_material_dirs(records_dir: Path) -> Iterator[Path]:
    materials_root = records_dir / "materials"
    if not materials_root.is_dir():
        raise FileNotFoundError(f"no materials/ under {records_dir}")
    yield from sorted(p for p in materials_root.iterdir() if p.is_dir())


def attempts_for_material(material_dir: Path) -> list[AttemptRow]:
    recipes_dir = material_dir / "recipes"
    if not recipes_dir.is_dir():
        return []

    candidate_files = sorted(
        path
        for pattern in ("*/trials/*/trial.json", "*/literature/L-*.json")
        for path in recipes_dir.glob(pattern)
    )
    if not candidate_files:
        return []

    material = _load_json(material_dir / "material.json")
    if material is None:
        return []

    material_auid = _material_auid(material_dir, material)
    raw_elements = material.get("elements") or {}
    elements = {
        str(symbol): float(fraction)
        for symbol, fraction in raw_elements.items()
    }

    rows: list[AttemptRow] = []

    for recipe_dir in sorted(p for p in recipes_dir.iterdir() if p.is_dir()):
        recipe_auid = str(
            recipe_dir.name.replace("R-", "M::R:", 1)
        )

        trials_dir = recipe_dir / "trials"
        if trials_dir.is_dir():
            for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
                trial = _load_json(trial_dir / "trial.json")
                if trial is None:
                    continue
                status = str(trial.get("phase_status") or "").strip().lower()
                if status not in PHASE_STATUS_LABELS:
                    continue
                recipe_of_trial = str(trial.get("recipe_auid") or recipe_auid)
                rows.append(
                    AttemptRow(
                        material_auid=material_auid,
                        elements=elements,
                        source="experiment",
                        source_id=f"{recipe_of_trial}:T{trial_dir.name}",
                        recipe_auid=recipe_of_trial,
                        label=PHASE_STATUS_LABELS[status],
                    )
                )

        literature_dir = recipe_dir / "literature"
        if literature_dir.is_dir():
            for lit_path in sorted(literature_dir.glob("L-*.json")):
                lit = _load_json(lit_path)
                if lit is None:
                    continue
                successful = lit.get("synthesis_successful")
                if successful is not True and successful is not False:
                    continue
                year = lit.get("year")
                rows.append(
                    AttemptRow(
                        material_auid=material_auid,
                        elements=elements,
                        source="literature",
                        source_id=lit_path.stem,
                        recipe_auid=str(lit.get("recipe_auid") or recipe_auid),
                        label=int(successful),
                        year=int(year) if isinstance(year, int) else None,
                        notes=str(lit.get("notes")) if lit.get("notes") else None,
                    )
                )

    return rows


def build_attempt_table(records_dir: str | Path) -> list[AttemptRow]:
    """Scan the whole archive and return every labeled synthesis attempt."""
    records_dir = Path(records_dir)
    rows: list[AttemptRow] = []
    seen: set[str] = set()
    for material_dir in iter_material_dirs(records_dir):
        for row in attempts_for_material(material_dir):
            if row.row_id() in seen:
                continue
            seen.add(row.row_id())
            rows.append(row)
    rows.sort(key=lambda r: (r.material_auid, r.source, r.source_id))
    return rows


# ==================== deterministic grouped splits (was loopml/split.py) ====================


import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SPLIT_SALT = "phase-split-v3"
ORDER_PRIME = 1_000_003  # prime; orders groups pseudo-randomly but stably


def stable_bucket(key: str, *, salt: str = SPLIT_SALT, modulus: int = ORDER_PRIME) -> int:
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulus


def strata_key(row_labels: Iterable[int], has_dft: bool = False) -> str:
    """Stratify by label (v2+) x DFT-twin coverage (v3+).

    ``row_labels`` is an iterable of labels for all rows of one group.
    A group whose labels conflict gets a ``mixed`` stratum.
    """
    labels = set(row_labels)
    if len(labels) > 1:
        return "mixed"
    return f"{labels.pop()}|dft" if has_dft else str(labels.pop())


def rank_groups(groups: Sequence[str], *, salt: str = SPLIT_SALT) -> list[str]:
    return sorted(groups, key=lambda g: (stable_bucket(g, salt=salt), g))


@dataclass(frozen=True)
class SplitAssignment:
    group: str
    stratum: str
    bucket: int
    split: str  # "train" | "test"
    fold: int  # 0..n_folds-1 cross-validation fold within the train side


def assign_split(
    group_strata: Mapping[str, str],
    *,
    test_fraction: float = 0.2,
    n_folds: int = 4,
    salt: str = SPLIT_SALT,
) -> list[SplitAssignment]:
    """Assign every group to train/test and a CV fold, deterministically.

    Each stratum keeps at least one group in train and one in test whenever it
    has two or more groups. Single-member strata stay in train and are
    reported so they can be backfilled.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    by_stratum: dict[str, list[str]] = {}
    for group, stratum in group_strata.items():
        by_stratum.setdefault(stratum, []).append(group)

    assignments: list[SplitAssignment] = []
    for stratum in sorted(by_stratum):
        ranked = rank_groups(by_stratum[stratum], salt=salt)
        n = len(ranked)
        n_test = max(1, round(test_fraction * n)) if n >= 2 else 0
        n_test = min(n_test, n - 1)
        test_members = set(ranked[:n_test])
        train_members = ranked[n_test:]

        for fold, group in enumerate(train_members):
            assignments.append(
                SplitAssignment(
                    group=group,
                    stratum=stratum,
                    bucket=stable_bucket(group, salt=salt),
                    split="train",
                    fold=fold % n_folds,
                )
            )
        for group in sorted(test_members):
            assignments.append(
                SplitAssignment(
                    group=group,
                    stratum=stratum,
                    bucket=stable_bucket(group, salt=salt),
                    split="test",
                    fold=-1,
                )
            )

    return assignments


def split_summary(
    assignments: Iterable[SplitAssignment],
    row_strata: Mapping[str, str],
) -> dict[str, int]:
    """Counts of rows per split x stratum for the manifest/report."""
    split_of = {a.group: a.split for a in assignments}
    counter: Counter[str] = Counter()
    for group, stratum in row_strata.items():
        split = split_of.get(group)
        if split is not None:
            counter[f"{split}|{stratum}"] += 1
    return dict(sorted(counter.items()))


# ==================== dataset assembly (was loopml/build_phase_dataset.py) ====================


import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from loopml import config

DATASET_VERSION = "phase-attempts-v3"
ROWS_FILENAME = "phase_rows.v3.jsonl"
MANIFEST_FILENAME = "phase_split.v3.json"


def _dft_covered_groups(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text()))


def _group_rows(rows) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row.material_auid].append(row)
    return groups


def build_dataset(
    records_dir: str | Path,
    *,
    test_fraction: float = 0.3,
    n_folds: int = 4,
    dft_match_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    rows = build_attempt_table(records_dir)
    if not rows:
        raise SystemExit(f"no labeled attempts found under {records_dir}")

    covered = _dft_covered_groups(
        Path(dft_match_path) if dft_match_path else config.LABELED_DFT_MATCH_PATH
    )
    groups = _group_rows(rows)
    group_strata = {
        group: strata_key(
            (row.label for row in group_rows), has_dft=group in covered
        )
        for group, group_rows in groups.items()
    }
    assignments = assign_split(
        group_strata, test_fraction=test_fraction, n_folds=n_folds
    )
    by_group = {a.group: a for a in assignments}

    enriched: list[dict] = []
    for row in rows:
        assignment = by_group[row.material_auid]
        record = row.to_dict()
        record.update(
            row_id=row.row_id(),
            dataset_version=DATASET_VERSION,
            stratum=assignment.stratum,
            split=assignment.split,
            cv_fold=assignment.fold if assignment.split == "train" else None,
            label_conflict=assignment.stratum == "mixed",
        )
        enriched.append(record)

    row_strata = {
        row.material_auid: by_group[row.material_auid].stratum for row in rows
    }
    label_pairs = Counter(
        f"{r['split']}|{r['source']}|{r['label']}" for r in enriched
    )
    conflicts = sorted(
        {
            r["material_auid"]
            for r in enriched
            if r["label_conflict"]
        }
    )
    singleton_strata = sorted(
        a.stratum
        for a in assignments
        if _stratum_size(assignments, a.stratum) == 1
    )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "records_dir": str(Path(records_dir).resolve()),
        "split_salt": SPLIT_SALT,
        "order_prime": ORDER_PRIME,
        "test_fraction": test_fraction,
        "n_folds": n_folds,
        "n_rows": len(enriched),
        "n_materials": len(groups),
        "rows_per_split_x_source_x_label": dict(sorted(label_pairs.items())),
        "materials_per_stratum": dict(
            sorted(Counter(group_strata.values()).items())
        ),
        "rows_per_split_x_stratum": split_summary(assignments, row_strata),
        "materials_with_label_conflict": conflicts,
        "singleton_strata_train_only": singleton_strata,
    }
    return enriched, manifest


def _stratum_size(assignments: list[SplitAssignment], stratum: str) -> int:
    return sum(1 for a in assignments if a.stratum == stratum)


def _print_summary(manifest: dict) -> None:
    print(f"rows: {manifest['n_rows']}  materials: {manifest['n_materials']}")
    print("\nrows per split x source x label")
    for key, count in manifest["rows_per_split_x_source_x_label"].items():
        print(f"  {key:35s} {count}")
    print("\nmaterials per stratum")
    for key, count in manifest["materials_per_stratum"].items():
        print(f"  {key:35s} {count}")
    if manifest["materials_with_label_conflict"]:
        print(
            "\nlabel conflicts (kept, stratum=mixed): "
            + ", ".join(manifest["materials_with_label_conflict"])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(config.RECORDS_DIR), type=Path)
    parser.add_argument("--out-dir", default=str(config.DATA_DIR), type=Path)
    parser.add_argument("--test-fraction", default=0.3, type=float)
    parser.add_argument("--n-folds", default=4, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    enriched, manifest = build_dataset(
        args.records,
        test_fraction=args.test_fraction,
        n_folds=args.n_folds,
    )

    train_mats = {r["material_auid"] for r in enriched if r["split"] == "train"}
    test_mats = {r["material_auid"] for r in enriched if r["split"] == "test"}
    assert not (train_mats & test_mats), "material leaked across splits"

    _print_summary(manifest)

    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / ROWS_FILENAME
    with rows_path.open("w", encoding="utf-8") as handle:
        for record in enriched:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest_path = args.out_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {rows_path} and {manifest_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
