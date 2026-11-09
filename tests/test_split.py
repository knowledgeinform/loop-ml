import json
import tempfile
import unittest
from pathlib import Path

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "collect_data", Path(__file__).resolve().parent.parent / "1-collect-data.py")
_cd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cd)

ORDER_PRIME = _cd.ORDER_PRIME
assign_split = _cd.assign_split
stable_bucket = _cd.stable_bucket
strata_key = _cd.strata_key


class TestStableBucket(unittest.TestCase):
    def test_range_and_determinism(self):
        first = stable_bucket("M:abc", salt="s")
        second = stable_bucket("M:abc", salt="s")
        self.assertEqual(first, second)
        self.assertIn(first, range(ORDER_PRIME))

    def test_salt_changes_assignment(self):
        keys = [f"M:{i:04d}" for i in range(200)]
        a = {stable_bucket(k, salt="v1") for k in keys}
        b = {stable_bucket(k, salt="v2") for k in keys}
        self.assertNotEqual(a, b)


class TestStrataKey(unittest.TestCase):
    def test_label_only_strata(self):
        # v2: literature and experiments pooled; source is irrelevant
        self.assertEqual(strata_key([1, 1]), "1")
        self.assertEqual(strata_key([0]), "0")

    def test_conflicting_labels_are_mixed(self):
        self.assertEqual(strata_key([0, 1]), "mixed")

    def test_generator_input_is_fully_consumed_once(self):
        # regression: strata_key must not iterate its argument twice
        gen = (label for label in [0, 0])
        self.assertEqual(strata_key(gen), "0")


class TestAssignSplit(unittest.TestCase):
    def _strata(self, counts):
        strata = {}
        for name, n in counts.items():
            for i in range(n):
                strata[f"{name}|group-{i}"] = name
        return strata

    def test_every_multi_member_stratum_appears_in_both_splits(self):
        assignments = assign_split(self._strata({"a|1": 40, "a|0": 3, "b|1": 2}))
        splits = {}
        for a in assignments:
            splits.setdefault(a.stratum, set()).add(a.split)
        for stratum, seen in splits.items():
            self.assertEqual(seen, {"train", "test"}, stratum)

    def test_singleton_stratum_stays_in_train(self):
        assignments = assign_split({"only": "solo|1"})
        self.assertEqual([(a.split,) for a in assignments], [("train",)])

    def test_groups_assigned_exactly_once(self):
        strata = self._strata({"a|1": 17, "a|0": 5})
        assignments = assign_split(strata)
        groups = [a.group for a in assignments]
        self.assertEqual(sorted(groups), sorted(strata))
        self.assertEqual(len(groups), len(set(groups)))

    def test_fold_ids_valid_for_train_only(self):
        assignments = assign_split(self._strata({"a|1": 30}), n_folds=4)
        for a in assignments:
            if a.split == "test":
                self.assertEqual(a.fold, -1)
            else:
                self.assertIn(a.fold, range(4))

    def test_test_fraction_controls_share(self):
        strata = self._strata({"a|1": 100, "a|0": 40})
        assignments = assign_split(strata, test_fraction=0.25)
        n_test = sum(1 for a in assignments if a.split == "test")
        self.assertEqual(n_test, 25 + 10)

    def test_determinism_across_calls(self):
        strata = self._strata({"a|1": 50, "b|0": 9})
        first = assign_split(strata)
        second = assign_split(dict(reversed(list(strata.items()))))
        self.assertEqual(
            sorted(first, key=lambda a: a.group),
            sorted(second, key=lambda a: a.group),
        )


class TestRecordsScan(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.records = Path(self._tmp.name)
        mat = self.records / "materials" / "M-aaa111"
        trial = mat / "recipes" / "R-111" / "trials" / "1"
        trial.mkdir(parents=True)
        (mat / "material.json").write_text(
            json.dumps({"_id": "M:aaa111", "elements": {"Co": 0.5, "O": 0.5}})
        )
        (trial / "trial.json").write_text(json.dumps({"phase_status": "single_phase"}))
        lit = mat / "recipes" / "R-111" / "literature"
        lit.mkdir()
        (lit / "L-1.json").write_text(json.dumps({"synthesis_successful": False}))
        unlabeled = self.records / "materials" / "M-bbb222"
        t2 = unlabeled / "recipes" / "R-222" / "trials" / "1"
        t2.mkdir(parents=True)
        (unlabeled / "material.json").write_text(json.dumps({"_id": "M:bbb222"}))
        (t2 / "trial.json").write_text(json.dumps({"phase_status": "not_confirmed"}))
        self.addCleanup(self._tmp.cleanup)

    def test_only_labelable_rows_are_kept(self):
        build_attempt_table = _cd.build_attempt_table

        rows = build_attempt_table(self.records)
        self.assertEqual(len(rows), 2)
        by_source = {r.source: r for r in rows}
        self.assertEqual(by_source["experiment"].label, 1)
        self.assertEqual(by_source["literature"].label, 0)
        self.assertEqual(by_source["experiment"].material_auid, "M:aaa111")
        self.assertEqual(by_source["experiment"].elements, {"Co": 0.5, "O": 0.5})


class TestEndToEndDataset(unittest.TestCase):
    def test_build_dataset_splits_disjoint_and_covers_experiments(self):
        build_dataset = _cd.build_dataset

        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp)
            mat = records / "materials" / "M-ccc333"
            trial = mat / "recipes" / "R-333" / "trials" / "1"
            trial.mkdir(parents=True)
            (mat / "material.json").write_text(json.dumps({"_id": "M:ccc333"}))
            (trial / "trial.json").write_text(
                json.dumps({"phase_status": "multi_phase"})
            )
            for i in range(10):
                lit_mat = records / "materials" / f"M-lit{i:03d}"
                lit = lit_mat / "recipes" / "R-lit" / "literature"
                lit.mkdir(parents=True)
                (lit_mat / "material.json").write_text(
                    json.dumps({"_id": f"M:lit{i:03d}"})
                )
                (lit / "L-1.json").write_text(
                    json.dumps({"synthesis_successful": i % 2 == 0})
                )

            enriched, manifest = build_dataset(records, test_fraction=0.3)

        train_mats = {r["material_auid"] for r in enriched if r["split"] == "train"}
        test_mats = {r["material_auid"] for r in enriched if r["split"] == "test"}
        self.assertFalse(train_mats & test_mats)

        experiment_rows = [r for r in enriched if r["source"] == "experiment"]
        self.assertEqual(len(experiment_rows), 1)
        # pooled label strata: both classes must appear on both split sides
        for key in ("train|0", "train|1", "test|0", "test|1"):
            self.assertIn(key, manifest["rows_per_split_x_stratum"])


if __name__ == "__main__":
    unittest.main()
