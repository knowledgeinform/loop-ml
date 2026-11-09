import math
import unittest
from pathlib import Path

from loopml.mlip import (
    MODEL_PATH,
    atoms_from_dict,
    pearson,
    read_results,
    spearman,
    task_key,
    weighted_stdev,
    efa_from_energies,
)

LIB_STRUCTURE = {
    "Lattice Matrix": [[0.0, 2.1, 2.1], [2.1, 0.0, 2.1], [2.1, 2.1, 0.0]],
    "Elements": ["Ti", "O", "O", "Zr"],
    "Atomic Positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0]],
    "Coordinate Type": "Direct(4) [A1B2C1]",
}

# first lib5 composition (Cd-Nb-O-Sn-Zn) as published by ChemScreen
EFORMS = [-1.32658, -1.35735, -1.38405, -1.21188, -1.38363, -1.38337, -1.44882, -1.02768, -1.57123]
DGS = [8] * len(EFORMS)


class WeightedStdevTests(unittest.TestCase):
    def test_matches_hand_formula(self):
        values, weights = [1.0, 2.0, 4.0], [1, 1, 2]
        mean = (1 + 2 + 8) / 4
        var = (1 * (1 - mean) ** 2 + 1 * (2 - mean) ** 2 + 2 * (4 - mean) ** 2) / 3
        self.assertAlmostEqual(weighted_stdev(values, weights), math.sqrt(var))

    def test_needs_two_values(self):
        with self.assertRaises(ValueError):
            weighted_stdev([1.0], [5])

    def test_efa_is_inverse_sigma(self):
        self.assertAlmostEqual(efa_from_energies(EFORMS, DGS), 1 / weighted_stdev(EFORMS, DGS))


class CorrelationTests(unittest.TestCase):
    def test_perfect_and_rank(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 35]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 1, 2]), -0.5)


class AtomsTests(unittest.TestCase):
    def test_from_lib_structure(self):
        atoms = atoms_from_dict(LIB_STRUCTURE)
        self.assertEqual(atoms.get_chemical_symbols(), ["Ti", "O", "O", "Zr"])
        self.assertAlmostEqual(atoms.get_cell().volume, 2 * 2.1**3, places=5)

    def test_task_key_stable_and_path_derived(self):
        k1 = task_key(Path("/a/b/AB_cF8_225__rocksalt/POSCAR_H0C1"))
        k2 = task_key(Path("/elsewhere/AB_cF8_225__rocksalt/POSCAR_H0C1"))
        self.assertEqual(k1, k2)  # machine-independent: dir + file only
        self.assertEqual(k1, "AB_cF8_225__rocksalt/POSCAR_H0C1")


class LedgerTests(unittest.TestCase):
    """relax results join back to structures by tid, newest-converged wins."""

    def test_read_results_joins_by_tid(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "bshard0.jsonl").write_text(json.dumps(
                {"tid": "chem__proto/POSCAR_H0C0", "energy_atom": -5.0,
                 "converged": True}) + "\n")
            (d / "bshard1.jsonl").write_text(json.dumps(
                {"tid": "chem__proto/POSCAR_H0C1", "energy_atom": None,
                 "converged": False}) + "\n")
            done = read_results(d)
        self.assertEqual(set(done), {"chem__proto/POSCAR_H0C0"})  # failed row kept out


def model_available() -> bool:
    return MODEL_PATH.is_file()


def batch_available() -> bool:
    try:
        import torch_sim  # noqa: F401
        from torch_sim.models.mace import MaceModel  # noqa: F401
    except ImportError:
        return False
    return MODEL_PATH.is_file()


@unittest.skipUnless(model_available(), "MACE checkpoint not in models/")
class RelaxTests(unittest.TestCase):
    def test_cu_bulk_relaxes_to_fcc_energy(self):
        from ase.build import bulk

        from loopml.mlip import relax_energy

        result = relax_energy(bulk("Cu", crystalstructure="fcc", a=3.7), fmax=0.02, steps=150)
        self.assertTrue(result["converged"])
        self.assertLess(result["energy_atom"], -3.3)  # cohesive sanity
        self.assertGreater(result["energy_atom"], -4.5)


@unittest.skipUnless(batch_available(), "MACE checkpoint + torch-sim not available")
class BatchRelaxTests(unittest.TestCase):
    def test_batch_matches_ase_bfgs(self):
        from ase.build import bulk

        from loopml.mlip import relax_batch, relax_energy

        systems = [
            bulk("Cu", crystalstructure="fcc", a=3.7),
            bulk("Cu", crystalstructure="fcc", a=3.7).repeat((2, 2, 2)),
            bulk("MgO", crystalstructure="rocksalt", a=4.2).repeat((2, 2, 2)),
        ]
        batched = relax_batch(systems, fmax=0.02, steps=500)
        for atoms, result in zip(systems, batched):
            reference = relax_energy(atoms, fmax=0.02, steps=200)
            self.assertTrue(result["converged"])
            self.assertTrue(reference["converged"])
            self.assertLess(abs(result["energy_atom"] - reference["energy_atom"]), 1e-4)
            self.assertLess(
                abs(result["volume_per_atom"] - reference["volume_per_atom"]), 0.05
            )
            self.assertAlmostEqual(
                result["static_energy_atom"], reference["static_energy_atom"], places=4
            )


if __name__ == "__main__":
    unittest.main()


def test_efa_pick_lowest_energy_boltzmann():
    from loopml.mlip import efa_pick_lowest_energy
    ensembles = [
        {"name": "lowT-winner", "energies": [-9.0, -8.0], "dgs": [1, 1]},
        {"name": "loser", "energies": [-8.0, -7.0], "dgs": [1, 1]},
    ]
    pick = efa_pick_lowest_energy(ensembles, temperature=50.0)
    assert pick["selected"] == "lowT-winner"
    assert len(pick["candidates"]) == 2
    assert pick["efa"] > 0 and pick["temperature"] == 50.0
