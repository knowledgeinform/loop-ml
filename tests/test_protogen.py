import json
import re
import unittest
from pathlib import Path

from loopml.protogen import (
    ProtogenError,
    aflow_binary,
    assign_sites,
    build_partcar_multi,
    parse_composition,
    parse_pocc_output,
)

POCC_FIXTURE = """\
some log noise 00000 MESSAGE
[VASP_POSCAR_MODE_EXPLICIT]START
Hf0.2Nb0.2Ta0.2Ti0.2Zr0.2O H0C113 HNF(n=5,1/5)=[1 0 0; 0 1 0; 0 0 5] site_config(114/120)=[5 3 4 1 0; 2 2 2 2 2] DG=1 [POCC doi: 10.1021/acs.chemmater.6b01449]
1.000000
   0.00000000000000   2.39707456053384   2.39707456053384
   2.39707456053384   0.00000000000000   2.39707456053384
   11.98537280266921  11.98537280266921   0.00000000000000
1 1 5 1 1 1
 Direct(10) [A1B1C5D1E1F1]
   0.00000000000000   0.00000000000000   0.00000000000000  Hf
   0.00000000000000   0.00000000000000   0.80000000000000  Nb
   0.50000000000000   0.50000000000000   0.30000000000000  O
   0.50000000000000   0.50000000000000   0.50000000000000  O
   0.50000000000000   0.50000000000000   0.70000000000000  O
   0.50000000000000   0.50000000000000   0.90000000000000  O
   0.50000000000000   0.50000000000000   0.10000000000000  O
   0.00000000000000   0.00000000000000   0.40000000000000  Ta
   0.00000000000000   0.00000000000000   0.60000000000000  Ti
   0.00000000000000   0.00000000000000   0.20000000000000  Zr
[VASP_POSCAR_MODE_EXPLICIT]STOP
trailing noise
"""

ROCKSALT = {
    "prototype": "AB_cF8_225_a_b",
    "lattice": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    "blocks": [
        {"dummy": "Ti", "mult": 1, "coords": [(0.0, 0.0, 0.0)]},
        {"dummy": "O", "mult": 1, "coords": [(0.5, 0.5, 0.5)]},
    ],
}

SPINEL = {
    "prototype": "AB2C4_cF56_227_b_c_e-001",
    "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "blocks": [
        {"dummy": "Mg", "mult": 2, "coords": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]},
        {"dummy": "Ti", "mult": 4, "coords": [(0.5, 0.0, 0.0)] * 4},
        {"dummy": "O", "mult": 8, "coords": [(0.25, 0.25, 0.25)] * 8},
    ],
}

EQUIMOLAR_5 = {"Hf": 0.1, "Nb": 0.1, "Ta": 0.1, "Ti": 0.1, "Zr": 0.1, "O": 0.5}


def aflow_available() -> bool:
    try:
        aflow_binary()
        return True
    except ProtogenError:
        return False


class CompositionTests(unittest.TestCase):
    def test_formula_with_fractions(self):
        comp = parse_composition("Hf0.2Nb0.2Ta0.2Ti0.2Zr0.2O")
        self.assertAlmostEqual(sum(comp.values()), 1.0)
        self.assertAlmostEqual(comp["O"], 0.5)
        self.assertAlmostEqual(comp["Hf"], 0.1)

    def test_integer_formula_normalized(self):
        comp = parse_composition("La1Sr1Mn3O8")
        self.assertAlmostEqual(comp["O"], 8 / 13)
        self.assertAlmostEqual(comp["Mn"], 3 / 13)

    def test_bad_formula(self):
        with self.assertRaises(ProtogenError):
            parse_composition("")


class SiteAssignmentTests(unittest.TestCase):
    def test_single_site_all_elements(self):
        assignment = assign_sites(EQUIMOLAR_5, ROCKSALT)
        self.assertEqual(set(assignment.values()), {0})

    def test_two_site_packing(self):
        comp = {"Ti": 1 / 3, "Mg": 1 / 6, "O": 0.5}
        assignment = assign_sites(comp, SPINEL)
        self.assertEqual(assignment["Ti"], 1)  # 4x supply -> big site
        self.assertEqual(assignment["Mg"], 0)  # 1x -> small site

    def test_equimolar_spinel_needs_splitting(self):
        # equal quaternary supply cannot fit 2:4 integer sites; packing
        # reports failure rather than silently changing composition
        self.assertIsNone(assign_sites(EQUIMOLAR_5, SPINEL))

    def test_requires_oxygen(self):
        with self.assertRaises(ProtogenError):
            assign_sites({"Ti": 0.5, "Zr": 0.5}, ROCKSALT)


class PartcarMultiTests(unittest.TestCase):
    def test_species_globally_alphabetized(self):
        assignment = assign_sites(EQUIMOLAR_5, ROCKSALT)
        partcar = build_partcar_multi(EQUIMOLAR_5, ROCKSALT, 4.2, assignment)
        species = [ln.split()[-1] for ln in partcar.splitlines()[7:]]
        self.assertEqual(species, ["Hf", "Nb", "O", "Ta", "Ti", "Zr"])

    def test_occupancies_scale_with_site_multiplicity(self):
        comp = {"Ti": 2 / 3, "Mg": 1 / 3, "O": 0.5}
        assignment = assign_sites(comp, SPINEL)
        partcar = build_partcar_multi(comp, SPINEL, 8.4, assignment)
        counts = partcar.splitlines()[5].split()
        # Ti (fraction 2/3 of cations) x 6 cation-sites / 4-site-block = 1.0
        # Mg (1/3) x 6 / 2 = 1.0; O block full
        # alphabetized blocks: Mg-site(2x1=full), O block, Ti-site(4x1=full)
        self.assertEqual(counts, ["2*1", "8", "4*1"])

    def test_occupancy_cap_rejects_bad_assignment(self):
        comp = {"Ti": 0.9, "Mg": 0.1, "O": 0.5}
        assignment = {"Ti": 0, "Mg": 1}  # Ti supply 3.0 on a 2-site: 1.5 > 1
        with self.assertRaises(ProtogenError):
            build_partcar_multi(comp, SPINEL, 8.4, assignment)


class ParseOutputTests(unittest.TestCase):
    def test_fixture_block(self):
        (structure,) = parse_pocc_output(POCC_FIXTURE)
        self.assertEqual(structure["id"], "H0C113")
        self.assertEqual(structure["det"], 5)
        self.assertEqual(structure["hnf_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 5]])
        self.assertEqual(structure["degeneracy"], 1)
        self.assertEqual(structure["site_config_count"], 120)
        self.assertEqual(structure["counts"], {"Hf": 1, "Nb": 1, "O": 5, "Ta": 1, "Ti": 1, "Zr": 1})
        poscar_lines = structure["poscar"].splitlines()
        self.assertEqual(poscar_lines[0], "Hf1Nb1O5Ta1Ti1Zr1")  # comment line (VASP5)
        self.assertEqual(poscar_lines[1], "1.0")
        self.assertEqual(poscar_lines[5], "Hf Nb O Ta Ti Zr")
        self.assertEqual(poscar_lines[7], "Direct")
        self.assertNotIn("Direct(", structure["poscar"])


@unittest.skipUnless(aflow_available(), "aflow binary not configured")
class IntegrationTests(unittest.TestCase):
    def test_rocksalt_five_component_sample(self):
        import tempfile

        from loopml.protogen import generate_prototypes

        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_prototypes(EQUIMOLAR_5, Path(tmp), sample_number=3)
            self.assertEqual(len(manifest["structures"]), 3)
            self.assertEqual(manifest["schema"], "pocc-protogen-v2")
            ids = {entry["id"] for entry in manifest["structures"]}
            self.assertTrue(all(re.fullmatch(r"H\d+C\d+", i) for i in ids))
            for entry in manifest["structures"]:
                self.assertTrue((Path(tmp) / entry["file"]).is_file())
                self.assertEqual(entry["counts"], {"Hf": 1, "Nb": 1, "O": 5, "Ta": 1, "Ti": 1, "Zr": 1})
            json.loads((Path(tmp) / "manifest.json").read_text())

    def test_deterministic(self):
        import tempfile

        from loopml.protogen import generate_prototypes

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            m1 = generate_prototypes(EQUIMOLAR_5, Path(tmp1), sample_number=2)
            m2 = generate_prototypes(EQUIMOLAR_5, Path(tmp2), sample_number=2)
            self.assertEqual(
                [e["sha256"] for e in m1["structures"]],
                [e["sha256"] for e in m2["structures"]],
            )


if __name__ == "__main__":
    unittest.main()
