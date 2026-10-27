"""
EVOCURE v1.0.0 Full End-to-End API and Integration Verification Suite.
Validates all endpoints, models, responses, and epistemic contracts.
"""

import sys
import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestEvocureWave4B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_version(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("version"), "1.0.0")
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("active_disease", data)
        self.assertIn("ai_runtime", data)

    def test_02_diseases_and_selection(self):
        # List diseases
        res = self.client.get("/api/v1/diseases")
        self.assertEqual(res.status_code, 200)
        diseases = res.json()
        self.assertGreaterEqual(len(diseases), 1)

        # Select target disease
        sel_res = self.client.post("/api/v1/diseases/select", json={"disease_id": "NSCLC_KRAS", "seed": 42})
        self.assertEqual(sel_res.status_code, 200)
        sel_data = sel_res.json()
        self.assertTrue(sel_data.get("success"))

        # Switch back to PDAC
        sel_pdac = self.client.post("/api/v1/diseases/select", json={"disease_id": "PDAC", "seed": 1337})
        self.assertEqual(sel_pdac.status_code, 200)

    def test_03_editor_operations_and_apply(self):
        # Get operations
        res = self.client.get("/api/v1/editor/operations")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("substituents", data)
        self.assertIn("atom_swaps", data)
        self.assertIn("ring_transformations", data)

        # Apply substituent
        gemcitabine_smiles = "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
        edit_res = self.client.post("/api/v1/editor/apply", json={
            "smiles": gemcitabine_smiles,
            "operation": "add_substituent",
            "substituent": "fluoro"
        })
        self.assertEqual(edit_res.status_code, 200)
        edit_data = edit_res.json()
        self.assertTrue(edit_data.get("success"))
        details = edit_data.get("data", {})
        self.assertIn("parent_svg", details)
        self.assertIn("derivative_svg", details)
        self.assertIn("property_deltas", details)

    def test_04_editor_diff(self):
        smiles_a = "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
        smiles_b = "CC1=CN(C(=O)NC1=O)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F"
        diff_res = self.client.post("/api/v1/editor/diff", json={
            "smiles_a": smiles_a,
            "smiles_b": smiles_b
        })
        self.assertEqual(diff_res.status_code, 200)
        diff_data = diff_res.json()
        self.assertTrue(diff_data.get("success"))
        data = diff_data.get("data", {})
        self.assertIn("mcs_smarts", data)
        self.assertIn("svg_a", data)
        self.assertIn("svg_b", data)

    def test_05_tumor_combination_synergy(self):
        combo_res = self.client.post("/api/v1/tumor/combination", json={
            "compound_a": "Gemcitabine",
            "dose_a_um": 5.0,
            "compound_b": "Sotorasib",
            "dose_b_um": 10.0,
            "hours": 48.0
        })
        self.assertEqual(combo_res.status_code, 200)
        combo_data = combo_res.json()
        self.assertTrue(combo_data.get("success"))
        res = combo_data.get("data", {})
        self.assertIn("bliss_excess_pct", res)
        self.assertIn("loewe_ci", res)
        self.assertIn("synergy_matrix", res)
        self.assertEqual(len(res.get("synergy_matrix", [])), 8)
        self.assertIn("trial_arms", res)

    def test_06_tumor_coevolution(self):
        coev_res = self.client.post("/api/v1/tumor/coevolution", json={
            "n_cycles": 2,
            "hours_per_cycle": 12.0,
            "applied_concentration_um": 5.0
        })
        self.assertEqual(coev_res.status_code, 200)
        data = coev_res.json()
        self.assertTrue(data.get("success"))

    def test_07_assistant_status_and_triage(self):
        # Assistant status
        st_res = self.client.get("/api/v1/assistant/status")
        self.assertEqual(st_res.status_code, 200)
        st_data = st_res.json()
        self.assertIn("gemma", st_data)
        self.assertIn("laya", st_data)
        self.assertIn("qsar", st_data)

        # Laya triage
        tr_res = self.client.post("/api/v1/assistant/laya-triage")
        self.assertEqual(tr_res.status_code, 200)
        tr_data = tr_res.json()
        self.assertTrue(tr_data.get("success"))
        triage_info = tr_data.get("data", {})
        self.assertIn("ranked_compounds", triage_info)

    def test_08_genealogy_and_export(self):
        gen_res = self.client.get("/api/v1/genealogy?min_similarity=0.2&min_persistence=1")
        self.assertEqual(gen_res.status_code, 200)
        gdata = gen_res.json()
        self.assertIn("nodes", gdata)
        self.assertIn("edges", gdata)

        exp_res = self.client.get("/api/v1/genealogy/export")
        self.assertEqual(exp_res.status_code, 200)
        edata = exp_res.json()
        self.assertEqual(edata.get("export_format"), "evocure_genealogy_dag_v1")

    def test_09_experiments_and_pdf_export(self):
        # Save experiment
        save_res = self.client.post("/api/v1/experiments/save", json={"title": "E2E Automated Verification Test"})
        self.assertEqual(save_res.status_code, 200)
        sdata = save_res.json().get("data", {})
        exp_id = sdata.get("experiment_id")
        self.assertTrue(exp_id)

        # List experiments
        list_res = self.client.get("/api/v1/experiments")
        self.assertEqual(list_res.status_code, 200)
        exps = list_res.json()
        self.assertGreaterEqual(len(exps), 1)

        # Export as PDF for specific exp_id
        pdf_res = self.client.get(f"/api/v1/export/{exp_id}/pdf")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers.get("content-type"), "application/pdf")
        self.assertTrue(pdf_res.content.startswith(b"%PDF"))

        # Export as PDF for latest session
        pdf_latest = self.client.get("/api/v1/export/latest/pdf")
        self.assertEqual(pdf_latest.status_code, 200)
        self.assertEqual(pdf_latest.headers.get("content-type"), "application/pdf")
        self.assertTrue(pdf_latest.content.startswith(b"%PDF"))

    def test_10_compare_worlds(self):
        cmp_res = self.client.post("/api/v1/compare-worlds", json={
            "scenario_left": "chemical_life",
            "scenario_right": "cancer_battle",
            "steps": 20,
            "repeats": 1,
            "seed": 42
        })
        self.assertEqual(cmp_res.status_code, 200)
        cdata = cmp_res.json()
        self.assertTrue(cdata.get("success"))
        self.assertIn("metrics_left", cdata.get("data", {}))
        self.assertIn("metrics_right", cdata.get("data", {}))


if __name__ == "__main__":
    unittest.main()
