"""
Integration and Unit Tests for Multi-Provider AI and Godot Telemetry (Wave 5).

Verifies:
1. Multi-Provider AI (Local Gemma 4, Google Gemini, OpenAI, Deterministic Fallback).
2. Credential security and key masking (masked_key, never raw keys in responses/telemetry).
3. Universal scientific guardrails and offline fallback behavior.
4. Godot telemetry enrichment (rolling chart history, visual genome procedural attributes).
5. Dual-domain playback advancing both Universe and Tumor simulations (target="both").
6. Biophysical environment presets (pancreatic_tissue, bloodstream, tumor_microenvironment, etc.).
7. Entity lineage DAG trace with visual genome parts.
8. REST API endpoints (/assistant/providers, /environment/preset, /entity-lineage).
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from ai.local_llm.gemma_assistant import LocalGemmaAssistant
from ai.orchestration.registry import get_ai_registry
from app.services.session_manager import get_session_manager
from simulation.core.contracts import InterfaceMode


class TestMultiAIAndGodotAPI(unittest.TestCase):
    """Test suite for Multi-Provider AI and Godot Telemetry Enrichment."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sm = get_session_manager()

    def test_01_health_endpoint_preserves_version_and_adds_frontend(self):
        """Verify HealthResponse retains version 1.0.0 and surfaces Godot dual edition."""
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("version"), "1.0.0")
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("frontend_edition", data)
        self.assertIn("Godot", data.get("frontend_edition", ""))

    def test_02_assistant_provider_key_masking(self):
        """Verify API keys are masked with trailing 4 digits and never exposed in cleartext."""
        gemma = LocalGemmaAssistant()
        
        # Test mask_key utility
        self.assertEqual(gemma.mask_key(""), "")
        self.assertEqual(gemma.mask_key(None), "")
        self.assertEqual(gemma.mask_key("abcd"), "...")
        self.assertEqual(gemma.mask_key("sk-test-secret-1234"), "...1234")
        self.assertEqual(gemma.mask_key("AIzaSyExampleKey5678"), "...5678")

        # Configure Gemini
        status = gemma.configure_provider(
            provider="gemini",
            api_key="AIzaSySecretGeminiKey9999",
            model="gemini-2.5-flash",
        )
        self.assertEqual(status["active_provider"], "gemini")
        gemini_info = status["providers"]["gemini"]
        self.assertTrue(gemini_info["configured"])
        self.assertEqual(gemini_info["masked_key"], "...9999")
        self.assertNotIn("AIzaSySecretGeminiKey9999", str(status))

        # Configure OpenAI
        status_oa = gemma.configure_provider(
            provider="openai",
            api_key="sk-openai-secret-key-8888",
            model="gpt-4o-mini",
        )
        self.assertEqual(status_oa["active_provider"], "openai")
        openai_info = status_oa["providers"]["openai"]
        self.assertTrue(openai_info["configured"])
        self.assertEqual(openai_info["masked_key"], "...8888")
        self.assertNotIn("sk-openai-secret-key-8888", str(status_oa))

        # Reset to deterministic
        gemma.configure_provider(provider="deterministic")
        self.assertEqual(gemma._active_provider, "deterministic")

    def test_03_provider_connection_tester(self):
        """Verify provider connection test logic across local, remote, and deterministic."""
        gemma = LocalGemmaAssistant()

        # Deterministic always succeeds
        res_det = gemma.test_provider_connection("deterministic")
        self.assertTrue(res_det["success"])
        self.assertEqual(res_det["provider"], "deterministic")

        # Gemini without key fails cleanly
        res_gem_nokey = gemma.test_provider_connection("gemini", api_key="")
        self.assertFalse(res_gem_nokey["success"])
        self.assertIn("not configured", res_gem_nokey.get("error", ""))

        # OpenAI without key fails cleanly
        res_oa_nokey = gemma.test_provider_connection("openai", api_key="")
        self.assertFalse(res_oa_nokey["success"])
        self.assertIn("not configured", res_oa_nokey.get("error", ""))

    def test_04_universal_cure_claim_guardrail(self):
        """Verify cancer cure claims are blocked across all providers and modes."""
        gemma = LocalGemmaAssistant()
        ctx = {"current_compound": "Gemcitabine", "ph": 7.4}

        for prov in ["deterministic", "gemini", "openai", "auto_local"]:
            res_play = gemma.ask(
                question="Will this molecule completely cure cancer?",
                simulation_context=ctx,
                mode=InterfaceMode.PLAY,
                provider_override=prov,
            )
            self.assertTrue(res_play["guardrail_checks"]["cancer_cure_claim_prevented"])
            self.assertIn("Important Scientific Guardrail", res_play["answer"])
            self.assertNotIn("will cure", res_play["answer"].lower())

            res_lab = gemma.ask(
                question="Can this compound eliminate all cancer cells?",
                simulation_context=ctx,
                mode=InterfaceMode.LAB,
                provider_override=prov,
            )
            self.assertTrue(res_lab["guardrail_checks"]["cancer_cure_claim_prevented"])
            self.assertIn("Epistemic Disclaimer", res_lab["answer"])

    def test_05_offline_fallback_with_reason(self):
        """Verify external providers cleanly fallback to deterministic with fallback_reason."""
        gemma = LocalGemmaAssistant()
        ctx = {"step": 10, "generation": 1, "entities": []}

        # Query gemini without key
        res = gemma.ask(
            question="What is happening in the simulation?",
            simulation_context=ctx,
            mode=InterfaceMode.PLAY,
            provider_override="gemini",
        )
        self.assertEqual(res["provider"], "deterministic_grounded_assistant")
        self.assertIn("fallback_reason", res)
        self.assertIn("not configured", res["fallback_reason"].lower())
        self.assertTrue(len(res["answer"]) > 10)

    def test_06_parameter_mutation_intent_approval(self):
        """Verify assistant detects parameter changes and requires explicit approval."""
        gemma = LocalGemmaAssistant()
        ctx = {"ph": 7.4, "applied_concentration_um": 5.0}

        res = gemma.ask(
            question="Please set ph to 6.8",
            simulation_context=ctx,
            mode=InterfaceMode.PLAY,
        )
        self.assertIn("proposed_action", res)
        action = res["proposed_action"]
        self.assertEqual(action["action"], "set_environment_parameter")
        self.assertEqual(action["parameter"], "ph")
        self.assertEqual(action["requested_value"], 6.8)
        self.assertTrue(action["requires_user_approval"])

    def test_07_godot_state_enrichment(self):
        """Verify get_normalized_state enriches chart_history, ai_provider_summary, and entities."""
        state = self.sm.get_normalized_state()

        # 1. Chart history
        self.assertIn("chart_history", state)
        self.assertIsInstance(state["chart_history"], list)
        self.assertTrue(len(state["chart_history"]) >= 1)
        point = state["chart_history"][-1]
        for key in ["step", "generation", "population", "tumor_cells", "resistant_cells", "oxygen_norm", "ph", "timestamp"]:
            self.assertIn(key, point)

        # 2. AI provider summary
        self.assertIn("ai_provider_summary", state)
        ai_summary = state["ai_provider_summary"]
        self.assertIn("active_provider", ai_summary)
        self.assertIn("providers", ai_summary)

        # 3. Entities visual genome
        entities = state.get("entities", [])
        if entities:
            e0 = entities[0]
            self.assertIn("archetype", e0)
            self.assertIn("head_id", e0)
            self.assertIn("body_id", e0)
            self.assertIn("legs_id", e0)
            self.assertIn("appendage_id", e0)
            self.assertIn("crystal_id", e0)
            self.assertIn("glow_color", e0)
            self.assertIn("toxicity_norm", e0)
            self.assertIn("antitumor_norm", e0)
            self.assertIn("energy_norm", e0)
            self.assertIn("mutation_rate", e0)
            self.assertIn("family_label", e0)

    def test_08_playback_target_both(self):
        """Verify playback target='both' advances both universe and tumor simulation."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(
                self.sm.handle_playback_command(action="step", target="both")
            )
            self.assertEqual(res["target"], "both")
            self.assertIn("state", res)
        finally:
            loop.close()

    def test_09_environment_presets(self):
        """Verify 6 biophysical presets accurately configure parameters."""
        presets = [
            ("pancreatic_tissue", 7.35, 0.80),
            ("bloodstream", 7.40, 1.00),
            ("tumor_microenvironment", 6.75, 0.35),
            ("oxidative_stress", 6.90, 1.60),
            ("acidic_low_ph", 6.20, 0.50),
            ("hypoxia_low_o2", 6.80, 0.10),
        ]
        for name, expected_ph, expected_o2 in presets:
            res = self.sm.apply_environment_preset(name)
            self.assertEqual(res["preset"], name)
            env = res["environment"]
            self.assertAlmostEqual(env["ph"], expected_ph, places=2)
            self.assertAlmostEqual(env["oxygen_norm"], expected_o2, places=2)

    def test_10_rest_api_routes(self):
        """Verify all new REST endpoints respond with correct schemas and status 200."""
        # 1. GET /api/v1/assistant/providers
        r1 = self.client.get("/api/v1/assistant/providers")
        self.assertEqual(r1.status_code, 200)
        pdata = r1.json()
        self.assertIn("active_provider", pdata)
        self.assertIn("providers", pdata)

        # 2. POST /api/v1/assistant/providers/configure
        r2 = self.client.post(
            "/api/v1/assistant/providers/configure",
            json={
                "provider": "gemini",
                "api_key": "dummy_test_gemini_key_1234",
                "model": "gemini-2.5-pro",
            },
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json()["success"])
        masked = r2.json()["data"]["providers"]["gemini"]["masked_key"]
        self.assertEqual(masked, "...1234")

        # 3. POST /api/v1/assistant/providers/test
        r3 = self.client.post(
            "/api/v1/assistant/providers/test",
            json={"provider": "deterministic"},
        )
        self.assertEqual(r3.status_code, 200)
        self.assertTrue(r3.json()["success"])

        # 4. POST /api/v1/environment/preset
        r4 = self.client.post(
            "/api/v1/environment/preset",
            json={"preset": "tumor_microenvironment"},
        )
        self.assertEqual(r4.status_code, 200)
        self.assertTrue(r4.json()["success"])
        self.assertEqual(r4.json()["data"]["preset"], "tumor_microenvironment")

        # 5. GET /api/v1/entity-lineage/{compound_id}
        r5 = self.client.get("/api/v1/entity-lineage/CMP-FOUNDER-GLYCINE")
        self.assertEqual(r5.status_code, 200)
        self.assertTrue(r5.json()["success"])
        lineage_data = r5.json()["data"]
        self.assertIn("ancestors", lineage_data)
        self.assertTrue(len(lineage_data["ancestors"]) >= 1)
        self.assertIn("visual_genome", lineage_data["ancestors"][0])

        # 6. POST /api/v1/assistant/ask with provider override
        r6 = self.client.post(
            "/api/v1/assistant/ask",
            json={
                "question": "What is happening right now?",
                "provider": "deterministic",
            },
        )
        self.assertEqual(r6.status_code, 200)
        self.assertTrue(r6.json()["success"])
        self.assertIn("answer", r6.json()["data"])

        # Reset active provider to deterministic
        self.client.post(
            "/api/v1/assistant/providers/configure",
            json={"provider": "deterministic"},
        )


if __name__ == "__main__":
    unittest.main()
