"""
EVOCURE AI Model & Resource Ledger (Orchestration).

Tracks loaded local models, versions, hardware resource consumption (RAM, CPU, VRAM),
and maintains an audit log of AI explanations and triage decisions per Sections 2.15, 3, and 27.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from simulation.core.contracts import EpistemicCategory
from simulation.core.paths import project_root

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata and resource tracking for a loaded local model."""
    model_id: str
    model_type: str  # 'qsar_predictor', 'llm_assistant', 'decision_engine'
    version: str
    artifact_path: str
    status: str  # 'unloaded', 'loaded', 'active', 'failed'
    provider: str
    ram_mb_estimated: float = 0.0
    vram_mb_estimated: float = 0.0
    loaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    inference_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.inference_count, 2)


@dataclass
class DecisionAuditEntry:
    """Immutable audit record for an AI reasoning, explanation, or triage event."""
    audit_id: str
    timestamp: str
    module: str  # 'gemma_assistant', 'laya_engine', 'qsar_predictor'
    action_type: str  # 'explanation', 'event_classification', 'candidate_triage', 'prediction'
    query_summary: str
    output_summary: str
    provider: str
    epistemic_category: str
    latency_ms: float
    guardrails_triggered: List[str] = field(default_factory=list)
    proposed_actions: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """
    Central Ledger for AI Models, System Resources, and Decision Auditing.
    """

    _instance: Optional[ModelRegistry] = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._models: Dict[str, ModelMetadata] = {}
        self._audit_log: List[DecisionAuditEntry] = []
        self._counter: int = 0
        self._initialized = True
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default EVOCURE scientific models."""
        base_dir = project_root()

        # 1. PDAC Calibrated QSAR
        qsar_path = base_dir / "models" / "scientific" / "pdac_qsar_v1.joblib"
        self.register_model(
            model_id="pdac_qsar_v1",
            model_type="qsar_predictor",
            version="1.0.0",
            artifact_path=str(qsar_path),
            provider="scikit-learn (RandomForest + Split-Conformal + Sigmoid CalibratedClassifierCV)",
            status="loaded" if qsar_path.exists() else "unloaded",
            ram_mb=55.0,
        )

        # 2. Local Gemma 4 Assistant
        gemma_path = base_dir / "models" / "gemma" / "Gemma-4-E2B-Heretic-Q4_K_M.gguf"
        self.register_model(
            model_id="local_gemma_4",
            model_type="llm_assistant",
            version="1.0.0",
            artifact_path=str(gemma_path),
            provider="local_llama_runtime / deterministic_fallback",
            status="loaded" if gemma_path.exists() else "unloaded",
            ram_mb=1850.0,
            vram_mb=1850.0,
        )

        # 3. Laya Structured Decision Engine
        laya_path = base_dir / "models" / "laya" / "multilingual" / "model.safetensors"
        self.register_model(
            model_id="laya_decision_v1",
            model_type="decision_engine",
            version="1.0.0",
            artifact_path=str(laya_path),
            provider="laya_multilingual_neural_plus_calibrated_rules",
            status="loaded" if laya_path.exists() else "unloaded",
            ram_mb=620.0,
        )

        # 4. Google Gemini API (Multi-Provider AI)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.register_model(
            model_id="google_gemini_api",
            model_type="llm_assistant",
            version="2.5",
            artifact_path="https://generativelanguage.googleapis.com/v1beta",
            provider="Google Gemini (REST API)",
            status="configured" if gemini_key else "optional_unconfigured",
            ram_mb=0.0,
            vram_mb=0.0,
        )

        # 5. OpenAI API (Multi-Provider AI)
        openai_key = os.getenv("OPENAI_API_KEY", "")
        self.register_model(
            model_id="openai_api",
            model_type="llm_assistant",
            version="4o-mini",
            artifact_path=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            provider="OpenAI / Compatible REST API",
            status="configured" if openai_key else "optional_unconfigured",
            ram_mb=0.0,
            vram_mb=0.0,
        )

    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        artifact_path: str,
        provider: str,
        status: str = "unloaded",
        ram_mb: float = 0.0,
        vram_mb: float = 0.0,
    ) -> ModelMetadata:
        """Register or update model specifications in the ledger."""
        meta = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            artifact_path=artifact_path,
            status=status,
            provider=provider,
            ram_mb_estimated=ram_mb,
            vram_mb_estimated=vram_mb,
        )
        self._models[model_id] = meta
        return meta

    def update_model_status(
        self, model_id: str, status: str, provider: Optional[str] = None
    ) -> None:
        """Update live execution status of a registered model."""
        if model_id in self._models:
            self._models[model_id].status = status
            if provider:
                self._models[model_id].provider = provider

    def record_inference(
        self, model_id: str, latency_ms: float
    ) -> None:
        """Record an inference event with its latency."""
        if model_id in self._models:
            meta = self._models[model_id]
            meta.inference_count += 1
            meta.total_latency_ms += latency_ms

    def log_ai_decision(
        self,
        module: str,
        action_type: str,
        query_summary: str,
        output_summary: str,
        provider: str,
        epistemic_category: Union[EpistemicCategory, str],
        latency_ms: float = 0.0,
        guardrails_triggered: Optional[List[str]] = None,
        proposed_actions: Optional[Dict[str, Any]] = None,
    ) -> DecisionAuditEntry:
        """
        Record an immutable audit log entry for any AI explanation, triage, or prediction.
        """
        self._counter += 1
        audit_id = f"AUDIT-AI-{self._counter:06d}"
        now_iso = datetime.now(timezone.utc).isoformat()
        
        ep_val = (
            epistemic_category.value
            if isinstance(epistemic_category, EpistemicCategory)
            else str(epistemic_category)
        )

        entry = DecisionAuditEntry(
            audit_id=audit_id,
            timestamp=now_iso,
            module=module,
            action_type=action_type,
            query_summary=query_summary[:200],
            output_summary=output_summary[:300],
            provider=provider,
            epistemic_category=ep_val,
            latency_ms=round(latency_ms, 2),
            guardrails_triggered=guardrails_triggered or [],
            proposed_actions=proposed_actions,
        )
        self._audit_log.append(entry)
        return entry

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        """Collect real-time CPU, RAM, and GPU memory telemetry."""
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        proc_mem = round(proc.memory_info().rss / (1024 ** 2), 2)

        # Vendor neutral: NVIDIA, AMD, Intel and Apple are all reported here.
        gpu_info: Optional[Dict[str, Any]] = None
        all_gpus: List[Dict[str, Any]] = []
        try:
            from simulation.core.hardware import detect_gpus, primary_gpu
            all_gpus = detect_gpus()
            gpu_info = primary_gpu()
        except Exception:
            pass

        compute_backend: Dict[str, Any] = {}
        try:
            from simulation.core.acceleration import status as compute_status
            report = compute_status()
            compute_backend = {
                "backend": report["gpu_backend"],
                "vendor": report["gpu_vendor"],
                "precision": report["precision"],
                "numerical_backend": report["numerical_backend"],
                "preference": report["preference"],
            }
        except Exception:
            pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_mb": round(mem.total / (1024 ** 2), 1),
            "ram_available_mb": round(mem.available / (1024 ** 2), 1),
            "ram_used_pct": mem.percent,
            "process_rss_mb": proc_mem,
            "gpu": gpu_info,
            "gpus": all_gpus,
            "compute_backend": compute_backend,
        }

    def get_ledger_snapshot(self) -> Dict[str, Any]:
        """Return comprehensive snapshot of all registered models and resource usage."""
        return {
            "models": {
                mid: {
                    "model_id": m.model_id,
                    "model_type": m.model_type,
                    "version": m.version,
                    "artifact_path": m.artifact_path,
                    "status": m.status,
                    "provider": m.provider,
                    "ram_mb_estimated": m.ram_mb_estimated,
                    "vram_mb_estimated": m.vram_mb_estimated,
                    "inference_count": m.inference_count,
                    "avg_latency_ms": m.average_latency_ms,
                    "loaded_at": m.loaded_at,
                }
                for mid, m in self._models.items()
            },
            "telemetry": self.get_hardware_telemetry(),
            "audit_entries_count": len(self._audit_log),
        }

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent AI decision audit entries."""
        slice_entries = self._audit_log[-limit:] if limit > 0 else self._audit_log
        return [asdict(e) for e in reversed(slice_entries)]

    def export_audit_log(self, filepath: Union[str, Path]) -> None:
        """Export the full decision audit log to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self._audit_log], f, indent=2)


def get_ai_registry() -> ModelRegistry:
    """Retrieve singleton ModelRegistry instance."""
    return ModelRegistry()
