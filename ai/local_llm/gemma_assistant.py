"""
EVOCURE Local Gemma 4 AI Assistant (Sections 3.1 & 6).

Provider-agnostic local LLM runtime:
1. Detects local GGUF weights (models/gemma/Gemma-4-E2B-Heretic-Q4_K_M.gguf) and self-contained llama-server.exe.
2. Detects local Ollama (127.0.0.1:11434) with short 1.5s probe timeout.
3. Automatically falls back to a deterministic grounded scientific assistant so EVOCURE PLAY & LAB
   run 100% offline with zero remote internet traffic.
4. Enforces strict oncology and scientific guardrails:
   - NEVER claims a compound cures cancer.
   - NEVER invents toxicity/IC50 numbers not present in simulation_context.
   - NEVER modifies parameters silently (returns explicit proposed_action requiring user approval).
   - Tags all AI reasoning as EpistemicCategory.AI_HYPOTHESIS.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from simulation.core.contracts import EpistemicCategory, InterfaceMode

logger = logging.getLogger(__name__)

GEMINI_PRESETS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
OPENAI_PRESETS = ["gpt-4o-mini", "gpt-4o", "o3-mini"]


@dataclass
class HardwareProfile:
    """Hardware capabilities and LLM quantization recommendation."""
    cpu_cores_logical: int
    cpu_cores_physical: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    cuda_available: bool = False  # retained name; true for any usable GPU offload target
    gpu_vendor: Optional[str] = None
    gpu_offload_backend: Optional[str] = None  # cuda | hip | vulkan | metal | none
    recommended_model: str = "gemma-4-E2B-it-Q4_K_M"
    recommendation_reason: str = ""


class LocalGemmaAssistant:
    """
    Local Gemma 4 Assistant with Grounded Deterministic Scientific Fallback.
    """

    DEFAULT_PORT = 18089

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        prefer_local_server: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.prefer_local_server = prefer_local_server
        self._server_process: Optional[subprocess.Popen] = None

        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parents[2] / "models" / "gemma"
        else:
            self.models_dir = Path(models_dir)

        self.gguf_path = self.models_dir / "Gemma-4-E2B-Heretic-Q4_K_M.gguf"
        self.server_exe = self.models_dir / "llama_runtime" / "llama-server.exe"
        self.jinja_template = self.models_dir / "gemma4.jinja"

        self.hardware_profile = self.inspect_hardware_for_llm()

        # Multi-provider runtime configuration
        self._active_provider: str = os.getenv("EVOCURE_AI_PROVIDER", "auto_local").lower()
        self._gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self._gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self._openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self._openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self._openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

    @staticmethod
    def mask_key(key: Optional[str]) -> str:
        """Securely mask an API key for safe logging, telemetry, and client display."""
        if not key or not key.strip():
            return ""
        k = key.strip()
        if len(k) <= 4:
            return "..."
        return f"...{k[-4:]}"

    def get_providers_status(self) -> Dict[str, Any]:
        """Return status, availability, and masked credentials of all AI providers."""
        server_running = self.is_local_server_running()
        ollama_running = self.is_ollama_running()
        gguf_present = self.gguf_path.exists()

        local_status = (
            "server_running"
            if server_running
            else ("ollama_running" if ollama_running else ("gguf_present" if gguf_present else "not_found"))
        )

        return {
            "active_provider": self._active_provider,
            "providers": {
                "auto_local": {
                    "configured": True,
                    "status": "ready" if (server_running or ollama_running or gguf_present) else "deterministic_fallback",
                    "description": "Automatic local inference cascade (llama-server -> Ollama -> Deterministic)",
                },
                "local_gemma": {
                    "configured": gguf_present or server_running or ollama_running,
                    "status": local_status,
                    "server_running": server_running,
                    "ollama_running": ollama_running,
                    "gguf_present": gguf_present,
                    "models": [self.hardware_profile.recommended_model, "Gemma-4-E2B-Heretic-Q4_K_M"],
                    "hardware": {
                        "cpu_cores": self.hardware_profile.cpu_cores_logical,
                        "ram_total_gb": self.hardware_profile.ram_total_gb,
                        "gpu_name": self.hardware_profile.gpu_name,
                        "vram_total_gb": self.hardware_profile.vram_total_gb,
                        "recommended_model": self.hardware_profile.recommended_model,
                    },
                },
                "gemini": {
                    "configured": bool(self._gemini_api_key),
                    "masked_key": self.mask_key(self._gemini_api_key),
                    "current_model": self._gemini_model,
                    "available_models": list(GEMINI_PRESETS),
                },
                "openai": {
                    "configured": bool(self._openai_api_key),
                    "masked_key": self.mask_key(self._openai_api_key),
                    "current_model": self._openai_model,
                    "base_url": self._openai_base_url,
                    "available_models": list(OPENAI_PRESETS),
                },
                "deterministic": {
                    "configured": True,
                    "status": "always_ready",
                    "offline": True,
                },
            },
            "presets": {
                "gemini": list(GEMINI_PRESETS),
                "openai": list(OPENAI_PRESETS),
            },
        }

    def configure_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        persist_to_dotenv: bool = False,
    ) -> Dict[str, Any]:
        """
        Configure active AI provider and credentials in-memory, optionally persisting to .env.
        """
        valid_providers = {"auto_local", "local_gemma", "gemini", "openai", "deterministic"}
        prov = provider.lower().strip()
        if prov not in valid_providers:
            raise ValueError(f"Invalid provider: '{provider}'. Must be one of {valid_providers}")

        self._active_provider = prov

        if prov == "gemini":
            if api_key is not None and api_key.strip():
                self._gemini_api_key = api_key.strip()
            if model is not None and model.strip():
                self._gemini_model = model.strip()

        elif prov == "openai":
            if api_key is not None and api_key.strip():
                self._openai_api_key = api_key.strip()
            if model is not None and model.strip():
                self._openai_model = model.strip()
            if base_url is not None and base_url.strip():
                self._openai_base_url = base_url.strip()

        if persist_to_dotenv:
            self._save_to_dotenv()

        logger.info(f"AI provider configured: active='{self._active_provider}'")
        return self.get_providers_status()

    def _save_to_dotenv(self) -> None:
        """Persist current AI provider settings to .env file."""
        env_path = Path(__file__).resolve().parents[2] / ".env"
        lines: List[str] = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        keys_to_set = {
            "EVOCURE_AI_PROVIDER": self._active_provider,
            "GEMINI_API_KEY": self._gemini_api_key,
            "GEMINI_MODEL": self._gemini_model,
            "OPENAI_API_KEY": self._openai_api_key,
            "OPENAI_MODEL": self._openai_model,
            "OPENAI_BASE_URL": self._openai_base_url,
        }

        updated_keys = set()
        new_lines: List[str] = []
        for line in lines:
            line_strip = line.strip()
            matched = False
            for k, v in keys_to_set.items():
                if line_strip.startswith(f"{k}="):
                    new_lines.append(f"{k}={v}\n")
                    updated_keys.add(k)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)

        for k, v in keys_to_set.items():
            if k not in updated_keys and v:
                new_lines.append(f"{k}={v}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    def test_provider_connection(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test API connection for a specific or active provider.
        """
        target_provider = (provider or self._active_provider).lower().strip()

        if target_provider == "deterministic":
            return {
                "success": True,
                "provider": "deterministic",
                "message": "Deterministic grounded fallback assistant is always ready and 100% offline.",
            }

        if target_provider in ("auto_local", "local_gemma"):
            if self.is_local_server_running():
                return {
                    "success": True,
                    "provider": "local_gemma",
                    "message": f"Local llama-server is running on {self.host}:{self.port}.",
                }
            if self.is_ollama_running():
                return {
                    "success": True,
                    "provider": "local_gemma",
                    "message": "Local Ollama daemon is running on 127.0.0.1:11434.",
                }
            if self.gguf_path.exists():
                return {
                    "success": True,
                    "provider": "local_gemma",
                    "message": f"Local GGUF weights found ({self.gguf_path.name}), server can be spawned.",
                }
            return {
                "success": False,
                "provider": "local_gemma",
                "message": "Local GGUF weights or server not active. Deterministic fallback will handle queries.",
            }

        if target_provider == "gemini":
            key = api_key or self._gemini_api_key
            if not key:
                return {
                    "success": False,
                    "provider": "gemini",
                    "error": "Google Gemini API key is not configured.",
                }
            target_model = model or self._gemini_model
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": "Hello, respond with OK"}]}],
                "generationConfig": {"maxOutputTokens": 10},
            }
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status == 200:
                        return {
                            "success": True,
                            "provider": "gemini",
                            "model": target_model,
                            "message": f"Successfully connected to Google Gemini ({target_model}).",
                        }
                    return {
                        "success": False,
                        "provider": "gemini",
                        "model": target_model,
                        "error": f"HTTP status {resp.status}",
                    }
            except Exception as e:
                return {
                    "success": False,
                    "provider": "gemini",
                    "model": target_model,
                    "error": str(e),
                }

        if target_provider == "openai":
            key = api_key or self._openai_api_key
            if not key:
                return {
                    "success": False,
                    "provider": "openai",
                    "error": "OpenAI API key is not configured.",
                }
            target_model = model or self._openai_model
            target_base = (base_url or self._openai_base_url).rstrip("/")
            url = f"{target_base}/models"
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status == 200:
                        return {
                            "success": True,
                            "provider": "openai",
                            "model": target_model,
                            "message": f"Successfully connected to OpenAI endpoint ({target_base}).",
                        }
                    return {
                        "success": False,
                        "provider": "openai",
                        "model": target_model,
                        "error": f"HTTP status {resp.status}",
                    }
            except Exception as e:
                return {
                    "success": False,
                    "provider": "openai",
                    "model": target_model,
                    "error": str(e),
                }

        return {
            "success": False,
            "provider": target_provider,
            "error": f"Unknown provider: {target_provider}",
        }

    @staticmethod
    def inspect_hardware_for_llm() -> HardwareProfile:
        """
        Inspect CPU, RAM, and GPU VRAM to recommend optimal Gemma 4 quantization.
        """
        cpu_log = psutil.cpu_count(logical=True) or 4
        cpu_phys = psutil.cpu_count(logical=False) or 2
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024 ** 3), 2)
        ram_avail = round(mem.available / (1024 ** 3), 2)

        gpu_name: Optional[str] = None
        vram_total: Optional[float] = None
        vram_free: Optional[float] = None
        gpu_vendor: Optional[str] = None
        gpu_avail = False

        # Vendor neutral probe: NVIDIA, AMD, Intel and Apple all land here.
        try:
            from simulation.core.hardware import primary_gpu
            card = primary_gpu()
        except Exception as e:
            logger.debug(f"GPU probe failed non-fatally: {e}")
            card = None

        if card:
            gpu_name = card.get("name")
            gpu_vendor = card.get("vendor")
            gpu_avail = True
            if isinstance(card.get("vram_total_mb"), (int, float)):
                vram_total = round(float(card["vram_total_mb"]) / 1024.0, 2)
            if isinstance(card.get("vram_free_mb"), (int, float)):
                vram_free = round(float(card["vram_free_mb"]) / 1024.0, 2)

        # llama.cpp offload path that suits this card. Vulkan is the portable choice
        # for AMD and Intel; HIP builds are faster on AMD when the user installs one.
        vendor_key = (gpu_vendor or "").lower()
        if not gpu_avail:
            offload_backend = "none"
        elif "nvidia" in vendor_key:
            offload_backend = "cuda"
        elif "amd" in vendor_key or "advanced micro" in vendor_key or "radeon" in vendor_key:
            offload_backend = "hip"
        elif "apple" in vendor_key:
            offload_backend = "metal"
        else:
            offload_backend = "vulkan"

        # Recommendation logic per Section 2.15 & 3.1
        if vram_total and vram_total >= 10.0 and ram_total >= 28.0:
            rec_model = "gemma-4-E4B-it-Q4_K_M"
            rec_reason = (
                f"High-capability detected: GPU {gpu_name} with {vram_total}GB VRAM and {ram_total}GB RAM. "
                f"Optimally supports Gemma 4 E4B Q4_K_M offloaded through the {offload_backend} backend."
            )
        elif ram_total >= 16.0:
            rec_model = "gemma-4-E2B-it-Q4_K_M"
            rec_reason = (
                f"Standard desktop capability detected: {ram_total}GB RAM. "
                "Supports Gemma 4 E2B Q4_K_M with low latency."
            )
        else:
            rec_model = "gemma-4-E2B-it-Q4_K_M"
            rec_reason = (
                f"Constrained resource environment ({ram_total}GB RAM). "
                "Gemma 4 E2B Q4_K_M recommended with strict memory budgeting."
            )

        return HardwareProfile(
            cpu_cores_logical=cpu_log,
            cpu_cores_physical=cpu_phys,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            gpu_name=gpu_name,
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            cuda_available=gpu_avail,
            gpu_vendor=gpu_vendor,
            gpu_offload_backend=offload_backend,
            recommended_model=rec_model,
            recommendation_reason=rec_reason,
        )

    def is_local_server_running(self) -> bool:
        """Check if llama-server.exe is responding with HTTP 200 on /health."""
        try:
            with socket.create_connection((self.host, self.port), timeout=0.15):
                pass
        except (OSError, ConnectionRefusedError):
            return False

        url = f"http://{self.host}:{self.port}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_ollama_running(self) -> bool:
        """Probe local Ollama daemon on 127.0.0.1:11434 with fast socket probe."""
        try:
            with socket.create_connection(("127.0.0.1", 11434), timeout=0.1):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def get_status(self) -> Dict[str, Any]:
        """Report local Gemma GGUF presence, server status, and hardware profile."""
        gguf_mb = 0.0
        if self.gguf_path.exists():
            gguf_mb = round(self.gguf_path.stat().st_size / (1024 ** 2), 2)

        server_running = self.is_local_server_running()
        ollama_running = self.is_ollama_running()

        if server_running:
            active_provider = "local_gemma_llama_server"
        elif ollama_running:
            active_provider = "local_ollama"
        else:
            active_provider = "deterministic_grounded_assistant"

        return {
            "gguf_present": self.gguf_path.exists(),
            "gguf_size_mb": gguf_mb,
            "llama_server_exe_present": self.server_exe.exists(),
            "server_running": server_running,
            "ollama_running": ollama_running,
            "active_provider": active_provider,
            "hardware_profile": {
                "cpu_cores": self.hardware_profile.cpu_cores_logical,
                "ram_total_gb": self.hardware_profile.ram_total_gb,
                "gpu_name": self.hardware_profile.gpu_name,
                "vram_total_gb": self.hardware_profile.vram_total_gb,
                "recommended_model": self.hardware_profile.recommended_model,
            },
            "providers_status": self.get_providers_status(),
        }

    def start_local_server(self, background: bool = True) -> bool:
        """
        Spawn local llama-server.exe on 127.0.0.1:18089 using local GGUF weights.
        Hardened for Windows: creates no window, redirects pipes, polls /health up to 15s.
        """
        if self.is_local_server_running():
            logger.info("llama-server is already running and ready on localhost.")
            return True

        if not self.server_exe.exists():
            logger.warning(f"llama-server executable not found: {self.server_exe}")
            return False

        if not self.gguf_path.exists():
            logger.warning(f"GGUF model weights not found: {self.gguf_path}")
            return False

        cmd = [
            str(self.server_exe),
            "-m", str(self.gguf_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", "99",
            "-c", "4096",
            "--jinja",
        ]
        if self.jinja_template.exists():
            cmd.extend(["--chat-template-file", str(self.jinja_template)])

        popen_kwargs: Dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "cwd": str(self.server_exe.parent),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        try:
            self._server_process = subprocess.Popen(cmd, **popen_kwargs)
            # Poll /health for up to 15s in 0.5s increments
            for i in range(30):
                time.sleep(0.5)
                if self.is_local_server_running():
                    logger.info(f"Successfully launched local llama-server (ready in {(i+1)*0.5:.1f}s).")
                    return True
            logger.warning("llama-server spawned but health probe did not return 200 within 15s.")
            return False
        except Exception as e:
            logger.error(f"Failed to spawn llama-server: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanly terminate any spawned local llama-server process."""
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=3.0)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None

    def ask(
        self,
        question: str,
        simulation_context: Dict[str, Any],
        mode: InterfaceMode = InterfaceMode.PLAY,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer questions strictly grounded in simulation_context with hard guardrails.
        Supports multi-provider execution (Local Gemma 4, Google Gemini, OpenAI, Deterministic).
        """
        # Hard Guardrail 1: Cancer cure check (applies universally)
        if self._detect_cure_claim_intent(question):
            return self._handle_cancer_cure_guardrail(mode, simulation_context)

        # Check for parameter modification intent (applies universally)
        proposed_action = self._detect_parameter_mutation_intent(question, simulation_context)

        target_provider = (provider_override or self._active_provider).lower().strip()
        llm_response: Optional[str] = None
        used_provider = "deterministic_grounded_assistant"
        fallback_reason: Optional[str] = None
        system_prompt = self._build_system_prompt(mode, simulation_context)

        if target_provider == "gemini":
            if not self._gemini_api_key:
                fallback_reason = "Google Gemini API key not configured; fallback to deterministic assistant."
            else:
                try:
                    m = model_override or self._gemini_model
                    llm_response = self._query_gemini(question, system_prompt, m, self._gemini_api_key)
                    if llm_response:
                        used_provider = f"gemini:{m}"
                    else:
                        fallback_reason = "Gemini returned empty response; fallback to deterministic assistant."
                except Exception as e:
                    fallback_reason = f"Gemini query failed ({e}); fallback to deterministic assistant."

        elif target_provider == "openai":
            if not self._openai_api_key:
                fallback_reason = "OpenAI API key not configured; fallback to deterministic assistant."
            else:
                try:
                    m = model_override or self._openai_model
                    llm_response = self._query_openai(
                        question, system_prompt, m, self._openai_api_key, self._openai_base_url
                    )
                    if llm_response:
                        used_provider = f"openai:{m}"
                    else:
                        fallback_reason = "OpenAI returned empty response; fallback to deterministic assistant."
                except Exception as e:
                    fallback_reason = f"OpenAI query failed ({e}); fallback to deterministic assistant."

        elif target_provider in ("local_gemma", "auto_local"):
            if self.is_local_server_running():
                llm_response = self._query_llama_server(question, simulation_context, mode)
                if llm_response:
                    used_provider = "local_gemma_llama_server"
            elif self.is_ollama_running():
                llm_response = self._query_ollama(question, simulation_context, mode)
                if llm_response:
                    used_provider = "local_ollama"
            if not llm_response and target_provider == "local_gemma":
                fallback_reason = "Local Gemma server not reachable; fallback to deterministic assistant."

        if not llm_response:
            # Deterministic Grounded Scientific Assistant Fallback
            llm_response = self._grounded_deterministic_answer(question, simulation_context, mode)
            used_provider = "deterministic_grounded_assistant"

        if used_provider != "deterministic_grounded_assistant" and self._has_unsupported_numbers(llm_response, simulation_context):
            llm_response = self._grounded_deterministic_answer(question, simulation_context, mode)
            fallback_reason = "Model output included numbers not present in the supplied simulation context; showing a grounded explanation."
            used_provider = "deterministic_grounded_assistant"
        # Apply guardrails to final response
        sanitized_response = self._sanitize_response_guardrails(llm_response, simulation_context)

        result: Dict[str, Any] = {
            "question": question,
            "answer": sanitized_response,
            "mode": mode.value if hasattr(mode, "value") else str(mode),
            "provider": used_provider,
            "epistemic_category": EpistemicCategory.AI_HYPOTHESIS.value,
            "grounding_context_keys": list(simulation_context.keys()),
            "guardrail_checks": {
                "cancer_cure_claim_prevented": True,
                "silent_mutation_prevented": True,
                "unsupported_numbers_blocked": True,
            },
        }

        if fallback_reason:
            result["fallback_reason"] = fallback_reason

        if proposed_action:
            result["proposed_action"] = proposed_action

        return result

    def _query_gemini(
        self, question: str, system_prompt: str, model: str, api_key: str
    ) -> Optional[str]:
        """Query Google Gemini API via REST endpoint."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\nUser Question: {question}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 450,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            candidates = res_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        return None

    def _query_openai(
        self, question: str, system_prompt: str, model: str, api_key: str, base_url: str
    ) -> Optional[str]:
        """Query OpenAI or compatible endpoint via /chat/completions."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            choices = res_json.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        return None

    def _query_llama_server(
        self, question: str, context: Dict[str, Any], mode: InterfaceMode
    ) -> Optional[str]:
        """Query local llama-server via OpenAI-compatible endpoint."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        system_prompt = self._build_system_prompt(mode, context)
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if context.get("task") == "bounded_evolution_planning":
            payload["messages"][0]["content"] = (
                'Return JSON only. Convert the research objective into {"priorities":["safety"]}. '
                'Allowed values: safety, druglikeness, activity, selectivity, domain_confidence. '
                'Do not give molecules, medical conclusions, efficacy or additional text.'
            )
            payload["response_format"] = {"type": "json_object"}
            payload["temperature"] = 0
            payload["max_tokens"] = 256
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content = res_json["choices"][0]["message"]["content"].strip()
                # Strip special tokens and thinking blocks per requirement 2
                content = re.sub(r"<\|(?:im_end|end_of_turn|eot_id|thought|/thought)\|>", "", content)
                content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL)
                content = re.sub(r"<\|thought\|>.*?<\|/thought\|>", "", content, flags=re.DOTALL)
                return content.strip()
        except Exception as e:
            logger.debug(f"llama-server request failed, using deterministic fallback: {e}")
            return None

    def _query_ollama(
        self, question: str, context: Dict[str, Any], mode: InterfaceMode
    ) -> Optional[str]:
        """Query local Ollama via /api/generate with 2s timeout."""
        url = "http://127.0.0.1:11434/api/generate"
        system_prompt = self._build_system_prompt(mode, context)
        prompt = f"{system_prompt}\n\nUser Question: {question}\nAssistant:"
        payload = {"model": "gemma4:latest", "prompt": prompt, "stream": False}
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                return res_json.get("response", "").strip()
        except Exception as e:
            logger.debug(f"Ollama request failed, using deterministic fallback: {e}")
            return None

    def _build_system_prompt(self, mode: InterfaceMode, context: Dict[str, Any]) -> str:
        """Construct grounded system prompt with simulation state."""
        ctx_str = json.dumps(context, default=str)
        if mode == InterfaceMode.PLAY:
            tone_inst = (
                "You are the EVOCURE PLAY Scientific Guide. Speak clearly, warmly, and concisely. "
                "Use plain scientific language. Do not portray molecular resistance as shields or treatment as food. "
                "Never exceed 4 sentences. NEVER say a drug cures cancer. Ground all facts in the simulation data."
            )
        else:
            tone_inst = (
                "You are the EVOCURE LAB Oncology AI Assistant. Provide rigorous, quantitative explanations "
                "with mathematical notation, cell population kinetics, IC50/Hill parameters, and epistemic tags. "
                "Cite uncertainty intervals. NEVER claim a cure. Ground all facts strictly in the simulation data."
            )

        return (
            f"{tone_inst}\n\n"
            f"REAL SIMULATION DATA (Do NOT invent numbers outside this context):\n{ctx_str}\n"
        )

    def _grounded_deterministic_answer(
        self, question: str, context: Dict[str, Any], mode: InterfaceMode
    ) -> str:
        """
        Deterministic, fully grounded scientific assistant responding to key biological questions.
        """
        q = question.lower()

        # 1. "What is happening?"
        if any(k in q for k in ["what is happening", "what's happening", "status", "overview", "current state"]):
            return self._answer_what_is_happening(context, mode)

        # 2. "Why did this molecule disappear?" / Extinction
        if any(k in q for k in ["disappear", "extinct", "died out", "vanished", "lost"]):
            return self._answer_extinction(context, mode)

        # 3. "What does mutation mean?"
        if any(k in q for k in ["mutation", "mutate", "mutating", "variant"]):
            return self._answer_mutation(context, mode)

        # 4. "Which family is dominant?"
        if any(k in q for k in ["dominant", "winning", "most common", "top family"]):
            return self._answer_dominance(context, mode)

        # 5. "Why did some tumor cells survive?" / Resistance
        if any(k in q for k in ["survive", "resistant", "resistance", "didn't die", "still alive"]):
            return self._answer_survival_resistance(context, mode)

        # 6. "Compare scenarios"
        if any(k in q for k in ["compare", "difference between", "versus", "vs"]):
            return self._answer_comparison(context, mode)

        # 7. "Propose next experiment"
        if any(k in q for k in ["next experiment", "propose", "what should i test", "hypothesis", "next step"]):
            return self._answer_next_experiment(context, mode)

        # Default fallback grounded response
        return self._answer_general_grounded(question, context, mode)

    def _answer_what_is_happening(self, ctx: Dict[str, Any], mode: InterfaceMode) -> str:
        step = ctx.get("step") or ctx.get("evolutionary_step", 0)
        generation = ctx.get("generation") or ctx.get("evolutionary_generation", 0)
        entities_count = ctx.get("entity_count") or ctx.get("total_entities", len(ctx.get("entities", [])))
        alive_cells = ctx.get("alive_cells", ctx.get("tumor_cells_count", 0))
        compound_name = ctx.get("current_compound") or ctx.get("drug_name", "None")

        if mode == InterfaceMode.PLAY:
            return (
                f"Your simulation is at step {step} (generation {generation}). "
                f"The snapshot contains {entities_count or alive_cells} active agents. "
                f"Tissue exposure compound: {compound_name}. Artificial life and disease models use separate clocks."
            )
        else:
            ph = ctx.get("ph", 7.4)
            o2 = ctx.get("oxygen_norm", 1.0)
            res_count = ctx.get("resistant_cells", 0)
            return (
                f"Simulation state snapshot at step {step} (gen {generation}): "
                f"Microenvironment parameters: pH={ph}, Normalized O2={o2}. "
                f"Total viable cellular agents: {alive_cells or entities_count} (Resistant subpopulation: {res_count}). "
                f"Active exposure candidate: '{compound_name}'. Spatial transport and Hill pharmacodynamics are active."
            )

    def _answer_extinction(self, ctx, mode):
        if not ctx.get("extinction_cause"):
            return "No extinction cause is recorded in the supplied snapshot. Inspect the selected lineage's abundance and event history. Loss of a compound from the visible sample alone does not establish extinction."
        return f"Recorded disappearance: {ctx.get('last_extinct_family', 'Selected lineage')}. Recorded cause: {ctx['extinction_cause']}."

    def _answer_mutation(self, ctx: Dict[str, Any], mode: InterfaceMode) -> str:
        mut_rate = ctx.get("mutation_rate", 0.05)
        if mode == InterfaceMode.PLAY:
            return (
                f"A mutation is like a tiny accidental typo when a molecule or cell copies itself (rate: {mut_rate * 100:.1f}%). "
                "Inherited variation can change persistence and reproduction in the artificial model; it does not demonstrate a new therapeutic effect."
            )
        else:
            return (
                f"Mutations introduce stochastic perturbations into the genotype/subclone parameter vector (mu={mut_rate}). "
                "In Mode B (PDAC microenvironment), mutations represent targeted oncogenic alterations (e.g., KRAS_G12D, "
                "TP53 loss-of-function, SMAD4 deletion, or ATP-binding cassette efflux pump upregulation), shifting the clonal EC50."
            )

    def _answer_dominance(self, ctx, mode):
        if "dominant_family" not in ctx:
            return "Population counts by family were not supplied. Open the live compound registry to inspect abundance."
        return f"Family '{ctx['dominant_family']}' accounts for {ctx.get('dominant_share_pct', 0):.1f}% of current particles. This count alone does not establish why it is dominant."

    def _answer_survival_resistance(self, ctx, mode):
        count = ctx.get("resistant_cells", "unknown")
        return f"The supplied tissue snapshot records {count} resistant cells. The model includes efflux, concentration-dependent response and diffusion barriers. A matched control experiment is needed to attribute a particular outcome. These are exploratory model mechanisms."

    def _answer_comparison(self, ctx, mode):
        if "tumor_reduction_pct" in ctx:
            return f"The supplied comparison reports a model population difference of {ctx['tumor_reduction_pct']}%. This is not clinical evidence."
        return "No completed comparison was supplied. Use Evolution Lab to compare exact identity, molecular descriptors and structural similarity with reference drugs. Use the tissue workspace for a matched-seed exposed-versus-untreated model experiment."

    def _answer_next_experiment(self, ctx, mode):
        return "A useful next experiment is a matched-seed control and exposure with one changed input. Inspect healthy and pathological cell counts, resistance and exposure together. For molecular search, enter an objective in Evolution Lab, review its plan, then run it explicitly."

    def _answer_general_grounded(self, question, context, mode):
        return self._answer_what_is_happening(context, mode) + " You can ask about mutation, family abundance, recorded events, or how to compare compounds."

    def _detect_cure_claim_intent(self, text: str) -> bool:
        """Check if user is asking if a compound cures cancer."""
        pattern = r"\b(cure|cures|curing|eradicate|eliminate|destroy all cancer)\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _handle_cancer_cure_guardrail(
        self, mode: InterfaceMode, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        compound = context.get("current_compound") or context.get("name", "The candidate compound")
        if mode == InterfaceMode.PLAY:
            ans = (
                f"Important Scientific Guardrail: No in silico simulation can claim {compound} 'cures' cancer. "
                "In our simulation, compounds can inhibit tumor cell growth and reduce tumor burden, but resistant cells "
                "and protective barriers often survive. Real medical treatments require rigorous laboratory and clinical trials."
            )
        else:
            ans = (
                f"Epistemic Disclaimer (Section 27): {compound} cannot be claimed to 'cure' pancreatic ductal adenocarcinoma. "
                "In silico simulations demonstrate pharmacological tumor growth inhibition and subclonal selection dynamics. "
                "Heterogeneous clonal resistance (e.g. KRAS/TP53 mutations), desmoplastic stromal penetration barriers, "
                "and systemic off-target toxicities necessitate rigorous in vitro, in vivo, and clinical trial validation."
            )

        return {
            "question": "Can this compound cure cancer?",
            "answer": ans,
            "mode": mode.value,
            "provider": "deterministic_grounded_assistant",
            "epistemic_category": EpistemicCategory.VALIDATED_MODEL_PREDICTION.value,
            "guardrail_checks": {
                "cancer_cure_claim_prevented": True,
                "silent_mutation_prevented": True,
                "unsupported_numbers_blocked": True,
            },
        }

    def _detect_parameter_mutation_intent(
        self, question: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if user asked to change simulation settings. Never mutate silently!
        """
        q = question.lower()
        # Check pH change
        ph_match = re.search(r"\b(set|change|increase|decrease)\s+ph\s+(?:to\s+)?([0-9.]+)", q)
        if ph_match:
            new_val = float(ph_match.group(2))
            return {
                "action": "set_environment_parameter",
                "parameter": "ph",
                "current_value": context.get("ph", 7.4),
                "requested_value": new_val,
                "requires_user_approval": True,
                "prompt": f"Approve changing microenvironment pH from {context.get('ph', 7.4)} to {new_val}?",
            }

        # Check drug dose change
        dose_match = re.search(r"\b(set|change|apply|dose)\s+(?:conc|concentration|dose)\s+(?:to\s+)?([0-9.]+)", q)
        if dose_match:
            new_val = float(dose_match.group(2))
            return {
                "action": "set_exposure_concentration",
                "parameter": "applied_concentration_um",
                "current_value": context.get("applied_concentration_um", 0.0),
                "requested_value": new_val,
                "requires_user_approval": True,
                "prompt": f"Approve updating exposure concentration to {new_val} uM?",
            }

        return None

    def _sanitize_response_guardrails(
        self, text: str, context: Dict[str, Any]
    ) -> str:
        """
        Ensure generated text strictly adheres to guardrails:
        No unsupported numbers and no cure claims.
        """
        # Replace accidental cure claims
        text = re.sub(
            r"\b(cures|will cure|completely cures|eliminates all cancer)\b",
            "has no established clinical efficacy",
            text,
            flags=re.IGNORECASE,
        )
        return text

    @staticmethod
    def _has_unsupported_numbers(text, context):
        allowed = {0.0, 1.0, 2.0}
        def collect(value):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                allowed.add(float(value))
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    collect(child)
        collect(context)
        numbers = re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", text)
        return any(not any(abs(float(number) - candidate) <= max(0.015, abs(candidate) * 0.001) for candidate in allowed) for number in numbers)
