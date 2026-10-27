"""
Extensible Biomedical Disease Registry (Section 23).

Registers and instantiates fully active mechanistic disease models for:
1. PDAC (Pancreatic Ductal Adenocarcinoma)
2. NSCLC_KRAS (KRAS-Mutant Non-Small Cell Lung Adenocarcinoma)
3. AMR_BACTERIAL (Carbapenem-Resistant Bacterial Biofilm Infection)
4. METABOLIC_MASLD (Metabolic Dysfunction-Associated Steatohepatitis)
"""

from __future__ import annotations

from typing import Any, Type

from simulation.core.contracts import DiseaseModule
from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.diseases.nsclc import NSCLCModule
from simulation.biology.diseases.amr_bacterial import AMRBacterialModule
from simulation.biology.diseases.masld import MASLDModule


class DiseaseRegistry:
    """
    Central registry for biomedical disease models (Section 23).
    Allows listing, querying, registering, and instantiating disease models.
    """

    _registry: dict[str, Type[DiseaseModule]] = {}

    @classmethod
    def register(cls, module_cls: Type[DiseaseModule]) -> None:
        """Register a DiseaseModule implementation class."""
        temp_instance = module_cls()
        disease_id = temp_instance.disease_id
        cls._registry[disease_id] = module_cls

    @classmethod
    def get(cls, disease_id: str, **kwargs: Any) -> DiseaseModule:
        """Instantiate a registered disease module with optional keyword arguments."""
        if disease_id not in cls._registry:
            raise KeyError(
                f"Disease '{disease_id}' is not registered in DiseaseRegistry. "
                f"Available diseases: {list(cls._registry.keys())}"
            )
        module_cls = cls._registry[disease_id]
        return module_cls(**kwargs)

    @classmethod
    def create(cls, disease_id: str, seed: int = 42, **kwargs: Any) -> DiseaseModule:
        """
        Factory helper instantiating a registered disease module with seed.
        Ensures registry.create(disease_id, seed=...) instantiates any of the 4 cleanly.
        """
        return cls.get(disease_id, seed=seed, **kwargs)

    @classmethod
    def get_class(cls, disease_id: str) -> Type[DiseaseModule]:
        """Return the class for the registered disease module."""
        if disease_id not in cls._registry:
            raise KeyError(f"Disease '{disease_id}' is not registered.")
        return cls._registry[disease_id]

    @classmethod
    def is_registered(cls, disease_id: str) -> bool:
        """Check if a disease model is registered."""
        return disease_id in cls._registry

    @classmethod
    def list_available(cls) -> list[dict[str, Any]]:
        """Return a structured list of all registered disease models and their metadata."""
        items: list[dict[str, Any]] = []
        for did, mcls in cls._registry.items():
            inst = mcls()
            meta = inst.metadata()
            items.append(
                {
                    "disease_id": did,
                    "display_name": inst.display_name,
                    "implementation_status": meta.get(
                        "implementation_status", "fully_active_mechanistic"
                    ),
                    "category": meta.get("category", "oncology"),
                    "organ": meta.get("organ", "Unknown"),
                    "epistemic_category": meta.get("epistemic_category", "Mechanistic simulation result"),
                }
            )
        return items


# Register all 4 fully active mechanistic disease modules
DiseaseRegistry.register(PDACModule)
DiseaseRegistry.register(NSCLCModule)
DiseaseRegistry.register(AMRBacterialModule)
DiseaseRegistry.register(MASLDModule)
