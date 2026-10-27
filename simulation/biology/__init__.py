"""
EVOCURE Cancer Biology & Extensible Disease Simulation Package.

Includes:
- PDACModule: Mechanistic 2D Pancreatic Ductal Adenocarcinoma microenvironment.
- NSCLCModule: KRAS-Mutant Non-Small Cell Lung Adenocarcinoma microenvironment.
- AMRBacterialModule: Carbapenem-Resistant Bacterial Biofilm Infection microenvironment.
- MASLDModule: Metabolic Dysfunction-Associated Steatohepatitis microenvironment.
- DiseaseRegistry: Central registry for fully active mechanistic disease models.
- CoevolutionEngine: Coupled chemical-tumor coevolution with dual lineage trees.
"""

from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.diseases.nsclc import NSCLCModule
from simulation.biology.diseases.amr_bacterial import AMRBacterialModule
from simulation.biology.diseases.masld import MASLDModule
from simulation.biology.registry import DiseaseRegistry
from simulation.biology.coevolution import CoevolutionEngine

__all__ = [
    "DiseaseRegistry",
    "PDACModule",
    "NSCLCModule",
    "AMRBacterialModule",
    "MASLDModule",
    "CoevolutionEngine",
]
