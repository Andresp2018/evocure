"""
EVOCURE Disease Modules Package.

Provides 4 fully active 2D mechanistic disease engines:
1. PDACModule (Pancreatic Ductal Adenocarcinoma)
2. NSCLCModule (KRAS-Mutant Non-Small Cell Lung Adenocarcinoma)
3. AMRBacterialModule (Carbapenem-Resistant Bacterial Biofilm Infection)
4. MASLDModule (Metabolic Dysfunction-Associated Steatohepatitis)
"""

from simulation.biology.diseases.pdac import PDACModule
from simulation.biology.diseases.nsclc import NSCLCModule
from simulation.biology.diseases.amr_bacterial import AMRBacterialModule
from simulation.biology.diseases.masld import MASLDModule

__all__ = [
    "PDACModule",
    "NSCLCModule",
    "AMRBacterialModule",
    "MASLDModule",
]
