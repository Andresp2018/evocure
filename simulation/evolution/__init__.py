"""
EVOCURE Evolution Module.

Sub-modules:
- `genealogy`: Complete lineage phylogeny DAG tracking and extinction dynamics.
- `darwinian`: Mode A Artificial Chemical Life emergent evolutionary ecosystem.
- `molecular_ga`: Mode B Grounded Chemistry NSGA-II multi-objective optimizer with RDKit operators.
"""

from simulation.evolution.genealogy import GenealogyTracker
from simulation.evolution.darwinian import DarwinianEcosystemEngine
from simulation.evolution.molecular_ga import DirectedMolecularOptimizer

__all__ = [
    "GenealogyTracker",
    "DarwinianEcosystemEngine",
    "DirectedMolecularOptimizer",
]
