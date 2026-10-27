"""
EVOCURE Chemistry Simulation Module.

Provides chemical structure sanitization, physicochemical descriptor calculation,
Bemis-Murcko scaffold identification, 2D vector depiction, Tanimoto similarity,
elemental stoichiometry checking, reaction execution, and interactive 2D structure
editing for both Grounded Chemistry (Mode B) and Artificial Chemical Life (Mode A).
"""

from simulation.chemistry.library import COMPOUND_LIBRARY, get_builtin_library
from simulation.chemistry.engine import ChemistryEngine
from simulation.chemistry.editor import MolecularStructureEditor

__all__ = [
    "ChemistryEngine",
    "MolecularStructureEditor",
    "COMPOUND_LIBRARY",
    "get_builtin_library",
]
