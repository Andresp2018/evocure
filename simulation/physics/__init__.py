"""
EVOCURE 2D Physics & Spatial Reaction-Diffusion Engine.

Provides O(N) spatial hashing, vectorized reaction-diffusion PDE integration
with CFL sub-stepping, Stokes-Einstein Langevin Brownian motion, and semi-permeable
compartment barriers.
"""

from simulation.physics.spatial import (
    PhysicsEngine2D,
    ReactionDiffusionGrid2D,
    SpatialHashGrid,
)

__all__ = [
    "SpatialHashGrid",
    "ReactionDiffusionGrid2D",
    "PhysicsEngine2D",
]
