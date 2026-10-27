"""
EVOCURE 2D Spatial Physics & Reaction-Diffusion PDE Solver.

This module implements:
1. SpatialHashGrid: O(N) spatial partitioning for neighbor and collision searches.
2. ReactionDiffusionGrid2D: Vectorized 2D NumPy Eulerian PDE solver for continuous
   concentration fields (substrate, oxygen, nutrient, drug, waste, energy_field)
   with automatic CFL sub-stepping and strict non-negativity.
3. PhysicsEngine2D: Langevin Brownian dynamics via Stokes-Einstein diffusivity,
   advection currents, tether springs, semi-permeable compartment barrier transmission/reflection,
   non-intentionality constraints for simple molecules, and elastic collision impulses.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from simulation.core.contracts import (
    CompartmentBarrier,
    EntityKind,
    EnvironmentConfig,
    SpatialEntity,
    SpatialRegion,
)


class SpatialHashGrid:
    """
    O(N) 2D Spatial Hash Grid for neighbor queries and collision detection.

    Partitions continuous 2D space into uniform square cells of size `cell_size`.
    Ensures O(1) expected cell query time and O(N) candidate collision pair detection.
    """

    def __init__(self, cell_size: float = 20.0) -> None:
        self.cell_size: float = max(1.0, float(cell_size))
        self.buckets: dict[tuple[int, int], list[SpatialEntity]] = defaultdict(list)

    def _hash(self, x: float, y: float) -> tuple[int, int]:
        """Map continuous coordinates (x, y) to discrete grid cell indices (cx, cy)."""
        return int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size))

    def clear(self) -> None:
        """Clear all entity references from the hash table."""
        self.buckets.clear()

    def insert(self, entity: SpatialEntity) -> None:
        """Insert an active spatial entity into its corresponding cell bucket."""
        cell = self._hash(entity.x, entity.y)
        self.buckets[cell].append(entity)

    def query_neighbors(self, x: float, y: float, radius: float) -> list[SpatialEntity]:
        """
        Query all entities within Euclidean distance `radius` of query point (x, y).

        Searches candidate bounding cells, filtering candidates by exact distance.
        """
        if radius <= 0.0:
            return []

        min_cx = int(math.floor((x - radius) / self.cell_size))
        max_cx = int(math.floor((x + radius) / self.cell_size))
        min_cy = int(math.floor((y - radius) / self.cell_size))
        max_cy = int(math.floor((y + radius) / self.cell_size))

        r_sq = radius * radius
        neighbors: list[SpatialEntity] = []

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = self.buckets.get((cx, cy))
                if not bucket:
                    continue
                for entity in bucket:
                    dx = entity.x - x
                    dy = entity.y - y
                    if (dx * dx + dy * dy) <= r_sq:
                        neighbors.append(entity)

        return neighbors

    def get_collision_pairs(
        self, entities: list[SpatialEntity]
    ) -> list[tuple[SpatialEntity, SpatialEntity]]:
        """
        Identify all colliding pairs of alive entities where dist <= e1.radius + e2.radius.

        Rebuilds spatial buckets and checks 9-neighbor cells per entity, returning
        unique unordered pairs (e1, e2) with average O(N) computational complexity.
        """
        self.clear()
        for e in entities:
            if e.alive:
                self.insert(e)

        pairs: list[tuple[SpatialEntity, SpatialEntity]] = []
        checked: set[tuple[str, str]] = set()

        for e1 in entities:
            if not e1.alive:
                continue

            cx, cy = self._hash(e1.x, e1.y)

            for dcx in (-1, 0, 1):
                for dcy in (-1, 0, 1):
                    bucket = self.buckets.get((cx + dcx, cy + dcy))
                    if not bucket:
                        continue

                    for e2 in bucket:
                        if not e2.alive or e1.entity_id >= e2.entity_id:
                            continue

                        pair_id = (e1.entity_id, e2.entity_id)
                        if pair_id in checked:
                            continue
                        checked.add(pair_id)

                        dx = e1.x - e2.x
                        dy = e1.y - e2.y
                        collision_dist = e1.radius + e2.radius

                        if (dx * dx + dy * dy) <= (collision_dist * collision_dist):
                            pairs.append((e1, e2))

        return pairs


class ReactionDiffusionGrid2D:
    """
    Vectorized 2D NumPy Eulerian PDE solver for continuous biochemical concentration fields.

    Governing equation for field C(x, y, t) >= 0:
        dC/dt = D * laplacian(C) - div(u * C) + S(x, y) - k_clear * C

    Key invariants:
    - Conservative finite-volume fluxes guarantee exact closed-box mass conservation.
    - Automatic CFL sub-stepping: if alpha = D * dt / min(dx, dy)^2 > 0.20, sub-steps automatically.
    - Strict non-negativity: np.maximum(field, 0.0) guarantees concentrations never drop below zero.
    - Zero-flux reflective Neumann boundary condition when boundary_permeability == 0.
    """

    def __init__(
        self,
        width: float = 1000.0,
        height: float = 700.0,
        nx: int = 50,
        ny: int = 35,
        default_fields: dict[str, float] | None = None,
    ) -> None:
        self.width: float = float(width)
        self.height: float = float(height)
        self.nx: int = int(nx)
        self.ny: int = int(ny)

        if self.nx < 2 or self.ny < 2:
            raise ValueError(f"Grid dimensions must be at least 2x2, got nx={nx}, ny={ny}")

        self.dx: float = self.width / self.nx
        self.dy: float = self.height / self.ny
        self.cell_area: float = self.dx * self.dy

        # Continuous biochemical fields
        self.standard_names = (
            "substrate",
            "oxygen",
            "nutrient",
            "drug",
            "waste",
            "energy_field",
        )
        self.fields: dict[str, np.ndarray] = {}

        for name in self.standard_names:
            self.fields[name] = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Baseline field concentrations
        self.fields["substrate"].fill(1.0)
        self.fields["oxygen"].fill(1.0)
        self.fields["nutrient"].fill(1.0)
        self.fields["energy_field"].fill(1.0)

        if default_fields:
            for k, v in default_fields.items():
                if k not in self.fields:
                    self.fields[k] = np.zeros((self.ny, self.nx), dtype=np.float64)
                self.fields[k].fill(float(v))

    def step_field(
        self,
        name: str,
        diff_coeff: float,
        dt: float,
        influx_rate: float = 0.0,
        clearance_rate: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        boundary_permeability: float = 0.0,
    ) -> None:
        """
        Advance concentration field forward in time by dt using conservative finite-volume integration.

        Guarantees CFL numerical stability by sub-stepping whenever:
            alpha = diff_coeff * dt / min(dx, dy)^2 > 0.20
        """
        if name not in self.fields:
            self.fields[name] = np.zeros((self.ny, self.nx), dtype=np.float64)

        field = self.fields[name]
        diff_coeff = max(0.0, float(diff_coeff))
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return

        min_spacing = min(self.dx, self.dy)
        min_spacing_sq = min_spacing * min_spacing

        # Determine sub-stepping count to strictly satisfy CFL limit (alpha <= 0.20)
        if diff_coeff > 0.0:
            alpha = (diff_coeff * dt) / min_spacing_sq
            if alpha > 0.20:
                n_substeps = int(math.ceil(alpha / 0.20))
            else:
                n_substeps = 1
        else:
            n_substeps = 1

        sub_dt = dt / n_substeps
        p_bound = max(0.0, min(1.0, float(boundary_permeability)))

        dx, dy = self.dx, self.dy
        nx, ny = self.nx, self.ny

        for _ in range(n_substeps):
            # --- 1. Conservative Diffusion Fluxes ---
            if diff_coeff > 0.0:
                # Horizontal interface fluxes (shape: (ny, nx + 1))
                fx = np.zeros((ny, nx + 1), dtype=np.float64)
                # Interior interface flux: F_{i+1/2} = -D * (C_{i+1} - C_i) / dx
                fx[:, 1:nx] = -diff_coeff * (field[:, 1:] - field[:, :-1]) / dx

                if p_bound > 0.0:
                    # Leaky boundary: outward leakage proportional to boundary permeability
                    fx[:, 0] = -p_bound * field[:, 0]
                    fx[:, nx] = p_bound * field[:, -1]
                # If p_bound == 0.0, fx[:, 0] = 0.0 and fx[:, nx] = 0.0 (Zero-flux Neumann)

                # Vertical interface fluxes (shape: (ny + 1, nx))
                fy = np.zeros((ny + 1, nx), dtype=np.float64)
                # Interior interface flux: F_{j+1/2} = -D * (C_{j+1} - C_j) / dy
                fy[1:ny, :] = -diff_coeff * (field[1:, :] - field[:-1, :]) / dy

                if p_bound > 0.0:
                    fy[0, :] = -p_bound * field[0, :]
                    fy[ny, :] = p_bound * field[-1, :]
                # If p_bound == 0.0, fy[0, :] = 0.0 and fy[ny, :] = 0.0 (Zero-flux Neumann)

                # Divergence of diffusion flux: - div(F) = - (dFx/dx + dFy/dy)
                diff_divergence = -((fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy)
                field += diff_divergence * sub_dt

            # --- 2. Advection Fluxes (1st-Order Upwind) ---
            if abs(vx) > 1e-12 or abs(vy) > 1e-12:
                fx_adv = np.zeros((ny, nx + 1), dtype=np.float64)
                if vx > 0.0:
                    fx_adv[:, 1:nx] = vx * field[:, :-1]
                    if p_bound > 0.0:
                        fx_adv[:, nx] = vx * field[:, -1]
                elif vx < 0.0:
                    fx_adv[:, 1:nx] = vx * field[:, 1:]
                    if p_bound > 0.0:
                        fx_adv[:, 0] = vx * field[:, 0]

                fy_adv = np.zeros((ny + 1, nx), dtype=np.float64)
                if vy > 0.0:
                    fy_adv[1:ny, :] = vy * field[:-1, :]
                    if p_bound > 0.0:
                        fy_adv[ny, :] = vy * field[-1, :]
                elif vy < 0.0:
                    fy_adv[1:ny, :] = vy * field[1:, :]
                    if p_bound > 0.0:
                        fy_adv[0, :] = vy * field[0, :]

                adv_divergence = -(
                    (fx_adv[:, 1:] - fx_adv[:, :-1]) / dx + (fy_adv[1:, :] - fy_adv[:-1, :]) / dy
                )
                field += adv_divergence * sub_dt

            # --- 3. Source Influx ---
            if influx_rate != 0.0:
                field += influx_rate * sub_dt

            # --- 4. First-Order Clearance / Decay ---
            if clearance_rate > 0.0:
                # Exponential decay guarantees concentration remains non-negative without truncation
                field *= math.exp(-clearance_rate * sub_dt)

            # --- 5. Strict Non-Negativity Projection ---
            np.maximum(field, 0.0, out=field)

    def sample(self, name: str, x: float, y: float) -> float:
        """
        Bilinearly interpolate continuous concentration of field `name` at spatial coordinate (x, y).

        Returns non-negative continuous concentration float.
        """
        if name not in self.fields:
            return 0.0

        field = self.fields[name]
        cx = max(0.0, min(self.width, float(x)))
        cy = max(0.0, min(self.height, float(y)))

        # Normalized coordinates relative to cell centers
        u = (cx / self.dx) - 0.5
        v = (cy / self.dy) - 0.5

        i0 = int(math.floor(u))
        j0 = int(math.floor(v))
        i1 = i0 + 1
        j1 = j0 + 1

        s = max(0.0, min(1.0, u - i0))
        t = max(0.0, min(1.0, v - j0))

        ci0 = max(0, min(self.nx - 1, i0))
        ci1 = max(0, min(self.nx - 1, i1))
        cj0 = max(0, min(self.ny - 1, j0))
        cj1 = max(0, min(self.ny - 1, j1))

        val = (
            (1.0 - s) * (1.0 - t) * field[cj0, ci0]
            + s * (1.0 - t) * field[cj0, ci1]
            + (1.0 - s) * t * field[cj1, ci0]
            + s * t * field[cj1, ci1]
        )
        return float(max(0.0, val))

    def get_gradient(self, name: str, x: float, y: float) -> tuple[float, float]:
        """Compute local continuous spatial gradient (dC/dx, dC/dy) via central finite differences."""
        eps_x = self.dx * 0.5
        eps_y = self.dy * 0.5
        cx_p = self.sample(name, x + eps_x, y)
        cx_m = self.sample(name, x - eps_x, y)
        cy_p = self.sample(name, x, y + eps_y)
        cy_m = self.sample(name, x, y - eps_y)

        gx = (cx_p - cx_m) / (2.0 * eps_x)
        gy = (cy_p - cy_m) / (2.0 * eps_y)
        return float(gx), float(gy)

    def consume_at(self, name: str, x: float, y: float, max_amount: float) -> float:
        """
        Consume mass from the grid cell containing (x, y) up to max_amount.

        Returns actual consumed mass >= 0.0 without ever allowing the cell concentration
        to drop below 0.0.
        """
        if max_amount <= 0.0 or name not in self.fields:
            return 0.0

        i = int(max(0, min(self.nx - 1, math.floor(x / self.dx))))
        j = int(max(0, min(self.ny - 1, math.floor(y / self.dy))))

        current_conc = self.fields[name][j, i]
        available_mass = current_conc * self.cell_area
        if available_mass <= 0.0:
            return 0.0

        consumed = min(float(max_amount), available_mass)
        new_conc = max(0.0, current_conc - (consumed / self.cell_area))
        self.fields[name][j, i] = new_conc
        return float(consumed)

    def deposit_at(self, name: str, x: float, y: float, amount: float) -> None:
        """Deposit mass into the grid cell containing (x, y)."""
        if amount <= 0.0 or name not in self.fields:
            return

        i = int(max(0, min(self.nx - 1, math.floor(x / self.dx))))
        j = int(max(0, min(self.ny - 1, math.floor(y / self.dy))))
        self.fields[name][j, i] += float(amount) / self.cell_area

    def total_mass(self, name: str) -> float:
        """Return the integrated total mass of field `name` across the entire domain: sum(C) * dx * dy."""
        if name not in self.fields:
            return 0.0
        return float(np.sum(self.fields[name]) * self.cell_area)

    def to_heatmap_list(self, name: str, downsample: int = 1) -> list[list[float]]:
        """Export 2D field concentration as a nested JSON-serializable list of floats."""
        if name not in self.fields:
            return []
        arr = self.fields[name]
        if downsample > 1:
            arr = arr[::downsample, ::downsample]
        return arr.tolist()


class PhysicsEngine2D:
    """
    2D Physics Engine handling Brownian motion, advection, tether springs,
    compartment barriers, non-intentionality for simple molecules, protocell chemotaxis,
    and elastic collision impulses.
    """

    BOLTZMANN_CONSTANT: float = 1.380649e-23  # J / K
    SPATIAL_DIFFUSIVITY_SCALE: float = 1.0e20  # Conversion scale factor mapping SI m^2/s to sim units^2/s

    @staticmethod
    def compute_brownian_diffusivity(
        radius_sim: float,
        temperature_k: float,
        viscosity_pa_s: float,
        use_physical_units: bool = True,
    ) -> float:
        """
        Calculate translational Brownian diffusivity D from the Stokes-Einstein equation:
            D = (k_B * T) / (6 * pi * eta * r)

        When use_physical_units is True:
            Evaluates SI Stokes-Einstein diffusivity and maps via documented spatial scale factor.
        When use_physical_units is False:
            Evaluates dimensionless baseline normalized to standard biological conditions.

        Both modes strictly satisfy physical monotonicity with T, 1/viscosity, and 1/radius.
        """
        r_eff = max(0.1, float(radius_sim))
        temp = max(0.0, float(temperature_k))
        visc = max(1e-7, float(viscosity_pa_s))

        if use_physical_units:
            d_val = (
                (PhysicsEngine2D.BOLTZMANN_CONSTANT * temp)
                / (6.0 * math.pi * visc * r_eff)
                * PhysicsEngine2D.SPATIAL_DIFFUSIVITY_SCALE
            )
            return float(d_val)
        else:
            # Dimensionless baseline: D0 = 5.0 at T0 = 310.15 K, eta0 = 0.00089 Pa*s, r0 = 5.0
            t0 = 310.15
            eta0 = 0.00089
            r0 = 5.0
            d0 = 5.0
            d_val = d0 * (temp / t0) * (eta0 / visc) * (r0 / r_eff)
            return float(d_val)

    @staticmethod
    def get_local_environment(x: float, y: float, env: EnvironmentConfig) -> dict[str, float]:
        """
        Evaluate local environmental parameters (temperature_k, ph, oxygen_norm, energy_flux)
        accounting for base EnvironmentConfig and any overlapping SpatialRegion instances.
        """
        t_k = float(env.temperature_k)
        ph = float(env.ph)
        o2 = float(env.oxygen_norm)
        flux = float(env.energy_source_flux)

        for r in env.regions:
            if r.x <= x <= (r.x + r.width) and r.y <= y <= (r.y + r.height):
                t_k += float(r.temperature_delta_k)
                if r.ph_override is not None:
                    ph = float(r.ph_override)
                o2 *= float(r.oxygen_multiplier)
                flux *= float(r.energy_flux)

        return {
            "temperature_k": float(max(0.0, t_k)),
            "ph": float(max(0.0, min(14.0, ph))),
            "oxygen_norm": float(max(0.0, o2)),
            "energy_flux": float(max(0.0, flux)),
        }

    @staticmethod
    def _intersect_barrier(
        p0: tuple[float, float],
        p1: tuple[float, float],
        barrier: CompartmentBarrier,
    ) -> tuple[bool, float, float]:
        """
        Test line segment intersection between motion vector p0->p1 and barrier segment.

        Returns (intersects, t_motion, u_barrier) where intersection point is p0 + t_motion * (p1 - p0).
        """
        rx = p1[0] - p0[0]
        ry = p1[1] - p0[1]
        sx = barrier.x2 - barrier.x1
        sy = barrier.y2 - barrier.y1

        denom = rx * sy - ry * sx
        if abs(denom) < 1e-12:
            return False, 0.0, 0.0

        qpx = barrier.x1 - p0[0]
        qpy = barrier.y1 - p0[1]

        t = (qpx * sy - qpy * sx) / denom
        u = (qpx * ry - qpy * rx) / denom

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True, float(t), float(u)
        return False, float(t), float(u)

    @classmethod
    def step_entities(
        cls,
        entities: list[SpatialEntity],
        env: EnvironmentConfig,
        grid: ReactionDiffusionGrid2D,
        dt: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """
        Advance spatial entities forward in time by dt:
        1. Brownian stochastic noise + advection current + reversible spring tether forces.
        2. Strict enforcement of Section 4.3 non-intentionality:
           Simple molecules (MOLECULE, NUTRIENT, CATALYST, THERAPEUTIC) never exhibit chemotaxis.
           ONLY PROTOCELL entities with catalytic_efficiency > 1.1 bias velocity along substrate
           gradient while paying an explicit energy cost.
        3. CompartmentBarrier line-segment intersection: stochastic transmission vs specular reflection.
        4. Domain boundary conditions: specular reflection or wrap-around.
        5. SpatialHashGrid O(N) pair detection and elastic impulse separation.
        """
        dt = max(0.0, float(dt))
        if dt <= 0.0 or not entities:
            return {
                "reactive_pairs": [],
                "collision_pairs": [],
                "collision_count": 0,
                "barrier_reflections": 0,
                "barrier_transmissions": 0,
                "mean_diffusivity": 0.0,
            }

        alive_entities = [e for e in entities if e.alive]
        entity_map = {e.entity_id: e for e in alive_entities}

        barrier_reflections = 0
        barrier_transmissions = 0
        diffusivities: list[float] = []

        # --- Phase 1: Kinematic & Force Integration ---
        for e in alive_entities:
            local_env = cls.get_local_environment(e.x, e.y, env)
            diff = cls.compute_brownian_diffusivity(
                e.radius,
                local_env["temperature_k"],
                env.viscosity_pa_s,
                env.use_physical_units,
            )
            diffusivities.append(diff)

            # Brownian displacement
            sigma = math.sqrt(2.0 * diff * dt)
            xi = rng.standard_normal(2)
            dx_brownian = sigma * float(xi[0])
            dy_brownian = sigma * float(xi[1])

            # Advection drift
            dx_adv = float(env.current_vx) * dt
            dy_adv = float(env.current_vy) * dt

            # Intrinsic velocity
            dx_vel = float(e.vx) * dt
            dy_vel = float(e.vy) * dt

            # Reversible tether spring force (if bound to a partner entity)
            dx_tether = 0.0
            dy_tether = 0.0
            if e.bound_to and e.bound_to in entity_map:
                partner = entity_map[e.bound_to]
                pdx = partner.x - e.x
                pdy = partner.y - e.y
                pdist = math.hypot(pdx, pdy)
                rest_len = e.radius + partner.radius

                if pdist > rest_len:
                    strain = pdist - rest_len
                    # Overdamped Langevin tether spring force: F = k_spring * strain
                    spring_k = 2.5
                    force_mag = spring_k * strain
                    fx = force_mag * (pdx / pdist)
                    fy = force_mag * (pdy / pdist)

                    # Effective hydrodynamic mobility: mu = 25.0 / mass
                    mobility = 25.0 / max(1.0, float(e.mass))
                    drift_vx = mobility * fx
                    drift_vy = mobility * fy

                    dx_tether = drift_vx * dt
                    dy_tether = drift_vy * dt
                    e.vx = (e.vx + drift_vx) * 0.85
                    e.vy = (e.vy + drift_vy) * 0.85

            # Section 4.3: Strict Chemotaxis Invariant
            dx_chemo = 0.0
            dy_chemo = 0.0
            if e.kind in (
                EntityKind.MOLECULE,
                EntityKind.NUTRIENT,
                EntityKind.CATALYST,
                EntityKind.THERAPEUTIC,
            ):
                # Simple molecules NEVER exhibit directed chemotaxis
                dx_chemo = 0.0
                dy_chemo = 0.0
            elif e.kind == EntityKind.PROTOCELL:
                # ONLY competent protocells (catalytic_efficiency > 1.1) with sufficient energy
                # bias their velocity weakly along local substrate gradient
                if e.catalytic_efficiency > 1.1:
                    chemo_cost = 0.05 * dt
                    if e.energy >= chemo_cost:
                        grad_x, grad_y = grid.get_gradient("substrate", e.x, e.y)
                        grad_mag = math.hypot(grad_x, grad_y)
                        if grad_mag > 1e-6:
                            chemo_speed = 3.0  # Weak directed drift speed
                            dx_chemo = chemo_speed * (grad_x / grad_mag) * dt
                            dy_chemo = chemo_speed * (grad_y / grad_mag) * dt
                            e.energy = max(0.0, e.energy - chemo_cost)

            # Total proposed displacement
            total_dx = dx_vel + dx_adv + dx_brownian + dx_tether + dx_chemo
            total_dy = dy_vel + dy_adv + dy_brownian + dy_tether + dy_chemo

            p0 = (e.x, e.y)
            p1 = (e.x + total_dx, e.y + total_dy)

            # --- Phase 2: CompartmentBarrier Line-Segment Crossing ---
            for barrier in env.barriers:
                intersects, t_hit, _ = cls._intersect_barrier(p0, p1, barrier)
                if intersects:
                    perm = max(0.0, min(1.0, float(barrier.permeability)))
                    roll = float(rng.random())
                    if roll < perm:
                        # Stochastic transmission
                        barrier_transmissions += 1
                    else:
                        # Specular reflection
                        barrier_reflections += 1
                        hit_x = p0[0] + t_hit * (p1[0] - p0[0])
                        hit_y = p0[1] + t_hit * (p1[1] - p0[1])

                        # Barrier normal
                        bx = barrier.x2 - barrier.x1
                        by = barrier.y2 - barrier.y1
                        blen = math.hypot(bx, by)
                        if blen > 1e-12:
                            nx = -by / blen
                            ny = bx / blen
                        else:
                            nx, ny = 1.0, 0.0

                        # Ensure normal opposes incoming motion
                        rx = p1[0] - p0[0]
                        ry = p1[1] - p0[1]
                        dot_rn = rx * nx + ry * ny
                        if dot_rn > 0.0:
                            nx, ny = -nx, -ny
                            dot_rn = -dot_rn

                        # Remaining path after collision
                        rem_x = (1.0 - t_hit) * rx
                        rem_y = (1.0 - t_hit) * ry
                        dot_rem = rem_x * nx + rem_y * ny
                        ref_rem_x = rem_x - 2.0 * dot_rem * nx
                        ref_rem_y = rem_y - 2.0 * dot_rem * ny

                        # Reflected position with safety epsilon away from barrier
                        p1 = (
                            hit_x + ref_rem_x + 1e-4 * nx,
                            hit_y + ref_rem_y + 1e-4 * ny,
                        )

                        # Specular velocity reflection
                        dot_v = e.vx * nx + e.vy * ny
                        e.vx = e.vx - 2.0 * dot_v * nx
                        e.vy = e.vy - 2.0 * dot_v * ny

            new_x, new_y = p1

            # --- Phase 3: Domain Boundary Handling ---
            if env.boundary_permeability <= 0.0:
                # Closed box with specular reflection
                if new_x - e.radius < 0.0:
                    new_x = e.radius + (e.radius - (new_x - e.radius))
                    e.vx = abs(e.vx)
                elif new_x + e.radius > env.width:
                    new_x = (env.width - e.radius) - ((new_x + e.radius) - env.width)
                    e.vx = -abs(e.vx)
                new_x = max(e.radius, min(env.width - e.radius, new_x))

                if new_y - e.radius < 0.0:
                    new_y = e.radius + (e.radius - (new_y - e.radius))
                    e.vy = abs(e.vy)
                elif new_y + e.radius > env.height:
                    new_y = (env.height - e.radius) - ((new_y + e.radius) - env.height)
                    e.vy = -abs(e.vy)
                new_y = max(e.radius, min(env.height - e.radius, new_y))
            else:
                # Open / leaky toroidal wrap-around
                if new_x < 0.0:
                    new_x += env.width
                elif new_x >= env.width:
                    new_x -= env.width

                if new_y < 0.0:
                    new_y += env.height
                elif new_y >= env.height:
                    new_y -= env.height

            e.x = new_x
            e.y = new_y

        # --- Phase 4: Spatial Hashing & Elastic Impulse Resolution ---
        hash_grid = SpatialHashGrid(cell_size=max(20.0, max((e.radius * 2.0 for e in alive_entities), default=20.0)))
        collision_pairs = hash_grid.get_collision_pairs(alive_entities)

        for e1, e2 in collision_pairs:
            dx = e1.x - e2.x
            dy = e1.y - e2.y
            dist = math.hypot(dx, dy)
            target_dist = e1.radius + e2.radius

            if dist < target_dist:
                overlap = target_dist - dist
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx / dist, dy / dist

                m1 = max(0.1, float(e1.mass))
                m2 = max(0.1, float(e2.mass))
                tot_m = m1 + m2
                w1 = m2 / tot_m
                w2 = m1 / tot_m

                # Positional separation to resolve overlap
                e1.x += nx * overlap * w1
                e1.y += ny * overlap * w1
                e2.x -= nx * overlap * w2
                e2.y -= ny * overlap * w2

                # Elastic impulse exchange
                rvx = e1.vx - e2.vx
                rvy = e1.vy - e2.vy
                v_norm = rvx * nx + rvy * ny

                if v_norm < 0.0:
                    restitution = 0.85
                    j = -(1.0 + restitution) * v_norm / (1.0 / m1 + 1.0 / m2)
                    imp_x = j * nx
                    imp_y = j * ny

                    e1.vx += imp_x / m1
                    e1.vy += imp_y / m1
                    e2.vx -= imp_x / m2
                    e2.vy -= imp_y / m2

        mean_diff = float(np.mean(diffusivities)) if diffusivities else 0.0

        return {
            "reactive_pairs": collision_pairs,
            "collision_pairs": collision_pairs,
            "collision_count": len(collision_pairs),
            "barrier_reflections": barrier_reflections,
            "barrier_transmissions": barrier_transmissions,
            "mean_diffusivity": mean_diff,
        }
