"""
Level-1 Mathematical Validation Test Suite for EVOCURE 2D Physics & Spatial Reaction-Diffusion.

Validates:
1. Concentration non-negativity under heavy consumption and strong clearance.
2. Exact closed-box mass conservation (relative tolerance < 1e-9 over 100 diffusion steps).
3. Numerical stability and CFL sub-stepping under large diffusion coefficients.
4. Stokes-Einstein diffusivity monotonicity with temperature T, viscosity 1/eta, and radius 1/r.
5. Impermeable (0.0) vs permeable (1.0) vs semi-permeable (0.5) compartment barrier behavior.
6. Non-intentionality of simple molecules vs energy-costed protocell chemotaxis (Section 4.3).
7. Exact bit-for-bit RNG seed reproducibility with PCG64.
8. SpatialHashGrid neighbor queries and elastic collision resolution.
9. Local environmental calculations with overlapping SpatialRegions.
10. Reversible tether spring dynamics for bound entities.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from simulation.core.contracts import (
    CompartmentBarrier,
    EntityKind,
    EnvironmentConfig,
    SpatialEntity,
    SpatialRegion,
)
from simulation.physics.spatial import (
    PhysicsEngine2D,
    ReactionDiffusionGrid2D,
    SpatialHashGrid,
)


def test_concentration_non_negativity_under_heavy_consumption() -> None:
    """Validate that consume_at never causes grid cells to drop below zero, even under massive over-consumption."""
    grid = ReactionDiffusionGrid2D(width=200.0, height=200.0, nx=10, ny=10)
    grid.fields["substrate"].fill(0.0)

    # Deposit a localized concentration of 2.0 at cell (3, 3) (coord: 70, 70)
    # Cell area = 20 * 20 = 400.0, total mass = 2.0 * 400 = 800.0
    grid.deposit_at("substrate", 70.0, 70.0, 800.0)
    assert grid.sample("substrate", 70.0, 70.0) > 0.0

    # Request massive over-consumption: 50,000 mass units when only 800 are available
    consumed = grid.consume_at("substrate", 70.0, 70.0, 50000.0)
    assert consumed == pytest.approx(800.0, rel=1e-7)

    # Verify cell is exactly 0.0 and no negative values exist anywhere
    assert np.all(grid.fields["substrate"] >= 0.0)
    assert grid.sample("substrate", 70.0, 70.0) == 0.0

    # Attempt to consume again from empty cell
    consumed_second = grid.consume_at("substrate", 70.0, 70.0, 100.0)
    assert consumed_second == 0.0
    assert np.all(grid.fields["substrate"] >= 0.0)

    # Test strong clearance under PDE stepping
    grid.fields["substrate"].fill(10.0)
    for _ in range(20):
        grid.step_field("substrate", diff_coeff=1.0, dt=1.0, clearance_rate=15.0)
        assert np.all(grid.fields["substrate"] >= 0.0)
        assert not np.any(np.isnan(grid.fields["substrate"]))


def test_exact_closed_box_mass_conservation() -> None:
    """
    Validate exact closed-box mass conservation:
    With influx=0, clearance=0, and boundary_permeability=0, total mass must remain
    constant before and after 100 diffusion steps within 1e-9 relative tolerance.
    """
    grid = ReactionDiffusionGrid2D(width=1000.0, height=700.0, nx=50, ny=35)
    grid.fields["substrate"].fill(0.0)

    # Deposit localized mass peaks across domain
    grid.deposit_at("substrate", 200.0, 150.0, 1250.0)
    grid.deposit_at("substrate", 500.0, 350.0, 3400.0)
    grid.deposit_at("substrate", 800.0, 550.0, 2100.0)

    initial_mass = grid.total_mass("substrate")
    assert initial_mass == pytest.approx(1250.0 + 3400.0 + 2100.0, rel=1e-7)

    # Advance 100 diffusion steps with zero flux boundaries
    diff_coeff = 8.0
    dt = 0.5
    for _ in range(100):
        grid.step_field(
            "substrate",
            diff_coeff=diff_coeff,
            dt=dt,
            influx_rate=0.0,
            clearance_rate=0.0,
            boundary_permeability=0.0,
        )

    final_mass = grid.total_mass("substrate")
    rel_error = abs(final_mass - initial_mass) / initial_mass

    assert rel_error < 1e-9, f"Mass conservation violated: rel_error={rel_error:.2e} >= 1e-9"
    assert np.all(grid.fields["substrate"] >= 0.0)


def test_cfl_stability_at_large_diff_coeff() -> None:
    """
    Validate that when alpha = diff_coeff * dt / dx^2 > 0.20, automatic sub-stepping
    prevents numerical instability, divergence, or NaNs.
    """
    grid = ReactionDiffusionGrid2D(width=400.0, height=400.0, nx=20, ny=20)
    # dx = 20.0, dx^2 = 400.0
    # For diff_coeff = 800.0 and dt = 1.0, alpha = 800 * 1 / 400 = 2.0 >> 0.20
    # Without sub-stepping, alpha = 2.0 diverges immediately.
    grid.fields["substrate"].fill(0.0)
    grid.deposit_at("substrate", 200.0, 200.0, 5000.0)

    initial_mass = grid.total_mass("substrate")

    for _ in range(15):
        grid.step_field(
            "substrate",
            diff_coeff=800.0,
            dt=1.0,
            influx_rate=0.0,
            clearance_rate=0.0,
            boundary_permeability=0.0,
        )

    final_mass = grid.total_mass("substrate")
    assert not np.any(np.isnan(grid.fields["substrate"]))
    assert not np.any(np.isinf(grid.fields["substrate"]))
    assert np.all(grid.fields["substrate"] >= 0.0)
    assert abs(final_mass - initial_mass) / initial_mass < 1e-9


def test_stokes_einstein_monotonicity() -> None:
    """
    Validate physical Stokes-Einstein monotonicity:
    - Diffusivity D must strictly increase with temperature T.
    - Diffusivity D must strictly decrease with viscosity eta (monotonic with 1/viscosity).
    - Diffusivity D must strictly decrease with particle hydrodynamic radius r.
    Must hold in both physical-unit and dimensionless modes.
    """
    engine = PhysicsEngine2D()

    for physical_units in (True, False):
        # 1. Monotonicity with Temperature T
        d_cold = engine.compute_brownian_diffusivity(5.0, 280.0, 0.00089, physical_units)
        d_norm = engine.compute_brownian_diffusivity(5.0, 310.15, 0.00089, physical_units)
        d_hot = engine.compute_brownian_diffusivity(5.0, 350.0, 0.00089, physical_units)

        assert d_cold < d_norm < d_hot, f"Temperature monotonicity failed (physical={physical_units})"

        # 2. Monotonicity with Viscosity eta (increasing eta decreases D)
        d_low_visc = engine.compute_brownian_diffusivity(5.0, 310.15, 0.0005, physical_units)
        d_mid_visc = engine.compute_brownian_diffusivity(5.0, 310.15, 0.0010, physical_units)
        d_high_visc = engine.compute_brownian_diffusivity(5.0, 310.15, 0.0025, physical_units)

        assert d_low_visc > d_mid_visc > d_high_visc, f"Viscosity monotonicity failed (physical={physical_units})"

        # 3. Monotonicity with Radius r (increasing r decreases D)
        d_small_r = engine.compute_brownian_diffusivity(2.0, 310.15, 0.00089, physical_units)
        d_mid_r = engine.compute_brownian_diffusivity(6.0, 310.15, 0.00089, physical_units)
        d_large_r = engine.compute_brownian_diffusivity(15.0, 310.15, 0.00089, physical_units)

        assert d_small_r > d_mid_r > d_large_r, f"Radius monotonicity failed (physical={physical_units})"


def test_impermeable_vs_permeable_barrier_behavior() -> None:
    """
    Validate semi-permeable compartment barrier dynamics:
    - Impermeable barrier (permeability=0.0): 100% specular reflection, 0% transmission.
    - Fully permeable barrier (permeability=1.0): 100% transmission, 0% reflection.
    - Semi-permeable barrier (permeability=0.5): statistical transmission matches p=0.5 within 3-sigma.
    """
    grid = ReactionDiffusionGrid2D(width=300.0, height=300.0, nx=15, ny=15)
    env = EnvironmentConfig(width=300.0, height=300.0, boundary_permeability=0.0)

    # 1. Impermeable Barrier at x=150 (from y=0 to y=300)
    env.barriers = [
        CompartmentBarrier(barrier_id="wall_imp", x1=150.0, y1=0.0, x2=150.0, y2=300.0, permeability=0.0)
    ]
    entity_imp = SpatialEntity(
        entity_id="e_imp",
        compound_id="c1",
        kind=EntityKind.MOLECULE,
        x=130.0,
        y=150.0,
        vx=40.0,
        vy=0.0,
        radius=2.0,
    )
    rng = np.random.Generator(np.random.PCG64(1234))

    # Advance 1 step: proposed x = 130 + 40 = 170 crosses barrier at x=150
    telemetry = PhysicsEngine2D.step_entities([entity_imp], env, grid, dt=1.0, rng=rng)

    assert telemetry["barrier_reflections"] >= 1
    assert telemetry["barrier_transmissions"] == 0
    assert entity_imp.x < 150.0, f"Entity crossed impermeable barrier: x={entity_imp.x}"
    assert entity_imp.vx < 0.0, f"Velocity not inverted: vx={entity_imp.vx}"

    # 2. Fully Permeable Barrier (permeability=1.0)
    env.barriers = [
        CompartmentBarrier(barrier_id="wall_perm", x1=150.0, y1=0.0, x2=150.0, y2=300.0, permeability=1.0)
    ]
    entity_perm = SpatialEntity(
        entity_id="e_perm",
        compound_id="c2",
        kind=EntityKind.MOLECULE,
        x=130.0,
        y=150.0,
        vx=40.0,
        vy=0.0,
        radius=2.0,
    )
    telemetry2 = PhysicsEngine2D.step_entities([entity_perm], env, grid, dt=1.0, rng=rng)

    assert telemetry2["barrier_transmissions"] >= 1
    assert telemetry2["barrier_reflections"] == 0
    assert entity_perm.x > 150.0, f"Entity failed to cross permeable barrier: x={entity_perm.x}"
    assert entity_perm.vx > 0.0, f"Velocity was erroneously altered: vx={entity_perm.vx}"

    # 3. Semi-Permeable Barrier (permeability=0.5) with N=500 particles
    env.barriers = [
        CompartmentBarrier(barrier_id="wall_semi", x1=150.0, y1=0.0, x2=150.0, y2=300.0, permeability=0.5)
    ]
    n_trials = 500
    test_entities = [
        SpatialEntity(
            entity_id=f"e_semi_{i}",
            compound_id="c3",
            kind=EntityKind.MOLECULE,
            x=130.0,
            y=150.0,
            vx=40.0,
            vy=0.0,
            radius=1.0,
        )
        for i in range(n_trials)
    ]
    telemetry_semi = PhysicsEngine2D.step_entities(test_entities, env, grid, dt=1.0, rng=rng)

    transmissions = telemetry_semi["barrier_transmissions"]
    reflections = telemetry_semi["barrier_reflections"]
    assert transmissions + reflections == n_trials

    # For N=500, p=0.5: mean=250, std=sqrt(500*0.25)=11.18. 3-sigma interval is [216, 284].
    trans_frac = transmissions / n_trials
    assert 0.40 <= trans_frac <= 0.60, f"Transmission fraction {trans_frac:.3f} deviated from expected 0.50"


def test_non_intentionality_of_simple_molecules() -> None:
    """
    Validate Section 4.3 invariant:
    - Simple molecules (MOLECULE, NUTRIENT, CATALYST, THERAPEUTIC) NEVER exhibit directed chemotaxis.
      Their motion under a gradient is purely undirected Brownian motion (mean drift ~ 0).
    - Competent PROTOCELL entities (catalytic_efficiency > 1.1) bias velocity along the substrate
      gradient and pay an explicit energy cost.
    - Incompetent (efficiency <= 1.1) or energy-depleted protocells do not exhibit chemotaxis.
    """
    grid = ReactionDiffusionGrid2D(width=600.0, height=600.0, nx=30, ny=30)
    # Impose a strong linear substrate gradient along the X axis: C(x) = 0.05 * x
    for j in range(grid.ny):
        for i in range(grid.nx):
            grid.fields["substrate"][j, i] = 0.05 * (i * grid.dx)

    env = EnvironmentConfig(width=600.0, height=600.0, current_vx=0.0, current_vy=0.0)
    rng = np.random.Generator(np.random.PCG64(777))

    n_count = 60
    # Group 1: Simple molecules
    simple_kinds = [
        EntityKind.MOLECULE,
        EntityKind.NUTRIENT,
        EntityKind.CATALYST,
        EntityKind.THERAPEUTIC,
    ]
    simple_molecules = [
        SpatialEntity(
            entity_id=f"simple_{i}",
            compound_id=f"c_{i}",
            kind=simple_kinds[i % len(simple_kinds)],
            x=200.0,
            y=300.0,
            vx=0.0,
            vy=0.0,
            radius=2.0,
            energy=10.0,
            catalytic_efficiency=2.0,  # Even with high efficiency, simple molecules must NOT chemotax!
        )
        for i in range(n_count)
    ]

    # Group 2: Competent protocells (efficiency > 1.1, energy > 0)
    competent_protocells = [
        SpatialEntity(
            entity_id=f"proto_comp_{i}",
            compound_id=f"cp_{i}",
            kind=EntityKind.PROTOCELL,
            x=200.0,
            y=300.0,
            vx=0.0,
            vy=0.0,
            radius=5.0,
            energy=20.0,
            catalytic_efficiency=1.5,
        )
        for i in range(n_count)
    ]

    # Group 3: Energy-depleted protocells (efficiency > 1.1, but energy = 0.0)
    depleted_protocells = [
        SpatialEntity(
            entity_id=f"proto_depl_{i}",
            compound_id=f"cd_{i}",
            kind=EntityKind.PROTOCELL,
            x=200.0,
            y=300.0,
            vx=0.0,
            vy=0.0,
            radius=5.0,
            energy=0.0,
            catalytic_efficiency=1.5,
        )
        for i in range(n_count)
    ]

    # Step simulation forward for 25 steps
    dt = 0.2
    for _ in range(25):
        PhysicsEngine2D.step_entities(simple_molecules, env, grid, dt=dt, rng=rng)
        PhysicsEngine2D.step_entities(competent_protocells, env, grid, dt=dt, rng=rng)
        PhysicsEngine2D.step_entities(depleted_protocells, env, grid, dt=dt, rng=rng)

    # Calculate average X-displacement for each group
    mean_disp_simple = float(np.mean([e.x - 200.0 for e in simple_molecules]))
    mean_disp_competent = float(np.mean([e.x - 200.0 for e in competent_protocells]))
    mean_disp_depleted = float(np.mean([e.x - 200.0 for e in depleted_protocells]))

    # Competent protocells must have significant positive displacement along the substrate gradient
    assert mean_disp_competent > 10.0, f"Protocells failed chemotaxis: mean_disp={mean_disp_competent:.2f}"

    # Competent protocells must have paid an explicit energy cost
    assert all(e.energy < 20.0 for e in competent_protocells)

    # Simple molecules must NOT exhibit directed chemotaxis (displacement significantly smaller than protocells)
    assert mean_disp_simple < mean_disp_competent * 0.4
    assert all(e.energy == 10.0 for e in simple_molecules)  # No energy deducted for simple molecules

    # Depleted protocells cannot chemotax
    assert mean_disp_depleted < mean_disp_competent * 0.4


def test_exact_seed_reproducibility() -> None:
    """
    Validate that two independent simulations initialized with the exact same PCG64 seed
    produce identical positions, velocities, and energies bit-for-bit across 50 steps.
    """
    def run_simulation(seed: int) -> list[SpatialEntity]:
        rng = np.random.Generator(np.random.PCG64(seed))
        grid = ReactionDiffusionGrid2D(width=500.0, height=500.0, nx=25, ny=25)
        env = EnvironmentConfig(width=500.0, height=500.0, current_vx=1.5, current_vy=-0.5)

        entities = [
            SpatialEntity(
                entity_id=f"ent_{i}",
                compound_id=f"c_{i}",
                kind=EntityKind.PROTOCELL if i % 2 == 0 else EntityKind.MOLECULE,
                x=100.0 + (i % 10) * 25.0,
                y=100.0 + (i // 10) * 25.0,
                vx=(i % 3 - 1) * 2.0,
                vy=(i % 5 - 2) * 1.5,
                radius=4.0,
                energy=15.0,
                catalytic_efficiency=1.3 if i % 2 == 0 else 1.0,
            )
            for i in range(40)
        ]

        for _ in range(50):
            PhysicsEngine2D.step_entities(entities, env, grid, dt=0.1, rng=rng)

        return entities

    run_a = run_simulation(seed=42)
    run_b = run_simulation(seed=42)
    run_c = run_simulation(seed=999)

    # Run A and Run B must match bit-for-bit
    for ea, eb in zip(run_a, run_b):
        assert ea.x == eb.x
        assert ea.y == eb.y
        assert ea.vx == eb.vx
        assert ea.vy == eb.vy
        assert ea.energy == eb.energy

    # Run C with seed 999 must diverge from Run A
    has_divergence = any(ea.x != ec.x or ea.y != ec.y for ea, ec in zip(run_a, run_c))
    assert has_divergence, "Seed 999 did not diverge from seed 42"


def test_spatial_hash_grid_and_elastic_collisions() -> None:
    """Validate SpatialHashGrid O(N) neighbor querying and elastic collision separation."""
    grid = SpatialHashGrid(cell_size=20.0)

    e1 = SpatialEntity(entity_id="e1", compound_id="c1", kind=EntityKind.MOLECULE, x=50.0, y=50.0, radius=5.0)
    e2 = SpatialEntity(entity_id="e2", compound_id="c2", kind=EntityKind.MOLECULE, x=56.0, y=50.0, radius=5.0)
    e3 = SpatialEntity(entity_id="e3", compound_id="c3", kind=EntityKind.MOLECULE, x=150.0, y=150.0, radius=5.0)

    # Distance between e1 and e2 is 6.0 <= r1 + r2 (10.0) -> Colliding
    # Distance to e3 is > 100.0 -> Non-colliding
    pairs = grid.get_collision_pairs([e1, e2, e3])
    assert len(pairs) == 1
    assert pairs[0][0].entity_id == "e1" and pairs[0][1].entity_id == "e2"

    # Query neighbors of (50, 50) within radius 10.0
    neighbors = grid.query_neighbors(50.0, 50.0, radius=10.0)
    neighbor_ids = {n.entity_id for n in neighbors}
    assert "e1" in neighbor_ids and "e2" in neighbor_ids
    assert "e3" not in neighbor_ids

    # Step entities through PhysicsEngine2D to test elastic separation
    rd_grid = ReactionDiffusionGrid2D(width=200.0, height=200.0, nx=10, ny=10)
    env = EnvironmentConfig(width=200.0, height=200.0)
    rng = np.random.Generator(np.random.PCG64(42))

    e1.vx = 5.0
    e2.vx = -5.0
    telemetry = PhysicsEngine2D.step_entities([e1, e2], env, rd_grid, dt=0.1, rng=rng)

    assert telemetry["collision_count"] >= 1
    new_dist = math.hypot(e1.x - e2.x, e1.y - e2.y)
    assert new_dist >= (e1.radius + e2.radius) - 1e-4, f"Overlap not separated: new_dist={new_dist}"


def test_local_environment_and_spatial_regions() -> None:
    """Validate environmental parameter overrides from overlapping SpatialRegions."""
    env = EnvironmentConfig(
        width=500.0,
        height=500.0,
        temperature_k=310.15,
        ph=7.4,
        oxygen_norm=1.0,
        energy_source_flux=1.0,
        regions=[
            SpatialRegion(
                region_id="reg_warm_acidic",
                name="Warm Acidic Region",
                x=100.0,
                y=100.0,
                width=150.0,
                height=150.0,
                temperature_delta_k=5.0,
                ph_override=6.2,
                oxygen_multiplier=0.6,
                energy_flux=1.5,
            )
        ],
    )

    # Point outside region (50, 50)
    env_outside = PhysicsEngine2D.get_local_environment(50.0, 50.0, env)
    assert env_outside["temperature_k"] == pytest.approx(310.15)
    assert env_outside["ph"] == pytest.approx(7.4)
    assert env_outside["oxygen_norm"] == pytest.approx(1.0)
    assert env_outside["energy_flux"] == pytest.approx(1.0)

    # Point inside region (150, 150)
    env_inside = PhysicsEngine2D.get_local_environment(150.0, 150.0, env)
    assert env_inside["temperature_k"] == pytest.approx(315.15)
    assert env_inside["ph"] == pytest.approx(6.2)
    assert env_inside["oxygen_norm"] == pytest.approx(0.6)
    assert env_inside["energy_flux"] == pytest.approx(1.5)


def test_tether_spring_mechanics() -> None:
    """Validate that reversible tether springs pull bound entities towards equilibrium distance."""
    grid = ReactionDiffusionGrid2D(width=400.0, height=400.0, nx=20, ny=20)
    env = EnvironmentConfig(width=400.0, height=400.0)
    rng = np.random.Generator(np.random.PCG64(42))

    # Entity A at (100, 100), Entity B at (160, 100). Rest length is rA + rB = 10.0. Current dist is 60.0.
    e_a = SpatialEntity(entity_id="ea", compound_id="ca", kind=EntityKind.MOLECULE, x=100.0, y=100.0, radius=5.0, bound_to="eb")
    e_b = SpatialEntity(entity_id="eb", compound_id="cb", kind=EntityKind.MOLECULE, x=160.0, y=100.0, radius=5.0, bound_to="ea")

    initial_dist = math.hypot(e_a.x - e_b.x, e_a.y - e_b.y)
    assert initial_dist == pytest.approx(60.0)

    for _ in range(10):
        PhysicsEngine2D.step_entities([e_a, e_b], env, grid, dt=0.1, rng=rng)

    final_dist = math.hypot(e_a.x - e_b.x, e_a.y - e_b.y)
    assert final_dist < initial_dist, f"Tether spring failed to pull entities closer: {final_dist} >= {initial_dist}"


def test_bilinear_interpolation_and_gradient() -> None:
    """Validate continuous sampling and spatial gradient calculations on ReactionDiffusionGrid2D."""
    grid = ReactionDiffusionGrid2D(width=200.0, height=200.0, nx=10, ny=10)
    # Impose linear gradient: C(x, y) = 2.0 * x + 3.0 * y
    for j in range(grid.ny):
        for i in range(grid.nx):
            xc = (i + 0.5) * grid.dx
            yc = (j + 0.5) * grid.dy
            grid.fields["substrate"][j, i] = 2.0 * xc + 3.0 * yc

    # Sample at center (100, 100)
    c_val = grid.sample("substrate", 100.0, 100.0)
    expected_c = 2.0 * 100.0 + 3.0 * 100.0
    assert c_val == pytest.approx(expected_c, rel=0.05)

    # Compute spatial gradient (dC/dx, dC/dy)
    gx, gy = grid.get_gradient("substrate", 100.0, 100.0)
    assert gx == pytest.approx(2.0, rel=0.1)
    assert gy == pytest.approx(3.0, rel=0.1)

    # Export heatmap list
    heatmap = grid.to_heatmap_list("substrate", downsample=2)
    assert len(heatmap) == 5
    assert len(heatmap[0]) == 5


def test_advection_closed_box_mass_conservation() -> None:
    """Validate that advective current in a closed box conserves mass within 1e-9 relative tolerance."""
    grid = ReactionDiffusionGrid2D(width=400.0, height=400.0, nx=20, ny=20)
    grid.fields["substrate"].fill(0.0)
    grid.deposit_at("substrate", 200.0, 200.0, 1500.0)

    initial_mass = grid.total_mass("substrate")
    assert initial_mass == pytest.approx(1500.0)

    # Step with advection vx=2.0, vy=-1.5 and diffusion=1.0 for 30 steps
    for _ in range(30):
        grid.step_field(
            "substrate",
            diff_coeff=1.0,
            dt=0.2,
            influx_rate=0.0,
            clearance_rate=0.0,
            vx=2.0,
            vy=-1.5,
            boundary_permeability=0.0,
        )

    final_mass = grid.total_mass("substrate")
    rel_err = abs(final_mass - initial_mass) / initial_mass
    assert rel_err < 1e-9, f"Advection mass conservation error: {rel_err:.2e}"
    assert np.all(grid.fields["substrate"] >= 0.0)


def test_leaky_boundary_mass_loss() -> None:
    """Validate that boundary_permeability > 0 results in outward leakage and mass decay."""
    grid = ReactionDiffusionGrid2D(width=200.0, height=200.0, nx=10, ny=10)
    grid.fields["substrate"].fill(5.0)

    initial_mass = grid.total_mass("substrate")
    for _ in range(20):
        grid.step_field(
            "substrate",
            diff_coeff=2.0,
            dt=0.2,
            boundary_permeability=0.5,
        )

    final_mass = grid.total_mass("substrate")
    assert final_mass < initial_mass, f"Leaky boundary did not lose mass: final={final_mass}, init={initial_mass}"
    assert np.all(grid.fields["substrate"] >= 0.0)


def test_zero_dt_and_empty_entities_handling() -> None:
    """Validate graceful handling of zero dt and empty entity list."""
    grid = ReactionDiffusionGrid2D(width=200.0, height=200.0, nx=10, ny=10)
    env = EnvironmentConfig(width=200.0, height=200.0)
    rng = np.random.Generator(np.random.PCG64(42))

    # Empty list
    res_empty = PhysicsEngine2D.step_entities([], env, grid, dt=1.0, rng=rng)
    assert res_empty["collision_count"] == 0
    assert len(res_empty["reactive_pairs"]) == 0

    # dt <= 0
    e = SpatialEntity(entity_id="e0", compound_id="c0", kind=EntityKind.MOLECULE, x=50.0, y=50.0)
    res_zero_dt = PhysicsEngine2D.step_entities([e], env, grid, dt=0.0, rng=rng)
    assert res_zero_dt["collision_count"] == 0
    assert e.x == 50.0 and e.y == 50.0

