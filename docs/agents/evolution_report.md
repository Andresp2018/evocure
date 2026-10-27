# EVOCURE Evolution Agent - Technical & Scientific Report

**Module:** `simulation/evolution`  
**Authors:** EVOCURE Evolution Agent (under Chief Orchestrator)  
**Status:** Completed & Validated (100% Level-3 Pytest Passing)  
**Epistemic Category:** `EpistemicCategory.MECHANISTIC_SIMULATION` & `EpistemicCategory.CALCULATED_PHYSICOCHEMICAL`  
**Governing Documents:** `docs/MATHEMATICAL_SPECIFICATION.md` (Section 4), `docs/AGENT_CONTRACTS.md`, `simulation/core/contracts.py`

---

## 1. Executive Summary

The EVOCURE Evolution module implements Darwinian evolutionary dynamics and multi-objective molecular optimization across two strictly separated modes of operation:
1. **Mode A (`ARTIFICIAL_LIFE`)**: Artificial chemical replicator ecosystem driven purely by emergent selection, spatial resource competition, metabolic maintenance, cooperative cross-feeding, parasitic interaction, and binary division with machine-precision thermodynamic/mass conservation.
2. **Mode B (`GROUNDED_CHEMISTRY`)**: Directed molecular optimization combining fast non-dominated sorting and crowding distance (NSGA-II) with 100% chemically sanitized RDKit mutation and recombination operators.
3. **Phylogeny & Lineage DAG (`GenealogyTracker`)**: Complete tracking of evolutionary origins, structural similarity, parent-child transitions, persistence time, and extinction logging without artificial fitness inflation.

The core evolutionary invariant is strictly upheld:
> **Zero Artificial Fitness Inflation**: Fitness is never artificially incremented by generation. Differential survival and reproductive output emerge strictly mechanistically from resource uptake, thermodynamic maintenance, and competitive exclusion.

---

## 2. Mathematical & Algorithmic Foundations

### 2.1. Emergent Selection & Metabolic Life Cycle (Mode A)

In Mode A, replicators (`SpatialEntity`) possess heritable traits:
- $\eta_{\text{cat}}$: Catalytic uptake and metabolic efficiency (`catalytic_efficiency`)
- $\sigma_{\text{stab}}$: Structural stability score (`stability_score` $\in [0.1, 1.0]$)
- $E_{\text{rep}}$: Binary division energy threshold (`replication_threshold`)
- $k_{\text{decay}}$: Basal maintenance decay rate (`decay_rate`)
- $\mathcal{R}_{\text{eco}}$: Ecological role (`autotroph`, `catalyst`, `cooperator`, `parasite`)

#### Resource Uptake and Biomass Storage
An entity at coordinate $\mathbf{x} = (x, y)$ consumes available local substrate $S(\mathbf{x})$:

$$u_i = \min\left(S(\mathbf{x}), k_{\text{uptake}} \cdot \eta_{\text{cat}, i} \cdot (1 + \beta_{\text{coop}}) \cdot \Delta t\right)$$

where $\beta_{\text{coop}}$ represents the additive cooperative boost from neighboring `cooperator` or `catalyst` replicators within interaction radius $R_{\text{coop}} = 25.0\,\mu\text{m}$.

#### Biologically Regulated Intracellular Metabolism
To prevent biomass depletion when energy is already sufficient for replication, catabolism is regulated by energy demand:

$$E_{\text{needed}} = \max\left(0, E_{\text{rep}, i} + E_{\text{buffer}} - E_i\right)$$

$$S_{\text{needed}} = \frac{E_{\text{needed}}}{Y_{\text{yield}} \cdot \sigma_{\text{stab}, i}}$$

$$m_i = \min\left(S_{\text{stored}, i}, k_{\text{metab}} \cdot \eta_{\text{cat}, i} \cdot \Delta t, S_{\text{needed}}\right)$$

$$S_{\text{stored}, i} \leftarrow S_{\text{stored}, i} - m_i, \qquad E_i \leftarrow E_i + m_i \cdot Y_{\text{yield}} \cdot \sigma_{\text{stab}, i}$$

#### Environmental Stress & Basal Maintenance
Deviations from the optimal physiological niche ($T_0 = 310.15\,\text{K}$, $\text{pH}_0 = 7.4$) increase basal maintenance, buffered by the entity's stability score $\sigma_{\text{stab}}$:

$$\Psi_{\text{stress}} = \left(\frac{|T - T_0|}{\tau_T} + \frac{|\text{pH} - \text{pH}_0|}{\tau_{\text{pH}}}\right) \cdot \max\left(0.1, 1.0 - 0.5 \sigma_{\text{stab}, i}\right)$$

$$E_i \leftarrow E_i - k_{\text{decay}, i} \cdot (1 + \Psi_{\text{stress}}) \cdot \Delta t$$

#### Parasitic Energy Draining & Colony Sharing
- **Parasitism**: If entity $p$ has role `parasite` and host $h$ is non-parasitic within $R_{\text{parasite}} = 20.0\,\mu\text{m}$:
  $$\Delta E_{\text{drain}} = \min\left(0.1 E_h, k_{\text{drain}} \cdot \Delta t\right)$$
  $$E_h \leftarrow E_h - \Delta E_{\text{drain}}, \qquad E_p \leftarrow E_p + \eta_{\text{drain}} \cdot \Delta E_{\text{drain}}$$
- **Colony Kin Sharing**: Kin sharing across non-null `colony_id` transfers energy from higher-energy to lower-energy kin:
  $$\Delta E_{\text{share}} = \frac{1}{2} k_{\text{colony}} (E_i - E_j) \cdot \Delta t$$

#### Lysis & Mass Recycling
When $E_i \le 0$, the entity undergoes lysis (`alive = False`), returning a fraction of its stored substrate and mass back to the local environment:

$$S_{\text{recycled}} = \gamma_{\text{rec}} \cdot S_{\text{stored}, i} + 0.05 \cdot M_i$$

#### Exact Thermodynamic & Mass Conservation at Binary Division
Division occurs if and only if $E_i \ge E_{\text{rep}, i}$ and $S_{\text{stored}, i} \ge 4.0$. Parent entity $P$ divides into daughters $C_1$ and $C_2$ with thermodynamic division overhead $E_{\text{overhead}}$:

$$E_{C_1} = E_{C_2} = \frac{1}{2} (E_P - E_{\text{overhead}})$$

$$S_{C_1} = S_{C_2} = \frac{1}{2} S_P, \qquad M_{C_1} = M_{C_2} = \frac{1}{2} M_P$$

Strictly satisfying machine-precision conservation:

$$E_{C_1} + E_{C_2} + E_{\text{overhead}} = E_P, \qquad S_{C_1} + S_{C_2} = S_P, \qquad M_{C_1} + M_{C_2} = M_P$$

---

### 2.2. Directed Molecular Optimization (Mode B / NSGA-II)

In Mode B, molecular candidates are optimized across multi-objective vectors $\mathbf{f}(x) = (f_1(x), f_2(x), \dots, f_m(x))$ using the Non-dominated Sorting Genetic Algorithm II (NSGA-II).

#### Pareto Dominance
Candidate $p$ dominates $q$ ($p \succ q$) if and only if:

$$\forall i \in \{1, \dots, m\}, \quad f_i(p) \ge f_i(q) \quad \text{and} \quad \exists j \in \{1, \dots, m\}, \quad f_j(p) > f_j(q)$$

Non-dominated sorting partitions the population into Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \dots$ in $O(M N^2)$ time.

#### Crowding Distance
Within each front $\mathcal{F}_k$, diversity is preserved via crowding distance $d_i$:

$$d_i = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$

Boundary solutions are assigned $d_i = \infty$.

#### RDKit Chemically Validated Genetic Operators
All structural modifications operate directly on RDKit `Mol` and `RWMol` objects, strictly enforcing `Chem.SanitizeMol`:
1. **Functional Group Attachment**: Appends `-F`, `-Cl`, `-OH`, `-NH2`, `-CH3`, `-OCH3`, `-CF3`, `-CN`, or `-C(=O)O` to atoms with accessible hydrogen valence ($H > 0$).
2. **Bioisosteric Substitution**: Aromatic $C \leftrightarrow N$, heteroatom $O \leftrightarrow S$ preserving ring aromaticity and octet rules.
3. **Substituent Swap**: Swaps terminal functional groups for alternative bioisosteres.
4. **Bond Modification**: Modifies acyclic single $\leftrightarrow$ double bonds where valences permit.
5. **BRICS & Single-Bond Crossover**: Cleaves non-ring single bonds via `Chem.FragmentOnBonds`, recombining chimeric fragments across dummy atom attachments with guaranteed sanitization.

#### Population Chemical Diversity
Quantified using mean pairwise Morgan fingerprint (radius 2, 2048-bit) Tanimoto distance:

$$\text{Diversity} = 1.0 - \frac{2}{N(N-1)} \sum_{i < j} \text{Tanimoto}(\mathbf{fp}_i, \mathbf{fp}_j)$$

---

## 3. Module Architecture & File Map

```
e:\MA\EVOCURE\simulation\evolution\
├── __init__.py          # Exposes GenealogyTracker, DarwinianEcosystemEngine, DirectedMolecularOptimizer
├── genealogy.py         # Lineage genealogy DAG, generational comparisons, tree queries, serialization
├── darwinian.py         # Mode A Artificial Chemical Life engine with 15 mechanistic dynamics
└── molecular_ga.py      # Mode B Directed Molecular Optimizer (NSGA-II + RDKit validated operators)

e:\MA\EVOCURE\tests\level3_evolution\
└── test_evolution.py   # 11 Level-3 scientific & architectural pytest validation tests
```

---

## 4. Empirical Test Validation (Level-3 Pytest Suite)

All 11 tests execute in **2.77 seconds** on the local Python 3.12 environment with 100% pass rate:

| Test ID | Test Name | Validation Principle | Status |
|---|---|---|---|
| 01 | `test_binary_division_conservation` | Exact machine-precision energy, substrate, and mass conservation ($10^{-9}$ tolerance) | **PASSED** |
| 02 | `test_heritability_without_mutation` | Strict inheritance of all kinetic traits when $\mu = 0$ | **PASSED** |
| 03 | `test_heritability_with_mutation` | Stochastic trait divergence within physiological bounds when $\mu = 1$ | **PASSED** |
| 04 | `test_differential_selection` | High-efficiency lineage outcompetes low-efficiency lineage under substrate limitation (1280 vs 40 entities) | **PASSED** |
| 05 | `test_extinction_and_cause_logging` | Proper transition to extinct state and cause logging upon population collapse | **PASSED** |
| 06 | `test_neutral_genetic_drift` | Stochastic drift without selective bias between functionally identical lineages | **PASSED** |
| 07 | `test_rdkit_chemical_validity_50_mutations_recombinations` | 100% chemical validity & `Chem.SanitizeMol` across 50 mutations and 50 recombinations | **PASSED** |
| 08 | `test_nsga2_pareto_sorting_and_crowding` | Correct identification of Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \mathcal{F}_2$ and crowding distances | **PASSED** |
| 09 | `test_genealogy_ancestor_tracing` | Complete lineage DAG traversal from target back to founder compounds | **PASSED** |
| 10 | `test_molecular_ga_population_evolution` | Multi-objective generation cycle with elitist survival and diversity scoring | **PASSED** |
| 11 | `test_genealogy_query_and_serialization` | Filtering by family/persistence/extinction and JSON dict serialization roundtrip | **PASSED** |

---

## 5. Epistemic Limitations & Assumptions

1. **Spatial Simplification (Mode A)**: Protocells are modeled as 2D spheres interacting via Brownian motion and spatial hashing. Intra-membrane structural mechanics are abstracted into `stability_score` and `radius`.
2. **Synthetic Valence Space (Mode B)**: Mutation operators operate on valid Lewis/Kekulé structures sanitized by RDKit; synthetic feasibility (SAscore) and stereochemical conformers are computational proxies, not wet-lab synthetic protocols.
3. **No Clinical Generalization**: Evolutionary fitness and Pareto dominance in silico represent exploratory computational design and do not guarantee in vivo pharmacokinetic absorption or human clinical efficacy.
