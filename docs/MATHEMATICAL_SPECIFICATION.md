# Mathematical Specification of the Engine - EVOCURE

This specification defines the governing equations, stoichiometric invariants, evolutionary operators, and pharmacological/oncological models implemented across `simulation/`.

---

## 1. Three Strictly Separated Clocks (Section 7.1)

The engine tracks three independent temporal coordinates:

1. **Computational Time ($t_{\text{comp}}$, in seconds)**: wall-clock CPU/GPU time measured via `time.perf_counter()`.
2. **Evolutionary Time ($s_{\text{evo}}, g_{\text{evo}}, y_{\text{virt}}$)**:
   $$g_{\text{evo}} = \left\lfloor \frac{s_{\text{evo}}}{N_{\text{steps/gen}}} \right\rfloor, \qquad y_{\text{virt}} = \frac{s_{\text{evo}}}{N_{\text{steps/gen}} \cdot G_{\text{year}}}$$
   where $y_{\text{virt}}$ is explicitly flagged as abstract artificial evolutionary time unless empirically calibrated.
3. **Biological Time ($t_{\text{bio}}$, in seconds, hours, and days)**: mechanistic step $\Delta t_{\text{bio}}$ advancing the cellular PDEs and systemic PK/PD ODEs.

Scientific trajectories depend solely on `numpy.random.PCG64(seed)` and $\Delta t$, ensuring **strict invariance to graphical frame rate (FPS)** or headless execution.

---

## 2. 2D Physics, Brownian Motion & Spatial Reaction-Diffusion (Section 4)

### 2.1. Overdamped Langevin / Stokes-Einstein Brownian Dynamics
In physical-unit mode (`use_physical_units = True`), translational diffusivity for an effective hydrodynamic radius $r_i$ at temperature $T$ (K) and dynamic viscosity $\eta$ ($\text{Pa}\cdot\text{s}$) is:

$$D_i = \frac{k_B T}{6 \pi \eta r_i}$$

Particle positions in $\Omega = [0, W] \times [0, H]$ under advection current $\mathbf{u}(\mathbf{x}, t)$ and short-range interaction forces $\mathbf{F}_i$ evolve via Euler-Maruyama integration:

$$\mathbf{x}_i(t + \Delta t) = \mathbf{x}_i(t) + \left(\mathbf{u}(\mathbf{x}_i, t) + \mu_i \mathbf{F}_i\right)\Delta t + \sqrt{2 D_{\text{eff}, i} \Delta t}\,\boldsymbol{\xi}_i, \qquad \boldsymbol{\xi}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$$

### 2.2. Semi-Permeable Barriers & $O(N)$ Spatial Hashing
- **Spatial Hash Grid**: partitions $\Omega$ into cells of side $\ell \ge 2 r_{\max}$, yielding average $O(N)$ collision and reaction-pair detection.
- **Semi-Permeable Barriers**: each barrier segment $B_k$ has permeability $p_k \in [0, 1]$; intersecting particles transmit with probability $p_k$ or undergo specular reflection with probability $1 - p_k$.

### 2.3. Eulerian Reaction-Diffusion PDEs (Substrates, $\text{O}_2$, Nutrients, Drug)
On an $N_x \times N_y$ grid, each continuous field $C(\mathbf{x}, t) \ge 0$ satisfies Fick's second law with source $S$, cellular/chemical sink $R$, and clearance $k_{\text{clear}}$:

$$\frac{\partial C}{\partial t} = D_C \nabla^2 C - \nabla \cdot (\mathbf{u} C) + S(\mathbf{x}, t) - R(C, \text{entities}) - k_{\text{clear}} C$$

integrated with explicit finite differences subject to the CFL stability bound ($\alpha = D_C \Delta t / \Delta x^2 \le 0.20$) and strict non-negativity projection $C \leftarrow \max(0, C)$.

---

## 3. Computational Chemistry, Stoichiometry & Reactions (Section 5)

### 3.1. Mode B (`GROUNDED_CHEMISTRY`)
Every species is sanitized via `RDKit` (`Chem.SanitizeMol`):
- **Atomic & Formal Charge Conservation**: an elementary reaction $\sum_a \nu_a A_a \longrightarrow \sum_b \mu_b B_b$ is accepted if and only if for every element $E$ and formal charge $Q$:
  $$\sum_a \nu_a N_E(A_a) = \sum_b \mu_b N_E(B_b), \qquad \sum_a \nu_a Q(A_a) = \sum_b \mu_b Q(B_b)$$
- **Temperature, pH & Catalytic Rate Law**:
  $$k(T, \text{pH}) = A_0 \exp\!\left(-\frac{E_a}{R T}\right) \cdot \exp\!\left(-\frac{(\text{pH} - \text{pH}_{\text{opt}})^2}{2 \sigma_{\text{pH}}^2}\right) \cdot (1 + \gamma_{\text{cat}} [\text{Cat}])$$

### 3.2. Mode A (`ARTIFICIAL_LIFE`)
Self-replicating chemical entities carry internal energy $E_i \ge 0$, substrate store $M_i \ge 0$, and heritable trait vectors. Total closed-box mass (grid substrate + internal entity mass + recycled detritus) is conserved up to floating-point precision whenever influx and clearance are zero ($S_{\text{in}} = k_{\text{out}} = 0$).

---

## 4. Darwinian Evolution & NSGA-II Multi-Objective Optimization (Section 6)

### 4.1. Emergent Ecological Evolution
Fitness is **never artificially incremented per generation**: differential persistence emerges mechanistically from resource competition ($E_i > E_{\text{min}}$) and binary division upon reaching $E_i \ge E_{\text{rep}}, M_i \ge M_{\text{rep}}$, splitting parent mass/energy across offspring ($E_1 + E_2 + E_{\text{cost}} = E_{\text{parent}}$) with stochastic mutation rate $\mu$.

### 4.2. Directed Molecular Evolution (NSGA-II + RDKit)
In Mode B, chemically valid candidates $x$ are optimized across the Pareto frontier of:

$$\mathbf{f}(x) = \Big(\underbrace{\hat{A}_{\text{PDAC}}(x)}_{\max}, \underbrace{\widehat{\text{SI}}(x)}_{\max}, \underbrace{\text{QED}(x)}_{\max}, \underbrace{-\hat{R}_{\text{tox}}(x)}_{\max}, \underbrace{-\hat{U}_{\text{AD}}(x)}_{\max}\Big)$$

where $\hat{U}_{\text{AD}}(x) = 1 - \max_{z \in \mathcal{D}_{\text{ref}}} \text{Tanimoto}(x, z)$ penalizes out-of-domain extrapolations using fast non-dominated sorting and crowding distance.

---

## 5. PDAC Tumor Microenvironment & Pharmacology/PBPK (Sections 8-14)

### 5.1. PDAC Subclonal & Desmoplastic Stroma Dynamics
Each tumor cell belongs to a subclone $c$ carrying explicit alteration profiles (e.g., sensitive `KRAS_G12D + TP53_LOF` vs. resistant `KRAS_G12D + TP53_LOF + CDKN2A_DEL + SMAD4_LOF + Efflux_High`).
- **Desmoplastic CAF Barrier**: local CAF density $\rho_{\text{CAF}}(\mathbf{x})$ attenuates effective drug diffusivity:
  $$D_{\text{drug, eff}}(\mathbf{x}) = \frac{D_{\text{drug}, 0}}{1 + \beta_{\text{ECM}} \rho_{\text{CAF}}(\mathbf{x})}$$
- **Applied vs. Free vs. Intracellular Exposure (Section 10)**:
  $$C_{\text{free}}(\mathbf{x}, t) = f_u \cdot C_{\text{total}}(\mathbf{x}, t), \qquad \frac{d C_{\text{intra}, i}}{dt} = k_{\text{perm}} C_{\text{free}}(\mathbf{x}_i, t) - (k_{\text{out}} + k_{\text{efflux}, i}) C_{\text{intra}, i}$$
- **Clone-Specific Hill Pharmacodynamics**:
  $$E_{\text{kill}, i}(t) = E_{\max, c} \frac{C_{\text{intra}, i}^h}{\text{EC}_{50, c}^h + C_{\text{intra}, i}^h}$$
  with Selectivity Index $\text{SI} = \text{EC}_{50, \text{healthy}} / \text{EC}_{50, \text{sens}}$.

### 5.2. 7-Compartment Systemic PBPK Module (Section 13)
Coupled ODEs across Blood, Liver, Kidneys, Bone Marrow, Pancreatic Tissue, Tumor, and Peripheral compartments:

$$V_{\text{blood}} \frac{dC_{\text{blood}}}{dt} = \sum_{k \neq \text{blood}} Q_k \left(\frac{C_k}{K_{p,k}} - C_{\text{blood}}\right) - \text{CL}_{\text{hep}} f_u C_{\text{blood}} - \text{CL}_{\text{ren}} f_u C_{\text{blood}} + I_{\text{dose}}(t)$$

$$V_k \frac{dC_k}{dt} = Q_k \left(C_{\text{blood}} - \frac{C_k}{K_{p,k}}\right)$$

When systemic PBPK is **disabled**, all systemic hematological/organ biomarkers are strictly tagged `EpistemicCategory.UNMODELED` (`"Not modeled"`). When **enabled**, they are tagged `EpistemicCategory.EXPLORATORY_MODEL` with explicit uncertainty intervals.
