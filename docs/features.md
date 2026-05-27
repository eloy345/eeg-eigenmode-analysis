# Feature Definitions

This document provides detailed definitions of all features extracted by the pipeline.

---

## Overview

The pipeline extracts two complementary feature sets from VAR(1) eigenmode decomposition:

1. **Global spectral geometry**: Features from eigenvalues λ ∈ ℂ
2. **Dominant-mode spatial distribution**: Features from eigenvectors v ∈ ℂⁿ

Each includes outputs at multiple temporal aggregation levels:
- **by_window**: Per sliding window (captures temporal dynamics)
- **pooled**: Median across windows per subject/condition
- **delta**: BASAL → PVT relative change

---

## Global Spectral Geometry

Features characterizing the **geometry of eigenvalues** in the complex plane.

### Conceptual Background

VAR(1) eigenvalues describe:
- **|λ|** (radius): Distance from origin; controls oscillation amplitude
- **arg(λ)** (angle): Position in complex plane; oscillation frequency
- **|λ| relative to 1**: Proximity to unit circle; system stability

Critical region: Eigenvalues near the unit circle indicate marginal stability (high sensitivity to perturbations).

### Feature Categories

#### 1. Radial Features (Distance from Origin)

**all__rad_median** [0, ∞)
- Median of |λ| across all eigenvalues
- Interpretation: Typical oscillation amplitude

**all__rad_p95** [0, ∞)
- 95th percentile of |λ|
- Interpretation: Extreme oscillation amplitude

**all__rad_std** [0, ∞)
- Standard deviation of |λ|
- Interpretation: Variability in oscillation amplitudes

**all__rad_abs_mean**, **all__rad_abs_std**
- Mean and std of |λ|
- Interpretation: Central tendency and spread

#### 2. Signed Distance from Unit Circle

**all__rad_signed_median**, **all__rad_signed_mean**, **all__rad_signed_std** [-∞, ∞)
- Signed distance: |λ| - 1
- Negative: Inside unit circle (stable)
- Positive: Outside unit circle (unstable)
- Interpretation: Cumulative deviation from criticality

#### 3. Stability Features

**all__rad_prop_in** [0, 1]
- Proportion of eigenvalues with |λ| < 1 (stable)
- Interpretation: Fraction of damping modes

**all__rad_prop_out** [0, 1]
- Proportion of eigenvalues with |λ| > 1 (unstable)
- Interpretation: Fraction of growing modes
- **High values indicate instability**

#### 4. Distance from Unit Circle

**all__dist_uc_abs_mean**, **all__dist_uc_abs_std** [0, ∞)
- Mean and std of ||λ| - 1|
- Interpretation: Proximity to critical transitions

**all__dist_p95** [0, ∞)
- 95th percentile of ||λ| - 1|
- Interpretation: Extreme distances to criticality

**all__dist_prop_gt_0.02**, **all__dist_prop_gt_0.05**, **all__dist_prop_gt_0.10** [0, 1]
- Proportion of eigenvalues beyond threshold distance from unit circle
- Interpretation: Fraction of modes far from criticality
- High values = system far from critical region (robust)
- Low values = system near critical region (sensitive)

#### 5. Oscillatory Mode Fraction

**osc_frac** [0, 1]
- Proportion of eigenvalues with Im(λ) > 0.20
- Interpretation: Fraction of strongly oscillatory modes
- **Key indicator of oscillatory dynamics**

#### 6. Oscillatory Mode Features

All radial and distance features computed separately for oscillatory subset:

**osc__rad_median**, **osc__rad_p95**, **osc__rad_std**
- Radial features for Im(λ) > 0.20 only
- Interpretation: Oscillation amplitude in the oscillatory subset

**osc__dist_uc_abs_mean**, **osc__dist_p95**, etc.
- Distance features for oscillatory modes
- Interpretation: How close oscillatory modes are to unit circle

#### 7. Angular Features (Oscillatory Modes)

**osc__angle_p95_deg** [0°, 90°]
- 95th percentile of |arg(λ)| in degrees for oscillatory modes
- Interpretation: Extreme oscillation frequency

**osc__angle_prop_gt_35deg**, **osc__angle_prop_gt_45deg** [0, 1]
- Proportion of oscillatory modes with |arg(λ)| beyond threshold
- Interpretation: Fraction of rapid oscillations

**osc__ang_abs_mean_deg**, **osc__ang_abs_std_deg**
- Mean and std of absolute angle for oscillatory modes

**osc__imag_abs_mean**, **osc__imag_abs_std**
- Mean and std of Im(λ) for oscillatory modes
- Interpretation: Central oscillation frequency

#### 8. Ring Features

**osc__ring0.02__angle_p95_deg**, **osc__ring0.02__angle_prop_gt_35deg**, etc.

Features for modes in "critical rings":
- **ring0.02**: Oscillatory modes with ||λ| - 1| < 0.02
- **ring0.05**: Oscillatory modes with ||λ| - 1| < 0.05

Interpretation: Angular distribution of near-critical oscillatory modes (highest sensitivity region).

---

## Dominant-Mode Spatial Distribution

Features characterizing **how dominant eigenvectors distribute power** across brain regions.

### Conceptual Background

Eigenvectors v_i describe the **spatial pattern** of each eigenmode:
- High |v_ij|: Mode is strong in channel j
- Participation vector p_j = Σᵢ |v_ij|² / Σₖ Σᵢ |v_ki|²
- Interpretation: How selected modes concentrate across channels

### Brain Regions (10-20 System)

**Core (temporoparietal)**
- T7, T8, P7, P8
- Key region for attention and memory processing
- Critical for PVT (visual attention task)

**Frontal**
- Fp1, Fpz, Fp2, AF3, AF4, F7, F3, Fz, F4, F8
- Executive function, attention control

**Central**
- FC5, FC1, FC2, FC6, C3, Cz, C4, CP1, CP2, CP5, CP6
- Sensorimotor processing

**Temporal**
- T7, T8 (same as core temporal electrodes)
- Auditory processing, language

**Parietal**
- P7, P3, Pz, P4, P8
- Spatial processing, integration

### Feature Categories

#### 1. Regional Mass Features

**s_core**, **s_frontal**, **s_central**, **s_temporal**, **s_parietal** [0, 1]
- Participation mass in each region
- Sum to 1 across all channels
- Interpretation: Regional dominance of dominant modes
- **s_core**: Most discriminative for AACC vs controls in PVT

#### 2. Disjoint Regional Decomposition

**sD_core**, **sD_rest** [0, 1]
- Participation split: core vs all other channels
- sD_core + sD_rest = 1

**sD_frontal**, **sD_central**, **sD_parietal_nocore** [0, 1]
- Decomposition that excludes core overlap
- Interpretation: Non-core regional contributions

#### 3. Dominance Metrics

**logit_core** (-∞, ∞)
- Logit transformation: log(s_core / (1 - s_core))
- Interpretation: Odds-ratio scale for core dominance
- Useful for statistical modeling

**core_to_rest** [0, ∞)
- Ratio: s_core / (1 - s_core)
- Interpretation: Relative dominance of core vs rest of brain
- **High values = strong core focus**

**dom_gap_core_vs_best_other** [-1, 1]
- Difference: s_core - max(s_other_regions)
- Interpretation: Core advantage over best competing region
- Positive: Core is most dominant
- Negative: Other region more dominant

**dom_max_region** [0, 1]
- Maximum participation mass in any single region (non-core)
- Interpretation: Strongest competing region

#### 4. Concentration Metrics

**hhi** [1/N, 1] (Herfindahl-Hirschman Index)
- HHI = Σⱼ p_j²
- Range: 1/N (uniform) to 1 (complete concentration)
- Interpretation: Concentration of modes across channels
- **High HHI = modes localized to few channels**

**n_eff** [1, N] (Effective number of channels)
- n_eff = 1 / HHI
- Interpretation: Equivalent number of equally-participating channels
- **High n_eff = distributed across many channels**

**entropy_norm** [0, 1] (Normalized Shannon entropy)
- H_norm = -Σⱼ p_j log(p_j) / log(N)
- 0: Complete localization
- 1: Uniform distribution
- Interpretation: Spatial disorder or diversity of mode distribution

**gini** [0, 1] (Gini coefficient)
- Measure of inequality in distribution
- 0: Perfectly equal
- 1: Completely unequal
- Interpretation: Skewness of regional participation

**kurtosis_p** [1, ∞) (Excess kurtosis of participation)
- 4th moment of p distribution
- >3: Heavy-tailed (indicates outlier channels)
- Interpretation: Presence of dominant channels

#### 5. Derived Features

**max_p** [0, 1]
- Maximum participation value across all channels
- Interpretation: Strongest single-channel dominance

**top4_mass** [0, 1]
- Sum of participation for top 4 channels
- Interpretation: Concentration in lead channels

---

## No-Occipital Sensitivity Analysis

### Purpose

**Pipeline variant**: `03_compute_eigenmodes_no_occipital.py`

Recomputes all eigenmodes **excluding occipital channels** (O1, O2, Oz, POz) to:
1. Separate posterior/visual contributions
2. Test robustness of findings to channel removal
3. Control for potential PVT-specific visual processing confounds

### Output

Same feature structure as all_channels variant:
- `results/eigenmodes/no_occipital/03_eigs_npz/`
- Separate by_window and pooled feature tables
- Same feature names as all_channels

### Interpretation

**Comparison of all_channels vs no_occipital features**:
- Large differences: Features driven by posterior regions
- Small differences: Features independent of posterior activity
- Selective findings: Robust across channel configurations

Example:
- If s_core differs between variants: Core activity affected by occipital removal
- If s_core unchanged: Core activity independent of visual processing

---

## Feature Aggregation Levels

### by_window

- **Unit**: Single window (1-16 seconds)
- **Rows**: Subject × Group × Condition × Window_size × Window_index
- **Use**: Temporal resolution, tercile analysis, within-condition dynamics

### pooled

- **Unit**: Median across all windows for a subject/condition
- **Rows**: Subject × Group × Condition
- **Use**: Subject-level classification, group statistics

### delta

- **Unit**: BASAL → PVT relative change
- **Formula**: Δ_rel = (PVT - BASAL) / (|BASAL| + ε)
- **Rows**: Subject × Group (no condition)
- **Use**: Task-induced changes, reactivity profiles, sensitivity analysis

---

## Statistical Ranges

### Expected Ranges

| Feature | Min | Max | Typical | Notes |
|---------|-----|-----|---------|-------|
| all__rad_median | 0.7 | 1.0 | 0.95 | Mostly near unit circle |
| osc_frac | 0 | 1 | 0.3-0.5 | Depends on recording type |
| s_core | 0 | 1 | 0.1-0.3 | Core rarely dominant in all modes |
| hhi | 0.05 | 1 | 0.15-0.35 | Typically dispersed across regions |
| entropy_norm | 0 | 1 | 0.6-0.8 | Typically well-distributed |

### NaN Values

Features may be NaN if:
- No oscillatory modes detected (e.g., all ring features)
- Single-channel recordings
- Numerical instability
- Missing data in window

---

## Delta Interpretation

Relative delta: Δ_rel = (PVT - BASAL) / (|BASAL| + ε)

- **Positive delta**: Feature increases PVT → BASAL (task-induced increase)
- **Negative delta**: Feature decreases PVT → BASAL (task-induced decrease)
- **Zero delta**: No task effect

Example:
- Δ(s_core) = +0.15 → Core participation increases 15% during PVT
- Δ(osc_frac) = -0.20 → Oscillatory activity decreases 20% during PVT

---

## Feature Reproducibility

### High Reproducibility
- Radial features (all__rad_*)
- Regional mass features (s_*)
- Concentration metrics (hhi, entropy_norm)

### Moderate Reproducibility
- Distance features (depends on data quality)
- Angular features (depends on sampling rate)

### Low Reproducibility
- Ring features (sensitive to boundary conditions)
- Features derived from small mode sets

See [`reproducibility_notes.md`](reproducibility_notes.md) for pipeline stability considerations.
