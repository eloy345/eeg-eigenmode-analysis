# Pipeline Workflow

This document describes the complete EEG eigenmode analysis pipeline, from raw data to extracted features.

---

## Overview

The pipeline consists of five main scripts executed sequentially:

```
Raw EEG → Preprocessing → Eigenmode Computation → Feature Extraction → Analysis
```

---

## Step 1: Preprocessing (`01_preprocess_eeg.py`)

### Input
- Cropped raw EEG recordings in `.fif` format
- Located in `data/EEG_crop/`
- Expected structure: organized by subject/condition

### Processing Steps

1. **Notch filtering** (50 Hz): Remove power line noise
2. **Bandpass filtering** (1-40 Hz): Remove slow drifts and high-frequency noise
3. **Bad channel interpolation**: Spherical spline interpolation of marked bad channels
4. **Common average reference (CAR)**: Rereferencing to the average of all channels

### Output
- Preprocessed EEG in `data/EEG_clean_v3_full/`
- Maintains original file structure
- Logs: `data/EEG_clean_v3_full/logs/`

### Notes
- Preprocessing parameters are configurable in `config.yaml`
- Order matters: Notch → Bandpass → Interpolation → CAR
- Files are saved in FIF format for consistency with MNE-Python

---

## Step 2: Eigenmode Computation (All Channels)

### Script: `02_compute_eigenmodes_all_channels.py`

Computes VAR(1) eigenmodes using **all available EEG channels**.

### Method

For each sliding window:

1. Extract EEG segment (1-16 seconds, configurable)
2. **Demean**: Subtract temporal mean from each channel
3. **Fit VAR(1) model**: X(t+1) = A·X(t) + ε
4. **Eigendecomposition**: Extract eigenvalues λ and eigenvectors v from A

The VAR(1) model captures temporal dependencies between channels:
- **Eigenvalues (λ ∈ ℂ)**: Characterize stability and oscillatory dynamics
- **Eigenvectors (v ∈ ℂⁿ)**: Describe spatial patterns of coherent activity

### Output Structure

```
results/eigenmodes/all_channels/03_eigs_npz/
└── win_1s/
    └── <group>/
        └── <subject_id>/
            └── <condition>/
                └── eigs_full.npz
```

Each `.npz` file contains:
- `evals`: Eigenvalues (W × N complex array)
- `evecs`: Eigenvectors (W × N × N complex array)
- Metadata: subject_id, group, condition, channel names, etc.

Where:
- W = number of windows
- N = number of channels

### Logs
- `results/eigenmodes/all_channels/logs/eigenmode_all_channels_run_log.csv`
- `results/eigenmodes/all_channels/logs/channel_reference_all_channels.csv`

---

## Step 3: Eigenmode Computation (No Occipital)

### Script: `03_compute_eigenmodes_no_occipital.py`

Identical to Step 2, but **excludes occipital channels** (O1, O2, Oz, POz) before computing eigenmodes.

### Purpose

**Sensitivity analysis**: Separate posterior/visual contributions during visual attention tasks (PVT condition).

### Output Structure

```
results/eigenmodes/no_occipital/03_eigs_npz/
└── win_1s/
    └── <group>/
        └── <subject_id>/
            └── <condition>/
                └── eigs_full.npz
```

Same format as all_channels, but:
- `drop_occipital: True` in metadata
- `occipital_channels_dropped: ["O1", "O2", "Oz", "POz"]`

---

## Step 4: Global Spectral Geometry Extraction

### Script: `04_extract_global_spectral_geometry.py`

Extracts features from **eigenvalues** (λ) characterizing the geometry of the eigenvalue distribution in the complex plane.

### Input
- `results/eigenmodes/all_channels/03_eigs_npz/`

### Feature Categories

1. **Radial features** (distance from origin)
   - Median, p95, std of |λ|
   - Signed distance: |λ| - 1
   - Proportion inside/outside unit circle

2. **Distance features** (proximity to unit circle)
   - Distance from unit circle: ||λ| - 1|
   - Proportion beyond thresholds (0.02, 0.05, 0.10)

3. **Oscillatory modes** (Im(λ) > 0.20)
   - Fraction of oscillatory modes
   - Radial, distance, and angular features for oscillatory subset

4. **Angular features**
   - Angle distribution in complex plane
   - Proportion beyond critical angles (35°, 45°)

5. **Ring features**
   - Angular distribution of near-critical modes (within 0.02, 0.05 of unit circle)

### Output

```
results/features/global_spectral_geometry/
├── 01_features/
│   ├── global_spectral_geometry_by_window.csv
│   ├── global_spectral_geometry_pooled.csv
│   ├── global_spectral_feature_columns_by_window.csv
│   └── global_spectral_feature_columns_pooled.csv
├── 07_delta_basal_to_pvt/
│   ├── global_spectral_delta_by_window.csv
│   └── global_spectral_delta_pooled.csv
└── 00_logs/
    └── global_spectral_extraction_log.csv
```

**by_window**: One row per subject/condition/window  
**pooled**: Median across windows per subject/condition  
**delta**: BASAL→PVT change per subject

---

## Step 5: Dominant-Mode Spatial Distribution Extraction

### Script: `05_extract_dominant_mode_spatial_distribution.py`

Extracts features from **eigenvectors** characterizing how dominant modes concentrate power across brain regions.

### Input
- `results/eigenmodes/all_channels/03_eigs_npz/`
- `results/eigenmodes/no_occipital/03_eigs_npz/`

### Method

For each window:

1. **Select dominant modes** (kind="all", "near0", or "osc")
2. **Compute participation vector**: How selected modes distribute across channels
   - p_i = (Σⱼ |v_ij|²) / (Σₖ Σⱼ |v_kj|²)
   - Where j indexes selected modes, i indexes channels
3. **Map to brain regions**:
   - **Core**: T7, T8, P7, P8 (temporoparietal)
   - **Frontal**: Fp1, Fpz, Fp2, AF3, AF4, F7, F3, Fz, F4, F8
   - **Central**: FC5, FC1, FC2, FC6, C3, Cz, C4, CP1, CP2, CP5, CP6
   - **Temporal**: T7, T8
   - **Parietal**: P7, P3, Pz, P4, P8

4. **Compute metrics**:
   - Regional masses (s_core, s_frontal, etc.)
   - Concentration: HHI, entropy, Gini coefficient, effective number of channels
   - Dominance: core vs other regions, core-to-rest ratio

### Output

```
results/features/dominant_mode_spatial_distribution/
├── 01_features/
│   ├── dominant_mode_spatial_distribution_by_window.csv
│   ├── dominant_mode_spatial_distribution_pooled.csv
│   └── dominant_mode_spatial_distribution_feature_columns.csv
├── 07_delta_basal_to_pvt/
│   ├── dominant_mode_spatial_distribution_delta_by_window.csv
│   └── dominant_mode_spatial_distribution_delta_pooled.csv
└── 00_logs/
    └── dominant_mode_spatial_distribution_extraction_log.csv
```

---

## Temporal Aggregation

### By-window tables
- One row per subject × condition × window size
- Captures temporal dynamics within recordings
- Use for: time-resolved analysis, tercile comparisons

### Pooled tables
- One row per subject × condition
- Median aggregation across all window sizes
- Use for: subject-level classification, group comparisons

### Delta tables
- One row per subject (BASAL→PVT change)
- Relative change: Δ = (PVT - BASAL) / (|BASAL| + ε)
- Use for: task-induced changes, reactivity analysis

---

## Data Flow Summary

```
EEG_crop/ (raw FIF)
    ↓ [01_preprocess_eeg.py]
EEG_clean_v3_full/ (preprocessed FIF)
    ↓ [02_compute_eigenmodes_all_channels.py]
    ↓ [03_compute_eigenmodes_no_occipital.py]
results/eigenmodes/{all_channels,no_occipital}/03_eigs_npz/ (NPZ files)
    ↓ [04_extract_global_spectral_geometry.py]
    ↓ [05_extract_dominant_mode_spatial_distribution.py]
results/features/{global_spectral_geometry,dominant_mode_spatial_distribution}/ (CSV)
    ↓ [Statistical analysis, ML, fusion - notebooks not included]
figures/ (final results)
```

---

## Execution Time Estimates

**Note**: Times vary significantly with dataset size and hardware.

| Step | Description | Approximate Time |
|------|-------------|------------------|
| 01 | Preprocessing | ~2-5 min per subject |
| 02-03 | Eigenmode computation | ~10-30 min per subject |
| 04-05 | Feature extraction | ~1-5 min total |

Total pipeline: ~30-60 minutes per subject on a modern workstation.

---

## Troubleshooting

### Common Issues

**"No NPZ files found"**
- Ensure eigenmodes were computed successfully
- Check paths in `config.yaml`
- Verify file structure matches expected format

**"No features extracted"**
- Check that conditions match expected values ("basal", "pvt")
- Verify window sizes are in expected list
- Check extraction logs for errors

**Memory errors during eigenmode computation**
- Reduce number of parallel processes
- Process subjects sequentially
- Use smaller window sizes

**Channel name mismatches**
- Ensure consistent channel naming (uppercase in code)
- Check that core/regional channels exist in your montage
- Verify occipital channel names match your system

### Debug Mode

To inspect intermediate outputs:
```python
# In scripts, add after loading data:
import pdb; pdb.set_trace()
```

Or add verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Next Steps

After running the pipeline:

1. **Verify outputs**: Check that all expected CSV files were generated
2. **Inspect logs**: Review extraction logs for warnings or errors
3. **Validate features**: Quick sanity checks (no all-NaN columns, expected ranges)
4. **Statistical analysis**: Load feature tables into your analysis framework
5. **Machine learning**: Use pooled or by-window tables for classification

See [`features.md`](features.md) for detailed feature descriptions.
