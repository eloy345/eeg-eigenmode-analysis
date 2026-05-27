# Data Structure

This document describes the expected directory structure and file formats for the EEG eigenmode analysis pipeline.

---

## Overview

The pipeline expects a specific directory structure with:
1. Raw/cropped EEG recordings
2. Metadata CSV with subject information
3. Configuration file with paths

---

## Directory Structure

```
project_root/
├── config.yaml                 # Configuration (copy from config.example.yaml)
├── scripts/                    # Analysis scripts
├── data/                       # Input data (NOT in git)
│   ├── EEG_crop/              # Cropped raw EEG files
│   ├── EEG_clean_v3_full/     # Preprocessed EEG (generated)
│   └── meta.csv               # Subject metadata
└── results/                    # Output data (NOT in git)
    ├── eigenmodes/            # Computed eigenmodes
    └── features/              # Extracted features
```

---

## Input: Raw EEG Files

### Location
`data/EEG_crop/`

### Format
- **File type**: FIF format (`.fif`) - MNE-Python's native format
- **Alternative formats**: Can be converted from EDF, BDF, SET, etc. using MNE

### Expected Organization

The pipeline expects FIF files organized in a hierarchy:

```
data/EEG_crop/
├── aacc/                       # AACC group
│   ├── subject_01/
│   │   ├── basal.fif
│   │   └── pvt.fif
│   └── subject_02/
│       ├── basal.fif
│       └── pvt.fif
└── control/                    # Control group
    ├── subject_03/
    │   ├── basal.fif
    │   └── pvt.fif
    └── subject_04/
        ├── basal.fif
        └── pvt.fif
```

**Or** a flat structure where filename contains metadata:
```
data/EEG_crop/
├── aacc_01_basal.fif
├── aacc_01_pvt.fif
├── aacc_02_basal.fif
├── control_01_basal.fif
└── control_01_pvt.fif
```

### File Contents

Each FIF file should contain:
- **EEG channels**: Standard 10-20 montage (e.g., Fp1, F3, C3, P3, O1, etc.)
- **Sampling rate**: Typically 250-1000 Hz
- **Duration**: Cropped to relevant task period (e.g., 5-10 minutes)
- **Channel info**: Locations and types properly set in MNE

### Converting from Other Formats

If you have EDF/BDF/SET files:

```python
import mne

# Example: Convert EDF to FIF
raw = mne.io.read_raw_edf('input.edf', preload=True)
raw.save('output.fif', overwrite=True)
```

---

## Input: Metadata CSV

### Location
`data/meta.csv`

### Format

CSV file with subject-level metadata:

```csv
ID,y,AACC
subject_01,1,SI
subject_02,1,SI
subject_03,0,NO
subject_04,0,NO
```

### Required Columns

The scripts look for these column names (case-insensitive):

**Subject identifier**: One of
- `ID`
- `id`
- `subject_id`

**Group label (numeric)**: One of
- `y` (1 = AACC, 0 = Control)
- `group` ("aacc" or "control")
- `AACC` ("SI"/"SÍ" = AACC, "NO" = Control)
- `grupo`

### Example Configurations

**Minimal**:
```csv
ID,y
subject_01,1
subject_02,0
```

**Extended**:
```csv
ID,y,AACC,age,sex,condition_order
subject_01,1,SI,15,M,basal_first
subject_02,0,NO,14,F,pvt_first
```

The pipeline uses only `ID` and group columns; additional columns are ignored but can be useful for your own analysis.

### Notes
- Subject IDs in meta.csv should match filenames or directory names
- The pipeline attempts fuzzy matching if exact matches fail
- If a subject is in the data but not in meta.csv, it will be skipped with a warning

---

## Output: Preprocessed EEG

### Location
`data/EEG_clean_v3_full/`

### Structure

Mirrors input structure:

```
data/EEG_clean_v3_full/
├── aacc/
│   ├── subject_01/
│   │   ├── basal.fif
│   │   └── pvt.fif
│   └── ...
├── control/
│   └── ...
└── logs/
    └── preprocessing_YYYYMMDD_HHMMSS.log
```

Each preprocessed FIF file contains:
- Notch-filtered data (50/60 Hz removed)
- Bandpass-filtered data (1-40 Hz)
- Interpolated bad channels
- Common average reference applied

---

## Output: Eigenmodes (NPZ Files)

### Location
`results/eigenmodes/{all_channels,no_occipital}/03_eigs_npz/`

### Structure

```
results/eigenmodes/all_channels/03_eigs_npz/
├── win_1s/
│   ├── aacc/
│   │   ├── subject_01/
│   │   │   ├── basal/
│   │   │   │   └── eigs_full.npz
│   │   │   └── pvt/
│   │   │       └── eigs_full.npz
│   │   └── ...
│   └── control/
│       └── ...
├── win_2s/
│   └── ...
└── win_16s/
    └── ...
```

### NPZ File Contents

Each `eigs_full.npz` contains:

**Primary keys** (new format):
- `evals`: Eigenvalues (W × N complex array)
- `evecs`: Eigenvectors (W × N × N complex array)
- `id`: Subject ID
- `group`: "aacc" or "control"
- `cond`: "basal" or "pvt"
- `ch_names`: Channel names list
- `win_sec`: Window size in seconds
- `step_sec`: Step size in seconds
- `sfreq`: Sampling frequency

**Compatibility aliases**:
- `eigenvalues` → `evals`
- `eigenvectors` → `evecs`
- `subject_id` → `id`
- `condition` → `cond`
- `channel_names` → `ch_names`

**Additional metadata**:
- `source_fif_relpath`: Relative path to source FIF
- `pipeline_variant`: "all_channels" or "no_occipital"
- `drop_occipital`: Boolean
- `occipital_channels_dropped`: List of dropped channels
- `window_indices`: Window index array
- `n_channels`: Number of channels

### Dimensions

For a recording with:
- N = 30 channels
- 5 minutes at 250 Hz
- 1-second windows with 1-second step

Expect:
- `evals`: (300, 30) - 300 windows, 30 eigenvalues each
- `evecs`: (300, 30, 30) - 300 windows, 30×30 eigenvector matrix each

---

## Output: Feature Tables (CSV)

### Location
`results/features/{global_spectral_geometry,dominant_mode_spatial_distribution}/`

### Structure

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

### CSV Format

**by_window.csv**: One row per subject × condition × window size

```csv
id,group,y,cond,win_sec,tercile,all__rad_median,all__rad_p95,...
subject_01,aacc,1,basal,1.0,all,0.9823,1.0234,...
subject_01,aacc,1,pvt,1.0,all,0.9912,1.0156,...
subject_01,aacc,1,basal,2.0,all,0.9801,1.0267,...
```

**pooled.csv**: One row per subject × condition (median across windows)

```csv
id,group,y,cond,tercile,all__rad_median,all__rad_p95,...
subject_01,aacc,1,basal,all,0.9815,1.0245,...
subject_01,aacc,1,pvt,all,0.9905,1.0178,...
```

**delta.csv**: One row per subject (PVT - BASAL)

```csv
id,group,y,tercile,delta__all__rad_median,delta__all__rad_p95,...
subject_01,aacc,1,all,0.0092,-0.0067,...
```

### Feature Column Format

Feature names follow the pattern: `<subset>__<metric>_<statistic>`

Examples:
- `all__rad_median`: Median radius across all eigenvalues
- `osc__angle_p95_deg`: 95th percentile angle for oscillatory modes
- `s_core`: Mass in core temporoparietal region

See [`features.md`](features.md) for complete definitions.

---

## Disk Space Requirements

Approximate storage needs:

| Data Type | Size per Subject | Notes |
|-----------|------------------|-------|
| Raw EEG (10 min, 250 Hz) | ~30 MB | Uncompressed FIF |
| Preprocessed EEG | ~30 MB | Same size as raw |
| Eigenmodes (all windows) | ~100 MB | Compressed NPZ |
| Feature tables | <1 MB | CSV text |

For a study with 50 subjects:
- Raw + preprocessed: ~3 GB
- Eigenmodes: ~5 GB
- Features: <50 MB

**Total: ~8 GB** (excluding intermediate files and logs)

---

## File Format Notes

### Why FIF?
- Native format for MNE-Python
- Preserves all channel information and metadata
- Efficient for time-series data
- Can be converted to/from other formats

### Why NPZ?
- Efficient storage of NumPy arrays
- Compression reduces file size
- Fast loading
- Stores complex eigenvalues naturally

### Why CSV?
- Human-readable
- Compatible with R, MATLAB, Excel
- Easy to version control diffs (for small files)
- Standard format for ML pipelines

---

## Data Privacy Considerations

**Raw EEG files are not included in the public repository** because:
1. EEG data are from minors
2. Potentially personally identifiable
3. Subject to ethics committee restrictions

The directory structure and file format documentation allow others to:
- Use the pipeline with their own data
- Understand the expected inputs
- Replicate the methodology (not the exact results)

See [`reproducibility_notes.md`](reproducibility_notes.md) for more details.
