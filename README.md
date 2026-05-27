# EEG Eigenmode Analysis Pipeline

**Code repository accompanying:** [Paper title and citation to be added]

This repository documents the EEG preprocessing and feature extraction pipeline used to analyze neural dynamics in intellectually gifted children compared to controls.

---

## ⚠️ Important Note on Data Availability

**The raw EEG data used in this study cannot be publicly shared** due to ethical and privacy restrictions (study involves minors). This repository provides:

- ✅ Complete preprocessing and analysis code
- ✅ Pipeline documentation and methodology
- ✅ Feature extraction algorithms
- ❌ Raw EEG recordings
- ❌ Preprocessed datasets
- ❌ Computed features

The purpose of this repository is **methodological transparency** and **pipeline reusability**, not full computational reproducibility of the published results without access to the original protected dataset.

---

## Overview

The pipeline analyzes EEG dynamics using VAR(1) eigenmode decomposition to extract two complementary feature sets:

1. **Global spectral geometry**: Features derived from VAR(1) eigenvalues characterizing system stability, oscillatory dynamics, and proximity to critical transitions
2. **Dominant-mode spatial distribution**: Features derived from eigenvectors characterizing how neural activity concentrates across brain regions

## Next Steps

After running the pipeline:

1. **Verify outputs**: Check that all expected CSV files were generated
2. **Inspect logs**: Review extraction logs for warnings or errors
3. **Validate features**: Quick sanity checks (no all-NaN columns, expected ranges)
4. **Statistical analysis**: Load feature tables into your analysis framework
5. **Machine learning**: Use pooled or by-window tables for classification or statistical modeling

See [`docs/features.md`](docs/features.md) for detailed feature descriptions.

---

## Pipeline Structure

```
EEG_crop/                    # Cropped raw EEG recordings
    ↓
01_preprocess_eeg.py         # Preprocessing (filtering, interpolation, CAR)
    ↓
EEG_clean_v3_full/           # Preprocessed EEG
    ↓
02_compute_eigenmodes_all_channels.py      # VAR(1) eigenmodes (all channels)
03_compute_eigenmodes_no_occipital.py      # VAR(1) eigenmodes (no occipital)
    ↓
results/eigenmodes/          # Eigenvalues and eigenvectors
    ↓
04_extract_global_spectral_geometry.py     # Eigenvalue-based features
05_extract_dominant_mode_spatial_distribution.py  # Eigenvector-based features
    ↓
results/features/            # Extracted feature tables
    ↓
statistics / ML / fusion     # Analysis notebooks (to be provided)
```

For detailed workflow, see [`docs/workflow.md`](docs/workflow.md).

---

## Repository Contents

```
├── scripts/
│   ├── 01_preprocess_eeg.py
│   ├── 02_compute_eigenmodes_all_channels.py
│   ├── 03_compute_eigenmodes_no_occipital.py
│   ├── 04_extract_global_spectral_geometry.py
│   └── 05_extract_dominant_mode_spatial_distribution.py
├── docs/
│   ├── workflow.md                  # Detailed pipeline description
│   ├── data_structure.md            # Expected data organization
│   ├── features.md                  # Feature definitions
│   └── reproducibility_notes.md     # Limitations and notes
├── data_example/
│   ├── README.md
│   └── meta.example.csv             # Metadata format example
├── config.example.yaml              # Configuration template
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.8+
- MNE-Python (EEG processing)
- NumPy, Pandas, scikit-learn
- PyYAML

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/eeg-eigenmode-analysis.git
cd eeg-eigenmode-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure
cp config.example.yaml config.yaml
# Edit config.yaml with your paths
```

---

## Usage

### 1. Prepare Your Data

Organize your EEG data according to the structure described in [`docs/data_structure.md`](docs/data_structure.md).

Create a metadata CSV with subject information:
```csv
ID,y,AACC
subject_01,1,SI
subject_02,0,NO
```

### 2. Configure Pipeline

Edit `config.yaml` with your paths:
```yaml
base_dir: "."
crop_dir: "data/EEG_crop"
clean_dir: "data/EEG_clean_v3_full"
meta_csv: "data/meta.csv"
```

### 3. Run Pipeline

```bash
# Step 1: Preprocess EEG
python scripts/01_preprocess_eeg.py

# Step 2: Compute eigenmodes
python scripts/02_compute_eigenmodes_all_channels.py
python scripts/03_compute_eigenmodes_no_occipital.py

# Step 3: Extract features
python scripts/04_extract_global_spectral_geometry.py
python scripts/05_extract_dominant_mode_spatial_distribution.py
```

Each script generates logs and outputs in the configured directories.

---

## Features

### Global Spectral Geometry

Features from VAR(1) eigenvalues (λ ∈ ℂ):

- **Radial**: Distance from origin, signed distance from unit circle
- **Stability**: Proportion inside/outside unit circle
- **Oscillatory**: Fraction of modes with Im(λ) > 0.20, angular features
- **Ring features**: Angular distribution of near-critical modes

See [`docs/features.md`](docs/features.md) for complete list.

### Dominant-Mode Spatial Distribution

Features from eigenvector participation across EEG regions:

- **Core temporoparietal** (T7, T8, P7, P8)
- **Frontal, central, temporal, parietal** regions
- **Concentration metrics**: HHI, entropy, Gini coefficient
- **Dominance gaps**: Core vs other regions

---


## License

License to be confirmed. Please contact the authors before reuse beyond academic review.

---

## Contact

For questions about the methodology or code:
- [Your name and email]
- [Lab/institution website]

**Note**: We cannot provide the raw EEG data. Please see [reproducibility notes](docs/reproducibility_notes.md) for details.

---

## Acknowledgments

This work was supported by [funding sources]. EEG data collection was approved by [ethics committee].
