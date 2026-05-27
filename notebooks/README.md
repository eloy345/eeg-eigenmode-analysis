# Analysis Notebooks

This directory contains Jupyter notebooks documenting the statistical analysis, machine learning, fusion analysis, and figure generation for the EEG eigenmode study.

---

## Notebooks

### statistics.ipynb

Statistical analysis of eigenmode-derived features:
- Group comparisons (AACC vs Control)
- Feature distributions and effect sizes
- Multiple testing correction (BH-FDR)
- Results by condition and window size

**Input**: Feature tables from `results/features/`  
**Output**: Statistical summaries and p-values

---

### machine_learning.ipynb

Machine learning classification pipeline:
- Feature selection and model training
- Cross-validation and performance metrics
- Balanced accuracy, F1 score, AUC
- Comparison across conditions and pipelines

**Input**: Feature tables from `results/features/`  
**Output**: Model performance, feature importance

---

### fusion_analysis.ipynb

Feature fusion strategies:
- Early fusion: Concatenate all features before training
- Late fusion: Train separate models, combine predictions
- Pipeline and representation comparisons
- Optimal model: Early fusion with no-occipital, PVT condition, 8s window

**Input**: Feature tables from `results/features/`  
**Output**: Fusion performance comparisons

---

### figures.ipynb

Figure generation for publication:
- Feature distributions and effect sizes
- ROC curves and confusion matrices
- Performance comparisons
- Saved to `figures/`

**Input**: Feature tables and model results  
**Output**: Publication-quality figures

---

## Running the Notebooks

1. Ensure preprocessing and feature extraction are complete:
   ```bash
   python scripts/01_preprocess_eeg.py
   python scripts/02_compute_eigenmodes_all_channels.py
   python scripts/03_compute_eigenmodes_no_occipital.py
   python scripts/04_extract_global_spectral_geometry.py
   python scripts/05_extract_dominant_mode_spatial_distribution.py
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Run notebooks in order:
   1. `statistics.ipynb`
   2. `machine_learning.ipynb`
   3. `fusion_analysis.ipynb`
   4. `figures.ipynb`

---

## Note on Reproducibility

These notebooks document the analytical workflow. Exact numerical reproduction requires access to the protected feature tables derived from the original EEG dataset, which is not included due to ethical and privacy restrictions involving minors.

The notebooks can be adapted to:
- Different EEG datasets
- Alternative feature sets
- Custom hyperparameter tuning
- Different statistical tests

See `../docs/reproducibility_notes.md` for more details.
