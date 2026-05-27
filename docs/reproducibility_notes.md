# Reproducibility Notes

This document clarifies the scope of this repository and limitations regarding computational reproducibility.

---

## What This Repository Provides

✅ **Methodological transparency**
- Complete preprocessing pipeline
- VAR(1) eigenmode computation methodology
- Feature extraction algorithms
- Documentation of all processing steps

✅ **Code reusability**
- Scripts can be applied to other EEG datasets
- Feature extraction can be adapted to different paradigms
- Methods can be validated on independent data

✅ **Results verification**
- Published statistical findings can be checked
- Feature definitions can be inspected
- Processing choices can be evaluated

---

## What This Repository Does NOT Provide

❌ **Raw EEG data**
- Cannot be shared due to ethical restrictions
- Data contain recordings from minors
- Subject to privacy and consent limitations

❌ **Preprocessed EEG**
- Derived from protected raw data
- Would require data sharing agreement

❌ **Computed features**
- Derived from protected raw data
- Would violate study ethics approval

❌ **Complete computational reproducibility**
- Cannot reproduce exact numerical results without original data
- Can replicate methodology on new datasets

---

## Why Data Cannot Be Shared

### Ethical Constraints

The EEG data used in this study:
1. Were collected from **minors**
2. Required **informed consent** from parents/guardians
3. Were approved by **ethics committee** under specific conditions
4. **Cannot be publicly released** regardless of anonymization

### Legal/Regulatory

- EU GDPR: Protects data of minors
- Local privacy laws: May restrict data sharing
- Institutional policy: Limits public data dissemination

### Scientific Practice

Even in studies that do share data, EEG from minors is typically restricted to verified researchers within institutions.

---

## Scope of Reproducibility

### ✅ Can Be Reproduced



### ❌ Cannot Be Reproduced

1. **Exact numerical results**
   - The statistical workflow and feature definitions can be inspected, but exact numerical results cannot be reproduced without access to the protected dataset.

2. **Group comparisons (AACC vs Control)**
   - Cannot form groups without original dataset

3. **Cross-validation results**
   - Requires original subject set

4. **Specific subject-level findings**
   - Tied to original cohort

---

## How to Use This Repository

### For Validation

1. Review preprocessing code → Verify filter choices match literature
2. Inspect eigenmode computation → Confirm VAR(1) implementation
3. Check feature definitions → Ensure mathematical correctness
4. Examine feature extraction → Trace from eigenmodes to outputs

### For Replication on New Data

1. Prepare EEG data in FIF format
2. Set up directory structure (see `data_structure.md`)
3. Configure paths in `config.yaml`
4. Run preprocessing → eigenmode extraction → feature extraction
5. Perform your own statistical analysis

### For Extension

1. **Add new features**: Extend feature extraction scripts
2. **Modify preprocessing**: Adjust filter parameters in config
3. **Test variants**: No-occipital pipeline already provides sensitivity analysis
4. **Different electrodes**: Update regional definitions in feature extraction

---

## Methodological Notes

### VAR(1) Model Assumptions

The VAR(1) model assumes:
- **Stationarity**: EEG within windows is approximately stationary
- **Linearity**: Temporal dynamics captured by linear relationships
- **Order-1 sufficiency**: All dependencies within 1 timestep

**Verification**: Check that:
- Windows are short enough (1-16 sec) for stationarity
- Eigenvalues have reasonable magnitudes
- Features are stable across window sizes

### Window Selection

Analysis used multiple window sizes: [1, 2, 4, 8, 16] seconds

**Rationale**:
- **Short windows (1-2 sec)**: Capture rapid transients
- **Medium windows (4-8 sec)**: Balance signal stability and resolution
- **Long windows (16 sec)**: Assess low-frequency structure

**Aggregation**: Pooled results use median across all windows
- Robust to outliers
- Captures overall characteristics
- Not sensitive to specific window choice

### Channel Reference

**Common average reference (CAR)** was chosen because:
- Reduces volume conduction artifacts
- Improves spatial specificity
- Standard in EEG research
- Computed after bad channel interpolation

**Impact on interpretation**:
- Features reflect activity relative to global average
- Regional "dominance" is relative to whole-brain activity
- Not absolute power, but relative distribution

### Temporal Aggregation

**Median across windows** (not mean):
- Robust to outliers
- Handles occasional artifacts gracefully
- More reproducible across preprocessing variations

### Occipital Removal Rationale

**No-occipital variant** separates:
- **Posterior visual processing**: O1, O2, Oz, POz
- **Broader systems activity**: Remaining 26 channels

**Use for**:
- Testing feature robustness
- Isolating non-visual contributions
- Separating visual attention effects (PVT is visual task)

---

## Pipeline Stability

### Sources of Variability

1. **Preprocessing parameters**
   - Filter cutoffs (1-40 Hz): Used standard range
   - Notch frequency (50 Hz): Specific to Europe (60 Hz in US)
   - Bad channel interpolation: Improves robustness

2. **Eigenmode computation**
   - Window size: Multiple sizes analyzed
   - Demeaning: Essential for stationarity
   - Complex eigenvalues: Inherent to VAR(1)

3. **Feature aggregation**
   - Median (not mean): More stable
   - Multiple terciles: Captures temporal variation
   - Delta computation: Uses relative change for scale-invariance

### Robustness

✅ **Features robust to**:
- Sampling rate (1-16 sec windows span 250-1000 Hz)
- Minor preprocessing variations
- Channel number reduction (no-occipital analysis)

⚠️ **Features sensitive to**:
- EEG quality (bad channels, noise)
- Recording duration (need sufficient data for VAR fitting)
- Task protocol (oscillatory patterns may be task-specific)

### Recommended Checks

1. **Verify data quality**
   - Check for bad channels
   - Inspect power spectra
   - Look for obvious artifacts

2. **Validate pipeline**
   - Test on small subset first
   - Inspect eigenvalue distributions
   - Compare with/without preprocessing steps

3. **Check outputs**
   - No all-NaN features
   - Reasonable value ranges (see `features.md`)
   - Consistent across similar windows

---

## Limitations

### Data-Specific Limitations

This pipeline was developed and tested on:
- **Subject group**: Minors
- **Recording equipment**: [Specify if known]
- **Electrode montage**: Standard 10-20 system (~30 channels)
- **Tasks**: Baseline (basal) and visual attention (PVT)
- **Recording duration**: ~10 minutes per condition

**Applicability to other datasets**:
- May work with other age groups (untested)
- Requires standard 10-20 montage (or adaptation)
- May need adjustment for different recording conditions

### Methodological Limitations

1. **VAR(1) order assumption**
   - Higher-order VAR may capture additional dynamics
   - Order selection was not formally tested
   - See literature on VAR model selection

2. **Stationarity assumption**
   - Windows assumed approximately stationary
   - May not hold for transitions or complex patterns
   - Consider adaptive methods for non-stationary data

3. **Feature selection**
   - Features chosen based on domain knowledge
   - Other eigenmode-derived features possible
   - No formal feature selection procedure applied

4. **No occipital variant**
   - Single sensitivity analysis (other variants possible)
   - Occipital removal may be too conservative
   - Alternative: Weighted analysis by region

### Computational Limitations

1. **Single-window analysis**
   - Features aggregated across windows
   - No explicit temporal modeling
   - Cross-window correlations not used

2. **Linear eigenmode analysis**
   - Nonlinear dynamics not captured
   - Phase coupling not explicitly modeled
   - Complex-valued eigenvectors assumed sufficient

3. **No group-level variance modeling**
   - Features extracted independently
   - No explicit mixed-effects framework
   - Inter-subject variability not modeled

---

## Recommended Extensions

### For Methodology
- Test VAR(2) or VAR(p) variants
- Explore alternative features from eigenvalues/eigenvectors
- Validate on held-out subjects
- Cross-dataset validation

### For Application
- Apply to different EEG paradigms
- Test on clinical populations
- Assess longitudinal stability
- Real-time online feature extraction

### For Features
- Add higher-order statistics
- Include phase relationships
- Compute cross-subject eigenmodes
- Develop adaptive window analysis

---

## Contact for Questions

For questions about:
- **Methodology**: [Author contact]
- **Code usage**: [Author contact]
- **Data access**: See ethics committee or [Institution contact]

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eegigenmode2024,
  title={EEG Eigenmode Analysis Pipeline},
  author={[García-Pérez Eloy]},
  year={2026},
  url={https://github.com/yourusername/eeg-eigenmode-analysis}
}
```

And cite the original paper:

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2024 | Initial public release |

---

## License

License to be confirmed. Please contact the authors before reuse beyond academic review.
