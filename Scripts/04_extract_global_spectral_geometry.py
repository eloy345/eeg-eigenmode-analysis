#!/usr/bin/env python3
"""
04_extract_global_spectral_geometry.py
=======================================

Extract global spectral geometry features from VAR(1) eigenvalues.

This script extracts geometric features characterizing the eigenvalue distribution
in the complex plane, including:
- Radial features (distance from origin, distance from unit circle)
- Angular features (oscillatory mode angles)
- Ring-based features (modes near unit circle)

Input:  results/eigenmodes/all_channels/03_eigs_npz/
Output: results/features/global_spectral_geometry/

Based on the eigenvalue geometry methodology from the AACC vs Control analysis.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config():
    """Load minimal config - adapt to use config.yaml if available"""
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Fallback defaults
        return {
            'base_dir': '.',
            'eigenmodes_all_dir': 'results/eigenmodes/all_channels',
            'global_spectral_geometry_dir': 'results/features/global_spectral_geometry',
            'windows': [1.0, 2.0, 4.0, 8.0, 16.0],
        }

config = load_config()

BASE = Path(config['base_dir'])
IN_NPZ_ROOT = BASE / config['eigenmodes_all_dir'] / '03_eigs_npz'
OUT_DIR = BASE / config['global_spectral_geometry_dir']

DIR_FEATURES = OUT_DIR / '01_features'
DIR_DELTA = OUT_DIR / '07_delta_basal_to_pvt'
DIR_LOGS = OUT_DIR / '00_logs'

for d in [DIR_FEATURES, DIR_DELTA, DIR_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# Parameters
WIN_SECS = config.get('windows', [1.0, 2.0, 4.0, 8.0, 16.0])
CONDS = ["basal", "pvt"]
TERCILES = ["all"]
FILTER_IMAG_POS = True
OSC_TAU = 0.20
EPS = 1e-9
DELTA_MODE = "rel"
CSV_FLOAT_FMT = "%.8e"

# =============================================================================
# HELPERS
# =============================================================================

def win_tag(w):
    return f"win_{int(w)}s"

def safe_percentile(arr, q):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, q))

def safe_mean(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))

def safe_median(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.median(arr))

def safe_std(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1))

def safe_proportion(arr, condition_func):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(condition_func(arr)))

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features_from_eigenvalues(evals, osc_tau=OSC_TAU, filter_imag_pos=FILTER_IMAG_POS):
    """
    Extract all global spectral geometry features from eigenvalues.
    
    Parameters:
    -----------
    evals : array complex, shape (W, N)
        Eigenvalues from VAR(1) model
    osc_tau : float
        Threshold for oscillatory mode (Im > osc_tau)
    filter_imag_pos : bool
        If True, only use eigenvalues with Im >= 0 (avoid conjugate duplicates)
    
    Returns:
    --------
    dict with all features
    """
    
    # Flatten and clean
    vals = evals.ravel()
    vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
    
    # Filter by positive imaginary part
    if filter_imag_pos:
        vals = vals[vals.imag >= 0]
    
    n_all = len(vals)
    
    if n_all == 0:
        return {key: np.nan for key in get_all_feature_names()}
    
    # Basic characteristics
    radius = np.abs(vals)
    signed_dist = radius - 1.0
    abs_dist = np.abs(signed_dist)
    angle_abs_deg = np.abs(np.angle(vals)) * 180.0 / np.pi
    imag_abs = np.abs(vals.imag)
    
    # Oscillatory set
    osc_mask = imag_abs > osc_tau
    n_osc = int(np.sum(osc_mask))
    osc_frac = float(n_osc) / float(n_all) if n_all > 0 else np.nan
    
    features = {}
    
    # === ALL modes features ===
    # Radial
    features["all__rad_median"] = safe_median(radius)
    features["all__rad_p95"] = safe_percentile(radius, 95)
    features["all__rad_std"] = safe_std(radius)
    features["all__rad_signed_median"] = safe_median(signed_dist)
    features["all__rad_signed_mean"] = safe_mean(signed_dist)
    features["all__rad_signed_std"] = safe_std(signed_dist)
    features["all__rad_prop_in"] = safe_proportion(radius, lambda r: r < 1.0)
    features["all__rad_prop_out"] = safe_proportion(radius, lambda r: r > 1.0)
    features["all__rad_abs_mean"] = safe_mean(radius)
    features["all__rad_abs_std"] = safe_std(radius)
    
    # Distance from unit circle
    features["all__dist_uc_abs_mean"] = safe_mean(abs_dist)
    features["all__dist_uc_abs_std"] = safe_std(abs_dist)
    features["all__dist_p95"] = safe_percentile(abs_dist, 95)
    features["all__dist_prop_gt_0.02"] = safe_proportion(abs_dist, lambda d: d > 0.02)
    features["all__dist_prop_gt_0.05"] = safe_proportion(abs_dist, lambda d: d > 0.05)
    features["all__dist_prop_gt_0.10"] = safe_proportion(abs_dist, lambda d: d > 0.10)
    
    # Oscillatory fraction
    features["osc_frac"] = osc_frac
    
    # === OSC modes features ===
    if n_osc == 0:
        # No oscillatory modes, fill with NaN
        osc_features = [
            "osc__rad_median", "osc__rad_p95", "osc__rad_std",
            "osc__rad_signed_median", "osc__rad_signed_mean", "osc__rad_signed_std",
            "osc__rad_prop_in", "osc__rad_prop_out",
            "osc__rad_abs_mean", "osc__rad_abs_std",
            "osc__dist_uc_abs_mean", "osc__dist_uc_abs_std",
            "osc__dist_p95",
            "osc__dist_prop_gt_0.02", "osc__dist_prop_gt_0.05", "osc__dist_prop_gt_0.10",
            "osc__angle_p95_deg", "osc__angle_prop_gt_35deg", "osc__angle_prop_gt_45deg",
            "osc__ang_abs_mean_deg", "osc__ang_abs_std_deg",
            "osc__imag_abs_mean", "osc__imag_abs_std",
            "osc__ring0.02__angle_p95_deg", "osc__ring0.02__angle_prop_gt_35deg",
            "osc__ring0.02__angle_prop_gt_45deg",
            "osc__ring0.05__angle_p95_deg", "osc__ring0.05__angle_prop_gt_35deg",
            "osc__ring0.05__angle_prop_gt_45deg"
        ]
        for feat in osc_features:
            features[feat] = np.nan
    else:
        # Extract oscillatory values
        vals_osc = vals[osc_mask]
        radius_osc = np.abs(vals_osc)
        signed_dist_osc = radius_osc - 1.0
        abs_dist_osc = np.abs(signed_dist_osc)
        angle_abs_deg_osc = np.abs(np.angle(vals_osc)) * 180.0 / np.pi
        imag_abs_osc = np.abs(vals_osc.imag)
        
        # Radial OSC
        features["osc__rad_median"] = safe_median(radius_osc)
        features["osc__rad_p95"] = safe_percentile(radius_osc, 95)
        features["osc__rad_std"] = safe_std(radius_osc)
        features["osc__rad_signed_median"] = safe_median(signed_dist_osc)
        features["osc__rad_signed_mean"] = safe_mean(signed_dist_osc)
        features["osc__rad_signed_std"] = safe_std(signed_dist_osc)
        features["osc__rad_prop_in"] = safe_proportion(radius_osc, lambda r: r < 1.0)
        features["osc__rad_prop_out"] = safe_proportion(radius_osc, lambda r: r > 1.0)
        features["osc__rad_abs_mean"] = safe_mean(radius_osc)
        features["osc__rad_abs_std"] = safe_std(radius_osc)
        
        # Distance OSC
        features["osc__dist_uc_abs_mean"] = safe_mean(abs_dist_osc)
        features["osc__dist_uc_abs_std"] = safe_std(abs_dist_osc)
        features["osc__dist_p95"] = safe_percentile(abs_dist_osc, 95)
        features["osc__dist_prop_gt_0.02"] = safe_proportion(abs_dist_osc, lambda d: d > 0.02)
        features["osc__dist_prop_gt_0.05"] = safe_proportion(abs_dist_osc, lambda d: d > 0.05)
        features["osc__dist_prop_gt_0.10"] = safe_proportion(abs_dist_osc, lambda d: d > 0.10)
        
        # Angular OSC
        features["osc__angle_p95_deg"] = safe_percentile(angle_abs_deg_osc, 95)
        features["osc__angle_prop_gt_35deg"] = safe_proportion(angle_abs_deg_osc, lambda a: a > 35.0)
        features["osc__angle_prop_gt_45deg"] = safe_proportion(angle_abs_deg_osc, lambda a: a > 45.0)
        features["osc__ang_abs_mean_deg"] = safe_mean(angle_abs_deg_osc)
        features["osc__ang_abs_std_deg"] = safe_std(angle_abs_deg_osc)
        features["osc__imag_abs_mean"] = safe_mean(imag_abs_osc)
        features["osc__imag_abs_std"] = safe_std(imag_abs_osc)
        
        # Ring features (near unit circle)
        for ring_thr in [0.02, 0.05]:
            ring_mask = abs_dist_osc <= ring_thr
            n_ring = int(np.sum(ring_mask))
            
            prefix = f"osc__ring{ring_thr:.2f}__"
            
            if n_ring == 0:
                features[f"{prefix}angle_p95_deg"] = np.nan
                features[f"{prefix}angle_prop_gt_35deg"] = np.nan
                features[f"{prefix}angle_prop_gt_45deg"] = np.nan
            else:
                angle_ring = angle_abs_deg_osc[ring_mask]
                features[f"{prefix}angle_p95_deg"] = safe_percentile(angle_ring, 95)
                features[f"{prefix}angle_prop_gt_35deg"] = safe_proportion(angle_ring, lambda a: a > 35.0)
                features[f"{prefix}angle_prop_gt_45deg"] = safe_proportion(angle_ring, lambda a: a > 45.0)
    
    # AACC p95 features (computed later with group-specific reference)
    features["all__dist_prop_gt_aacc_p95"] = np.nan
    features["osc__dist_prop_gt_aacc_p95"] = np.nan
    features["osc__angle_prop_gt_aacc_p95"] = np.nan
    features["osc__score_prop_gt_aacc_p95"] = np.nan
    
    return features


def get_all_feature_names():
    """Return ordered list of all feature names."""
    features = [
        # ALL radial
        "all__rad_median", "all__rad_p95", "all__rad_std",
        "all__rad_signed_median", "all__rad_signed_mean", "all__rad_signed_std",
        "all__rad_prop_in", "all__rad_prop_out",
        "all__rad_abs_mean", "all__rad_abs_std",
        
        # ALL distance
        "all__dist_uc_abs_mean", "all__dist_uc_abs_std",
        "all__dist_p95",
        "all__dist_prop_gt_0.02", "all__dist_prop_gt_0.05", "all__dist_prop_gt_0.10",
        
        # OSC fraction
        "osc_frac",
        
        # OSC radial
        "osc__rad_median", "osc__rad_p95", "osc__rad_std",
        "osc__rad_signed_median", "osc__rad_signed_mean", "osc__rad_signed_std",
        "osc__rad_prop_in", "osc__rad_prop_out",
        "osc__rad_abs_mean", "osc__rad_abs_std",
        
        # OSC distance
        "osc__dist_uc_abs_mean", "osc__dist_uc_abs_std",
        "osc__dist_p95",
        "osc__dist_prop_gt_0.02", "osc__dist_prop_gt_0.05", "osc__dist_prop_gt_0.10",
        
        # OSC angular
        "osc__angle_p95_deg", "osc__angle_prop_gt_35deg", "osc__angle_prop_gt_45deg",
        "osc__ang_abs_mean_deg", "osc__ang_abs_std_deg",
        "osc__imag_abs_mean", "osc__imag_abs_std",
        
        # Ring features
        "osc__ring0.02__angle_p95_deg", "osc__ring0.02__angle_prop_gt_35deg",
        "osc__ring0.02__angle_prop_gt_45deg",
        "osc__ring0.05__angle_p95_deg", "osc__ring0.05__angle_prop_gt_35deg",
        "osc__ring0.05__angle_prop_gt_45deg",
        
        # AACC p95 features (optional, with leakage warning if used)
        "all__dist_prop_gt_aacc_p95",
        "osc__dist_prop_gt_aacc_p95",
        "osc__angle_prop_gt_aacc_p95",
        "osc__score_prop_gt_aacc_p95"
    ]
    return features

# =============================================================================
# SCAN AND PROCESS NPZ FILES
# =============================================================================

def scan_and_extract():
    """Scan NPZ files and extract features."""
    
    print("\n" + "="*70)
    print("EXTRACTING GLOBAL SPECTRAL GEOMETRY FEATURES")
    print("="*70)
    print(f"Input:  {IN_NPZ_ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Windows: {WIN_SECS}")
    print("="*70 + "\n")
    
    all_rows = []
    log_rows = []
    
    # Find all NPZ files
    npz_files = list(IN_NPZ_ROOT.rglob('*.npz'))
    
    if not npz_files:
        print(f"⚠️  No NPZ files found in {IN_NPZ_ROOT}")
        return
    
    print(f"Found {len(npz_files)} NPZ files\n")
    
    for i, npz_path in enumerate(sorted(npz_files), 1):
        try:
            # Load NPZ
            d = np.load(npz_path, allow_pickle=True)
            
            # Try primary keys first, then fallbacks
            evals = d.get("evals", d.get("eigenvalues", None))
            if evals is None:
                print(f"[{i}/{len(npz_files)}] ⚠️  {npz_path.name}: no eigenvalues")
                continue
            
            # Load metadata
            subject_id = str(d.get("id", d.get("subject_id", "unknown")))
            group = str(d.get("group", "unknown")).strip().lower()
            cond = str(d.get("cond", d.get("condition", "unknown"))).strip().lower()
            win_sec = float(d.get("win_sec", d.get("window_sec", 0)))
            
            if cond not in CONDS or win_sec not in WIN_SECS:
                continue
            
            # Extract features
            features = extract_features_from_eigenvalues(evals)
            
            # Build row
            row = {
                "id": subject_id,
                "group": group,
                "y": 1 if group == "aacc" else 0,
                "cond": cond,
                "win_sec": win_sec,
                "tercile": "all",
            }
            row.update(features)
            all_rows.append(row)
            
            # Log
            log_rows.append({
                "npz_path": str(npz_path.relative_to(IN_NPZ_ROOT)),
                "id": subject_id,
                "group": group,
                "cond": cond,
                "win_sec": win_sec,
                "status": "SUCCESS",
                "n_windows": evals.shape[0] if evals.ndim > 1 else 1,
            })
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(npz_files)}...")
            
        except Exception as e:
            print(f"[{i}/{len(npz_files)}] ✗ {npz_path.name}: {e}")
            log_rows.append({
                "npz_path": str(npz_path),
                "status": "ERROR",
                "error": str(e),
            })
    
    if not all_rows:
        print("\n⚠️  No features extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    print(f"\n✓ Extracted features from {len(all_rows)} files")
    print(f"  Subjects: {df['id'].nunique()}")
    print(f"  AACC: {df[df['group']=='aacc']['id'].nunique()}")
    print(f"  Control: {df[df['group']=='control']['id'].nunique()}")
    
    # Save by_window version
    out_bywin = DIR_FEATURES / "global_spectral_geometry_by_window.csv"
    df.to_csv(out_bywin, index=False, float_format=CSV_FLOAT_FMT)
    print(f"\n✓ Saved: {out_bywin}")
    
    # Create pooled version (median across windows per subject/condition)
    meta_cols = ["id", "group", "y", "cond", "tercile"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["win_sec"]]
    
    pooled = df.groupby(meta_cols, as_index=False)[feature_cols].median(numeric_only=True)
    
    out_pooled = DIR_FEATURES / "global_spectral_geometry_pooled.csv"
    pooled.to_csv(out_pooled, index=False, float_format=CSV_FLOAT_FMT)
    print(f"✓ Saved: {out_pooled}")
    
    # Save feature manifests
    manifest_bywin = pd.DataFrame([{"feature": c} for c in feature_cols])
    manifest_bywin.to_csv(DIR_FEATURES / "global_spectral_feature_columns_by_window.csv", index=False)
    
    manifest_pooled = pd.DataFrame([{"feature": c} for c in feature_cols])
    manifest_pooled.to_csv(DIR_FEATURES / "global_spectral_feature_columns_pooled.csv", index=False)
    
    # Save extraction log
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        log_out = DIR_LOGS / "global_spectral_extraction_log.csv"
        log_df.to_csv(log_out, index=False)
        print(f"✓ Saved: {log_out}")
    
    # === COMPUTE DELTAS (BASAL → PVT) ===
    print("\n" + "-"*70)
    print("Computing BASAL → PVT deltas...")
    print("-"*70)
    
    # Delta by_window
    delta_bywin = compute_delta(
        df, 
        groupby=["id", "group", "y", "win_sec", "tercile"],
        feature_cols=feature_cols,
        mode=DELTA_MODE
    )
    
    if not delta_bywin.empty:
        out_delta_bywin = DIR_DELTA / "global_spectral_delta_by_window.csv"
        delta_bywin.to_csv(out_delta_bywin, index=False, float_format=CSV_FLOAT_FMT)
        print(f"✓ Saved: {out_delta_bywin}")
        print(f"  Subjects with delta: {delta_bywin['id'].nunique()}")
    else:
        print("⚠️  No delta by_window (need paired basal+pvt)")
    
    # Delta pooled
    delta_pooled = compute_delta(
        pooled,
        groupby=["id", "group", "y", "tercile"],
        feature_cols=feature_cols,
        mode=DELTA_MODE
    )
    
    if not delta_pooled.empty:
        out_delta_pooled = DIR_DELTA / "global_spectral_delta_pooled.csv"
        delta_pooled.to_csv(out_delta_pooled, index=False, float_format=CSV_FLOAT_FMT)
        print(f"✓ Saved: {out_delta_pooled}")
    else:
        print("⚠️  No delta pooled")
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)


def compute_delta(df, groupby, feature_cols, mode="rel"):
    """
    Compute BASAL → PVT delta for paired conditions.
    
    Parameters:
    -----------
    df : DataFrame
        Input with 'cond' column
    groupby : list
        Columns to group by (should NOT include 'cond')
    feature_cols : list
        Feature columns to compute delta on
    mode : str
        'rel' for relative: (pvt - basal) / (abs(basal) + EPS)
        'diff' for absolute: pvt - basal
    
    Returns:
    --------
    DataFrame with delta__ prefixed columns
    """
    
    # Separate basal and pvt
    df_basal = df[df['cond'] == 'basal'].copy()
    df_pvt = df[df['cond'] == 'pvt'].copy()
    
    if df_basal.empty or df_pvt.empty:
        return pd.DataFrame()
    
    # Merge on groupby keys
    merged = df_basal.merge(
        df_pvt,
        on=groupby,
        suffixes=('_basal', '_pvt'),
        how='inner'
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Compute deltas
    delta_data = merged[groupby].copy()
    
    for feat in feature_cols:
        basal_col = f"{feat}_basal"
        pvt_col = f"{feat}_pvt"
        
        if basal_col in merged.columns and pvt_col in merged.columns:
            if mode == "rel":
                delta_data[f"delta__{feat}"] = (
                    (merged[pvt_col] - merged[basal_col]) / 
                    (np.abs(merged[basal_col]) + EPS)
                )
            else:  # diff
                delta_data[f"delta__{feat}"] = merged[pvt_col] - merged[basal_col]
    
    return delta_data


if __name__ == '__main__':
    scan_and_extract()
