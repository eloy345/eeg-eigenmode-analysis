#!/usr/bin/env python3
"""
05_extract_dominant_mode_spatial_distribution.py
==================================================

Extract dominant-mode spatial distribution features from VAR(1) eigenvectors.

This script extracts features characterizing how dominant eigenmodes concentrate
their power across brain regions:
- Core temporoparietal region (T7, T8, P7, P8)
- Frontal, central, temporal, parietal regions
- Concentration metrics (HHI, entropy, Gini, etc.)

Methodology: participation vector from selected modes (dominant/osc/near0),
NOT generic mean across all eigenvectors.

Input:  results/eigenmodes/all_channels/03_eigs_npz/
        results/eigenmodes/no_occipital/03_eigs_npz/
Output: results/features/dominant_mode_spatial_distribution/
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config():
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'base_dir': '.',
            'eigenmodes_all_dir': 'results/eigenmodes/all_channels',
            'eigenmodes_no_occipital_dir': 'results/eigenmodes/no_occipital',
            'dominant_mode_spatial_dir': 'results/features/dominant_mode_spatial_distribution',
            'windows': [1.0, 2.0, 4.0, 8.0, 16.0],
        }

config = load_config()

BASE = Path(config['base_dir'])
PIPELINES_IN = {
    "no_occipital": BASE / config['eigenmodes_no_occipital_dir'] / '03_eigs_npz',
    "all_channels":  BASE / config['eigenmodes_all_dir'] / '03_eigs_npz',
}

# Try full name first, fallback to abbreviated
OUT_DIR = BASE / config.get(
    "dominant_mode_spatial_distribution_dir",
    config.get("dominant_mode_spatial_dir", "results/features/dominant_mode_spatial_distribution")
)
DIR_FEAT = OUT_DIR / '01_features'
DIR_LOGS = OUT_DIR / '00_logs'

for d in [DIR_FEAT, DIR_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# Parameters
WIN_SECS = config.get('windows', [1.0, 2.0, 4.0, 8.0, 16.0])
TERCILES = ["all"]
CONDS = ["basal", "pvt"]
KINDS = ["all"]  # Can extend to ["all", "near0", "osc"]

OSC_TAU_IMAG = 0.20
MAX_OSC_PER_WINDOW = 32
K_NEAR0_PER_WINDOW = 2
PARTICIPATION_POWER = 2

# Regions (REAL from your analysis)
CORE_CHANNELS = ["T7", "T8", "P7", "P8"]

REGIONS = {
    "frontal":  ["Fp1","Fpz","Fp2","AF3","AF4","F7","F3","Fz","F4","F8"],
    "central":  ["FC5","FC1","FC2","FC6","C3","Cz","C4","CP1","CP2","CP5","CP6"],
    "temporal": ["T7","T8"],
    "parietal": ["P7","P3","Pz","P4","P8"],
}

REGIONS_DISJOINT = {
    "core": CORE_CHANNELS,
    "frontal": REGIONS["frontal"],
    "central": REGIONS["central"],
    "parietal_nocore": ["P3","Pz","P4"],
    "temporal_nocore": [],
}

OCCIPITAL_SET = set(["O1", "O2", "OZ", "POZ"])
SIMPLE_EXTRA_FEATURES = True
CSV_FLOAT_FMT = "%.8e"

# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def entropy_norm_with_M(p, M):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if M <= 1:
        return np.nan
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    H = -np.sum(p * np.log(p))
    return float(H / np.log(float(M)))

def gini(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    s = p.sum()
    if not np.isfinite(s) or np.isclose(s, 0.0):
        return np.nan
    p = p / s
    x = np.sort(p)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / n - (n + 1.0) / n)

def kurtosis_plain(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size < 4:
        return np.nan
    mu = float(np.mean(p))
    sd = float(np.std(p, ddof=0))
    if sd <= 0:
        return np.nan
    z = (p - mu) / sd
    return float(np.mean(z**4))

def logit(x, eps=1e-9):
    x = float(x)
    x = min(max(x, eps), 1.0 - eps)
    return float(np.log(x / (1.0 - x)))

def tercile_slice(nW: int, terc: str):
    if nW <= 0:
        return slice(0, 0)
    if terc == "all":
        return slice(0, nW)
    a = nW // 3
    b = 2 * nW // 3
    if terc == "early":
        return slice(0, a)
    if terc == "mid":
        return slice(a, b)
    if terc == "late":
        return slice(b, nW)
    return slice(0, nW)

def _indices_from_names(ch_upper, names):
    names_u = {str(n).upper() for n in names}
    idx = [i for i, nm in enumerate(ch_upper) if nm in names_u]
    return np.array(idx, dtype=int)

def _mass_and_transforms(p, idx, M, k):
    if idx.size == 0:
        return (np.nan, np.nan, np.nan)
    s = float(np.sum(p[idx]))
    s_star = float(s - (float(k) / float(M)))
    lg = float(logit(s)) if np.isfinite(s) else np.nan
    return (s, s_star, lg)

# =============================================================================
# MODE SELECTION AND PARTICIPATION
# =============================================================================

def modes_indices_for_kind(ev_row: np.ndarray, kind: str):
    """Select mode indices based on kind."""
    N = ev_row.size
    if kind == "all":
        return np.arange(N, dtype=int)
    if kind == "near0":
        rad = np.abs(ev_row).astype(np.float64)
        k = int(min(K_NEAR0_PER_WINDOW, N))
        return np.argsort(rad)[:k].astype(int)
    if kind == "osc":
        im_abs = np.abs(ev_row.imag).astype(np.float64)
        sel = np.where(im_abs > float(OSC_TAU_IMAG))[0]
        if sel.size == 0:
            return np.array([], dtype=int)
        sel = sel[np.argsort(im_abs[sel])[::-1]]
        sel = sel[:int(min(MAX_OSC_PER_WINDOW, sel.size))]
        return sel.astype(int)
    return np.arange(N, dtype=int)

def participation_vector_from_window(evec_win: np.ndarray, mode_idx: np.ndarray, power: int):
    """
    Compute participation vector from selected modes.
    
    CRITICAL: This is NOT a generic mean of all eigenvectors.
    This computes how the SELECTED modes (dominant/osc/near0) distribute
    their power across channels.
    """
    if mode_idx.size == 0:
        return None
    V = evec_win[:, mode_idx]
    A = np.abs(V).astype(np.float64)
    if power == 2:
        A = A * A
    s = np.sum(A, axis=1)
    tot = float(np.sum(s))
    if not np.isfinite(tot) or tot <= 0:
        return None
    return (s / tot).astype(np.float64)

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def spatial_metrics_from_participation(p: np.ndarray, core_idx: np.ndarray, 
                                       region_idxs: dict, disjoint_idxs: dict):
    """Extract all spatial distribution features from participation vector."""
    p = np.asarray(p, float).ravel()
    M = int(p.size)
    if M <= 1:
        return {}

    hhi = float(np.sum(p * p))
    neff = float(1.0 / hhi) if hhi > 0 else np.nan
    hnorm = float(entropy_norm_with_M(p, M))

    s_core, s_core_star, logit_core = _mass_and_transforms(p, core_idx, M, int(core_idx.size))

    out = {
        "hhi": hhi,
        "n_eff": neff,
        "entropy_norm": hnorm,
        "s_core": s_core,
        "s_core_star": s_core_star,
        "logit_core": logit_core,
        "max_p": float(np.max(p)),
        "top4_mass": float(np.sum(np.sort(p)[::-1][:min(4, M)])),
        "gini": float(gini(p)),
        "kurtosis_p": float(kurtosis_plain(p)),
        "n_channels": M,
        "n_core_present": int(core_idx.size),
    }

    # Regional masses
    for rname, ridx in region_idxs.items():
        s, s_star, lg = _mass_and_transforms(p, ridx, M, int(ridx.size))
        out[f"s_{rname}"] = s
        out[f"s_{rname}_star"] = s_star
        out[f"logit_{rname}"] = lg
        out[f"n_{rname}_present"] = int(ridx.size)

    # Disjoint masses
    used_mask = np.zeros(M, dtype=bool)
    disjoint_masses = {}
    for dn, didx in disjoint_idxs.items():
        if didx.size == 0:
            disjoint_masses[dn] = np.nan
            continue
        disjoint_masses[dn] = float(np.sum(p[didx]))
        used_mask[didx] = True

    s_used = float(np.sum(p[used_mask])) if used_mask.any() else 0.0
    disjoint_masses["rest"] = float(1.0 - s_used) if np.isfinite(s_used) else np.nan

    for dn, val in disjoint_masses.items():
        out[f"sD_{dn}"] = val

    # Extra features
    if SIMPLE_EXTRA_FEATURES:
        out["core_to_rest"] = float(s_core / (1.0 - s_core)) if (np.isfinite(s_core) and 0 < s_core < 1) else np.nan
        reg_masses = [out.get(f"s_{r}", np.nan) for r in ["frontal", "central", "temporal", "parietal"]]
        reg_masses = [float(v) for v in reg_masses if np.isfinite(v)]
        best_other = float(np.max(reg_masses)) if reg_masses else np.nan
        out["dom_max_region"] = best_other
        out["dom_gap_core_vs_best_other"] = float(s_core - best_other) if (np.isfinite(s_core) and np.isfinite(best_other)) else np.nan

    return out

# =============================================================================
# NPZ PROCESSING
# =============================================================================

def extract_rows_from_npz(npz_path: Path, pipeline_variant: str):
    """Extract feature rows from single NPZ file."""
    try:
        d = np.load(npz_path, allow_pickle=True)

        # Try primary keys first, then fallbacks
        evals = d.get("evals", d.get("eigenvalues", None))
        evecs = d.get("evecs", d.get("eigenvectors", None))
        
        if evals is None or evecs is None:
            return []
        
        win_sec = safe_float(d.get("win_sec", d.get("window_sec", np.nan)))
        cond = str(d.get("cond", d.get("condition", ""))).strip().lower()
        group = str(d.get("group", "")).strip().lower()
        sid = str(d.get("id", d.get("subject_id", ""))).strip()
        ch_names = list(d.get("ch_names", d.get("channel_names", [])))

        if np.isfinite(win_sec):
            win_sec = float(np.round(win_sec, 1))

        if win_sec not in WIN_SECS or cond not in CONDS or group not in ["aacc", "control"] or sid == "":
            return []

        if evals.ndim != 2 or evecs.ndim != 3:
            return []
        W, N = evals.shape
        if W <= 0 or N <= 1:
            return []

        # Setup channel indices
        ch_upper = [str(x).upper() for x in ch_names]
        core_idx = _indices_from_names(ch_upper, CORE_CHANNELS)
        region_idxs = {rname: _indices_from_names(ch_upper, rlist) for rname, rlist in REGIONS.items()}
        disjoint_idxs = {dn: _indices_from_names(ch_upper, dlist) for dn, dlist in REGIONS_DISJOINT.items()}

        occ_flag = 1 if any((nm in OCCIPITAL_SET) for nm in ch_upper) else 0

        rows = []
        for kind in KINDS:
            for terc in TERCILES:
                sl = tercile_slice(W, terc)
                ev_block = evals[sl]
                vc_block = evecs[sl]
                Wt = int(ev_block.shape[0])
                if Wt <= 0:
                    continue

                metrics_list = []
                for wi in range(Wt):
                    idx_modes = modes_indices_for_kind(ev_block[wi], kind)
                    p = participation_vector_from_window(vc_block[wi], idx_modes, power=PARTICIPATION_POWER)
                    if p is None:
                        continue
                    m = spatial_metrics_from_participation(
                        p,
                        core_idx=core_idx,
                        region_idxs=region_idxs,
                        disjoint_idxs=disjoint_idxs
                    )
                    if m:
                        metrics_list.append(m)

                if not metrics_list:
                    continue

                # Aggregate across windows (median)
                dfm = pd.DataFrame(metrics_list)
                agg = dfm.median(numeric_only=True).to_dict()

                row = {
                    "pipeline_variant": pipeline_variant,
                    "id": sid,
                    "group": group,
                    "y": 1 if group == "aacc" else 0,
                    "cond": cond,
                    "win_sec": win_sec,
                    "tercile": terc,
                    "kind": kind,
                    "W_used": int(Wt),
                    "N_modes_total": int(N),
                    "participation_power": int(PARTICIPATION_POWER),
                    "occipital_present_flag": int(occ_flag),
                    "npz_path": str(npz_path),
                }
                row.update(agg)
                rows.append(row)

        return rows
    
    except Exception as e:
        return []

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("EXTRACTING DOMINANT-MODE SPATIAL DISTRIBUTION FEATURES")
    print("="*70)
    print(f"Output: {OUT_DIR}")
    print(f"Windows: {WIN_SECS}")
    print(f"Pipelines: {list(PIPELINES_IN.keys())}")
    print("="*70 + "\n")

    all_rows = []
    log_rows = []

    for pipeline_variant, npz_root in PIPELINES_IN.items():
        if not npz_root.exists():
            print(f"⚠️  Pipeline {pipeline_variant} not found: {npz_root}")
            continue

        print(f"\nProcessing pipeline: {pipeline_variant}")
        print(f"  Input: {npz_root}")

        npz_files = list(npz_root.rglob('*.npz'))
        print(f"  Found {len(npz_files)} NPZ files")

        for i, npz_path in enumerate(sorted(npz_files), 1):
            rows = extract_rows_from_npz(npz_path, pipeline_variant)
            
            if rows:
                all_rows.extend(rows)
                log_rows.append({
                    "npz_path": str(npz_path.relative_to(npz_root)),
                    "pipeline": pipeline_variant,
                    "n_rows": len(rows),
                    "status": "SUCCESS",
                })
            else:
                log_rows.append({
                    "npz_path": str(npz_path.relative_to(npz_root)),
                    "pipeline": pipeline_variant,
                    "status": "SKIP",
                })

            if i % 50 == 0:
                print(f"    Processed {i}/{len(npz_files)}...")

    if not all_rows:
        print("\n⚠️  No features extracted")
        return

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    print(f"\n✓ Extracted features from {len(all_rows)} file-condition combinations")
    print(f"  Subjects: {df['id'].nunique()}")
    print(f"  AACC: {df[df['group']=='aacc']['id'].nunique()}")
    print(f"  Control: {df[df['group']=='control']['id'].nunique()}")

    # Save by_window version
    out_bywin = DIR_FEAT / "dominant_mode_spatial_distribution_by_window.csv"
    df.to_csv(out_bywin, index=False, float_format=CSV_FLOAT_FMT)
    print(f"\n✓ Saved: {out_bywin}")

    # Create pooled version (median across windows)
    meta_cols = ["pipeline_variant", "id", "group", "y", "cond", "tercile", "kind"]
    feature_cols = [c for c in df.columns if c not in meta_cols + ["win_sec", "W_used", "N_modes_total", "participation_power", "occipital_present_flag", "npz_path"]]

    pooled = df.groupby(meta_cols, as_index=False)[feature_cols].median(numeric_only=True)

    out_pooled = DIR_FEAT / "dominant_mode_spatial_distribution_pooled.csv"
    pooled.to_csv(out_pooled, index=False, float_format=CSV_FLOAT_FMT)
    print(f"✓ Saved: {out_pooled}")

    # Save feature manifest
    manifest = pd.DataFrame([{"feature": c} for c in feature_cols])
    manifest.to_csv(DIR_FEAT / "dominant_mode_spatial_distribution_feature_columns.csv", index=False)

    # Save extraction log
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        log_out = DIR_LOGS / "dominant_mode_spatial_distribution_extraction_log.csv"
        log_df.to_csv(log_out, index=False)
        print(f"✓ Saved: {log_out}")
    
    # === COMPUTE DELTAS (BASAL → PVT) ===
    print("\n" + "-"*70)
    print("Computing BASAL → PVT deltas...")
    print("-"*70)
    
    # Delta by_window
    delta_bywin = compute_delta(
        df,
        groupby=["pipeline_variant", "id", "group", "y", "win_sec", "tercile", "kind"],
        feature_cols=feature_cols,
        mode="rel"
    )
    
    if not delta_bywin.empty:
        delta_dir = OUT_DIR / "07_delta_basal_to_pvt"
        delta_dir.mkdir(parents=True, exist_ok=True)
        
        out_delta_bywin = delta_dir / "dominant_mode_spatial_distribution_delta_by_window.csv"
        delta_bywin.to_csv(out_delta_bywin, index=False, float_format=CSV_FLOAT_FMT)
        print(f"✓ Saved: {out_delta_bywin}")
        print(f"  Subjects with delta: {delta_bywin['id'].nunique()}")
    else:
        print("⚠️  No delta by_window (need paired basal+pvt)")
    
    # Delta pooled
    delta_pooled = compute_delta(
        pooled,
        groupby=["pipeline_variant", "id", "group", "y", "tercile", "kind"],
        feature_cols=feature_cols,
        mode="rel"
    )
    
    if not delta_pooled.empty:
        delta_dir = OUT_DIR / "07_delta_basal_to_pvt"
        delta_dir.mkdir(parents=True, exist_ok=True)
        
        out_delta_pooled = delta_dir / "dominant_mode_spatial_distribution_delta_pooled.csv"
        delta_pooled.to_csv(out_delta_pooled, index=False, float_format=CSV_FLOAT_FMT)
        print(f"✓ Saved: {out_delta_pooled}")
    else:
        print("⚠️  No delta pooled")

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)


def compute_delta(df, groupby, feature_cols, mode="rel", eps=1e-9):
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
        'rel' for relative: (pvt - basal) / (abs(basal) + eps)
        'diff' for absolute: pvt - basal
    eps : float
        Epsilon for numerical stability
    
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
                    (np.abs(merged[basal_col]) + eps)
                )
            else:  # diff
                delta_data[f"delta__{feat}"] = merged[pvt_col] - merged[basal_col]
    
    return delta_data


if __name__ == '__main__':
    main()
