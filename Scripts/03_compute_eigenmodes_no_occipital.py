#!/usr/bin/env python3
"""
03_compute_eigenmodes_no_occipital.py
======================================

Compute eigenmodes from preprocessed EEG using VAR(1), EXCLUDING occipital channels.

Pipeline variant: "no_occipital"

Drops: O1, O2, Oz, POz (defined in config)

Output structure:
results/eigenmodes/no_occipital/03_eigs_npz/win_1s/<group>/<id>/<cond>/eigs_full.npz
"""

import mne
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'eigenmodes_no_occipital_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# =============================================================================
# METADATA (same as script 02)
# =============================================================================

def infer_subject_id_from_path(fif_path, meta_ids=None, logger=None):
    path_parts = fif_path.parts
    
    if meta_ids:
        for part in path_parts:
            if part in meta_ids:
                if logger:
                    logger.debug(f"    ID from meta.csv: {part}")
                return part, "meta_csv"
    
    try:
        eeg_idx = [i for i, p in enumerate(path_parts) if 'EEG' in p.upper()]
        if eeg_idx:
            next_idx = eeg_idx[0] + 1
            if next_idx < len(path_parts):
                subject_id = path_parts[next_idx]
                if logger:
                    logger.debug(f"    ID from EEG/next: {subject_id}")
                return subject_id, "eeg_folder"
    except:
        pass
    
    parent_folder = fif_path.parent.name
    if logger:
        logger.debug(f"    ID from parent folder: {parent_folder}")
    return parent_folder, "parent_folder"

def load_meta_ids(meta_csv_path):
    if not meta_csv_path or not Path(meta_csv_path).exists():
        return None
    
    try:
        df = pd.read_csv(meta_csv_path)
        for col in ['ID', 'id', 'subject_id', 'sujeto', 'participant']:
            if col in df.columns:
                return set(df[col].astype(str).unique())
        return None
    except:
        return None

def infer_group_from_meta(subject_id, meta_csv_path, logger=None):
    """
    Infer group from meta.csv with support for: ID, y, AACC, group, grupo columns
    """
    if not meta_csv_path or not Path(meta_csv_path).exists():
        return None
    
    try:
        df = pd.read_csv(meta_csv_path)
        
        matching = None
        for id_col in ['ID', 'id', 'subject_id', 'sujeto', 'participant']:
            if id_col in df.columns:
                matching = df[df[id_col].astype(str) == str(subject_id)]
                if len(matching) > 0:
                    break
        
        if matching is None or len(matching) == 0:
            return None
        
        row = matching.iloc[0]
        
        # Check 'y' column (1=aacc, 0=control)
        if 'y' in row.index:
            y_val = row['y']
            if pd.notna(y_val):
                if str(y_val).strip() in ['1', '1.0']:
                    return 'aacc'
                elif str(y_val).strip() in ['0', '0.0']:
                    return 'control'
        
        # Check 'AACC' column
        for aacc_col in ['AACC', 'aacc']:
            if aacc_col in row.index:
                aacc_val = str(row[aacc_col]).strip().upper()
                if aacc_val in ['SI', 'SÍ', 'YES', 'TRUE', '1']:
                    return 'aacc'
                elif aacc_val in ['NO', 'FALSE', '0']:
                    return 'control'
        
        # Check 'group'/'grupo' columns
        for group_col in ['group', 'grupo']:
            if group_col in row.index:
                group_val = str(row[group_col]).lower()
                
                if 'aacc' in group_val or 'gifted' in group_val:
                    return 'aacc'
                elif 'control' in group_val or 'ctrl' in group_val:
                    return 'control'
        
        return None
        
    except Exception as e:
        if logger:
            logger.warning(f"    Could not read meta.csv: {e}")
        return None

def infer_condition_from_filename(fif_path):
    filename = fif_path.stem.lower()
    
    if 'basal' in filename:
        return 'basal'
    elif 'vb' in filename or 'pvt' in filename:
        return 'pvt'
    else:
        return 'unknown'

# =============================================================================
# VAR(1) (same as script 02)
# =============================================================================

def fit_var1_eigenvalues(data):
    n_channels, n_times = data.shape
    X = data - data.mean(axis=1, keepdims=True)
    
    X_t = X[:, :-1].T
    X_t1 = X[:, 1:].T
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X_t, X_t1)
    A = model.coef_
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    return eigenvalues, eigenvectors, A

# =============================================================================
# CHANNEL REFERENCE
# =============================================================================

REF_CH_NAMES = None

def establish_reference_channels(raw, logger, occipital_chs):
    global REF_CH_NAMES
    
    if REF_CH_NAMES is None:
        raw_eeg = raw.copy().pick('eeg')
        
        ch_to_drop = [ch for ch in occipital_chs if ch in raw_eeg.ch_names]
        if ch_to_drop:
            raw_eeg.drop_channels(ch_to_drop)
            logger.info(f"  Dropped occipital: {ch_to_drop}")
        
        REF_CH_NAMES = raw_eeg.ch_names
        logger.info(f"  Reference channels established: {len(REF_CH_NAMES)} (no occipital)")
    
    return REF_CH_NAMES

def reorder_to_reference(raw, ref_ch_names, logger, occipital_chs):
    raw_eeg = raw.copy().pick('eeg')
    
    ch_to_drop = [ch for ch in occipital_chs if ch in raw_eeg.ch_names]
    if ch_to_drop:
        raw_eeg.drop_channels(ch_to_drop)
    
    current_chs = set(raw_eeg.ch_names)
    required_chs = set(ref_ch_names)
    
    missing = required_chs - current_chs
    if missing:
        logger.warning(f"    Missing channels: {missing}")
        return None
    
    raw_eeg.reorder_channels(ref_ch_names)
    
    return raw_eeg

# =============================================================================
# EIGENMODE COMPUTATION
# =============================================================================

def compute_eigenmodes_for_file(fif_path, config, logger, output_base, meta_csv_path, meta_ids, run_log):
    global REF_CH_NAMES
    
    occipital_chs = config.get('occipital_channels', ['O1', 'O2', 'Oz', 'POz'])
    
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        
        if REF_CH_NAMES is None:
            REF_CH_NAMES = establish_reference_channels(raw, logger, occipital_chs)
        
        raw_eeg = reorder_to_reference(raw, REF_CH_NAMES, logger, occipital_chs)
        if raw_eeg is None:
            run_log.append({
                'fif_path': str(fif_path),
                'status': 'SKIP',
                'reason': 'missing_channels',
                'id': None, 'group': None, 'cond': None
            })
            return False
        
        subject_id, id_criterion = infer_subject_id_from_path(fif_path, meta_ids, logger)
        logger.info(f"  Subject ID: {subject_id} (criterion: {id_criterion})")
        
        group = infer_group_from_meta(subject_id, meta_csv_path, logger)
        if group is None:
            logger.warning(f"  Cannot determine group for {subject_id}, SKIPPING")
            run_log.append({
                'fif_path': str(fif_path),
                'status': 'SKIP',
                'reason': 'no_group_in_meta',
                'id': subject_id, 'group': None, 'cond': None
            })
            return False
        
        logger.info(f"  Group: {group}")
        
        condition = infer_condition_from_filename(fif_path)
        logger.info(f"  Condition: {condition}")
        
        data = raw_eeg.get_data()
        sfreq = raw_eeg.info['sfreq']
        n_channels = data.shape[0]
        
        logger.info(f"  Channels: {n_channels}, sfreq: {sfreq} Hz (no occipital)")
        
        clean_dir = Path(config['base_dir']) / config['clean_dir']
        try:
            rel_path = fif_path.relative_to(clean_dir)
        except ValueError:
            rel_path = Path(fif_path.name)
        
        windows = config.get('windows', [1, 2, 4, 8, 16])
        step_sec = config.get('step_sec', 1.0)
        
        for win_sec in windows:
            win_samples = int(win_sec * sfreq)
            step_samples = int(step_sec * sfreq)
            
            logger.info(f"  Window: {win_sec}s ({win_samples} samples)")
            
            n_windows = (data.shape[1] - win_samples) // step_samples + 1
            
            if n_windows < 1:
                logger.warning(f"    Data too short, skipping")
                continue
            
            all_eigenvalues = []
            all_eigenvectors = []
            window_indices = []
            
            for i in range(n_windows):
                start = i * step_samples
                end = start + win_samples
                
                segment = data[:, start:end]
                eig_vals, eig_vecs, A = fit_var1_eigenvalues(segment)
                
                all_eigenvalues.append(eig_vals)
                all_eigenvectors.append(eig_vecs)
                window_indices.append(i)
            
            eigenvalues_arr = np.array(all_eigenvalues)
            eigenvectors_arr = np.array(all_eigenvectors)
            
            logger.info(f"    Computed {len(all_eigenvalues)} windows")
            
            win_tag = f'win_{int(win_sec)}s'
            output_dir = output_base / win_tag / group / subject_id / condition
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = 'eigs_full.npz'
            output_path = output_dir / output_filename
            
            ch_dropped = [ch for ch in occipital_chs if ch in raw.ch_names]
            
            np.savez_compressed(
                output_path,
                # PRIMARY KEYS
                evals=eigenvalues_arr,
                evecs=eigenvectors_arr,
                id=subject_id,
                group=group,
                cond=condition,
                ch_names=REF_CH_NAMES,
                win_sec=win_sec,
                step_sec=step_sec,
                sfreq=sfreq,
                # ALIASES
                eigenvalues=eigenvalues_arr,
                eigenvectors=eigenvectors_arr,
                subject_id=subject_id,
                condition=condition,
                channel_names=REF_CH_NAMES,
                # METADATA
                window_indices=np.array(window_indices),
                n_channels=n_channels,
                source_fif_relpath=str(rel_path),
                pipeline_variant='no_occipital',
                drop_occipital=True,
                occipital_channels_dropped=ch_dropped
            )
            
            logger.info(f"    ✓ Saved: {output_path}")
            
            run_log.append({
                'fif_path': str(fif_path),
                'id': subject_id,
                'group': group,
                'cond': condition,
                'win_sec': win_sec,
                'status': 'SUCCESS',
                'reason': '',
                'n_channels': n_channels,
                'n_windows': len(all_eigenvalues),
                'output_npz': str(output_path)
            })
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        run_log.append({
            'fif_path': str(fif_path),
            'status': 'ERROR',
            'reason': str(e),
            'id': None, 'group': None, 'cond': None
        })
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    global REF_CH_NAMES
    
    config = load_config()
    
    BASE = Path(config['base_dir'])
    input_dir = BASE / config['clean_dir']
    output_base = BASE / config['eigenmodes_no_occipital_dir'] / '03_eigs_npz'
    
    # CORREGIDO: construir meta_csv con BASE
    meta_csv_path = BASE / config['meta_csv'] if config.get('meta_csv') else None
    
    logger = setup_logging(output_base.parent / 'logs')
    
    logger.info("="*70)
    logger.info("EIGENMODE COMPUTATION - NO OCCIPITAL")
    logger.info("="*70)
    logger.info(f"BASE: {BASE}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Meta CSV: {meta_csv_path}")
    logger.info(f"Occipital to drop: {config.get('occipital_channels', ['O1', 'O2', 'Oz', 'POz'])}")
    logger.info("="*70)
    
    meta_ids = load_meta_ids(meta_csv_path)
    if meta_ids:
        logger.info(f"Loaded {len(meta_ids)} subject IDs from meta.csv")
    else:
        logger.warning("No meta.csv found or no IDs loaded")
    
    valid_filenames = ['aacc_basal.fif', 'aacc_vb.fif', 'aacc_pvt.fif']
    all_fif = list(input_dir.rglob('*.fif'))
    fif_files = sorted([f for f in all_fif if f.name in valid_filenames])
    
    if not fif_files:
        logger.error("No valid FIF files found")
        return
    
    logger.info(f"Found {len(fif_files)} valid FIF files")
    logger.info("")
    
    run_log = []
    success_count = 0
    
    for i, fif_path in enumerate(fif_files, 1):
        logger.info(f"[{i}/{len(fif_files)}] {fif_path.name}")
        
        if compute_eigenmodes_for_file(fif_path, config, logger, output_base,
                                      meta_csv_path, meta_ids, run_log):
            success_count += 1
        
        logger.info("")
    
    if REF_CH_NAMES:
        ref_path = output_base.parent / 'channel_reference_no_occipital.csv'
        pd.DataFrame({'channel': REF_CH_NAMES}).to_csv(ref_path, index=False)
        logger.info(f"✓ Saved channel reference: {ref_path}")
    
    if run_log:
        log_path = output_base.parent / 'eigenmode_no_occipital_run_log.csv'
        pd.DataFrame(run_log).to_csv(log_path, index=False)
        logger.info(f"✓ Saved run log: {log_path}")
    
    logger.info("="*70)
    logger.info(f"✓ Success: {success_count}/{len(fif_files)}")
    logger.info("="*70)

if __name__ == '__main__':
    mne.set_log_level('WARNING')
    main()
