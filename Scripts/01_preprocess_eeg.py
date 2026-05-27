#!/usr/bin/env python3
"""
01_preprocess_eeg.py
====================

Preprocessing pipeline for EEG data (FIF → cleaned FIF).

Applies (in order): Notch (50 Hz), Band-pass (1-40 Hz), Bad channel interpolation, CAR

Input:  FIF files from crop_dir (only aacc_basal.fif, aacc_vb.fif, aacc_pvt.fif)
Output: Cleaned FIF files in clean_dir, preserving folder hierarchy
"""

import mne
import yaml
import logging
from pathlib import Path
from datetime import datetime

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'preprocess_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def preprocess_fif(fif_path, output_path, config, logger):
    try:
        logger.info(f"Processing: {fif_path.name}")
        
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        
        # Read from preprocessing section if exists, otherwise from root
        prep = config.get('preprocessing', config)
        
        notch_freq = prep.get('notch_freq', 50)
        bp_low = prep.get('bandpass_low', 1)
        bp_high = prep.get('bandpass_high', 40)
        apply_car = prep.get('apply_car', True)
        interpolate = prep.get('interpolate_bads', True)
        
        if notch_freq:
            logger.info(f"  Notch filter: {notch_freq} Hz")
            raw.notch_filter(freqs=notch_freq, verbose=False)
        
        logger.info(f"  Band-pass: {bp_low}-{bp_high} Hz")
        raw.filter(l_freq=bp_low, h_freq=bp_high, verbose=False)
        
        if interpolate and len(raw.info['bads']) > 0:
            logger.info(f"  Interpolating bad channels: {raw.info['bads']}")
            raw.interpolate_bads(reset_bads=True, verbose=False)
        
        if apply_car:
            logger.info("  Applying CAR")
            raw.set_eeg_reference('average', projection=False, verbose=False)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_path, overwrite=True, verbose=False)
        
        logger.info(f"  ✓ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed {fif_path.name}: {e}")
        return False

def main():
    config = load_config()
    
    # CORREGIDO: usar BASE correctamente
    BASE = Path(config['base_dir'])
    input_dir = BASE / config['crop_dir']
    output_dir = BASE / config['clean_dir']
    
    logger = setup_logging(output_dir / 'logs')
    
    logger.info("="*70)
    logger.info("EEG PREPROCESSING PIPELINE")
    logger.info("="*70)
    logger.info(f"BASE: {BASE}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*70)
    
    # CORREGIDO: filtrar solo archivos válidos
    valid_filenames = ['aacc_basal.fif', 'aacc_vb.fif', 'aacc_pvt.fif']
    
    all_fif_files = list(input_dir.rglob('*.fif'))
    fif_files = [f for f in all_fif_files if f.name in valid_filenames]
    
    if not fif_files:
        logger.warning(f"No valid FIF files found. Expected: {valid_filenames}")
        return
    
    logger.info(f"Found {len(fif_files)} valid files (skipped {len(all_fif_files) - len(fif_files)})")
    logger.info("")
    
    success_count = 0
    
    for i, fif_path in enumerate(fif_files, 1):
        logger.info(f"[{i}/{len(fif_files)}] {fif_path.name}")
        
        # CORREGIDO: conservar jerarquía con relative_to
        rel_path = fif_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        if output_path.exists():
            logger.info("  → Already exists, skipping")
            continue
        
        if preprocess_fif(fif_path, output_path, config, logger):
            success_count += 1
        
        logger.info("")
    
    logger.info("="*70)
    logger.info(f"✓ Success: {success_count}/{len(fif_files)}")
    logger.info("="*70)

if __name__ == '__main__':
    mne.set_log_level('WARNING')
    main()
