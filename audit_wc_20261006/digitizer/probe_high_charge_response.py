"""Minimal current-code demonstration of missing high-charge PMT response.

Run only under Slurm milano/roma. This measures LUCiD linearity; it does not
invent or fit the missing detector calibration curve.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('actual_digitizer', ROOT / 'lucid/simulation/digitizer.py')
dig = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dig
spec.loader.exec_module(dig)
config = json.loads((ROOT / 'config/SK_WAND_physics_config.json').read_text())
model = dig.resolve_model_config(config['digitizer'])
rng = np.random.default_rng(1901190)
results = {'job': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'], 'response': []}
for npe in [30, 200, 1190, 5000]:
    charge, _ = dig.apply_readout_resolution(
        np.full(300_000, npe), np.zeros(300_000), model, rng)
    row = {'true_pe': npe, 'mean_digit_pe': float(charge.mean(dtype=np.float64)),
           'mean_response_per_pe': float(charge.mean(dtype=np.float64) / npe),
           'standard_error_per_pe': float(charge.std(dtype=np.float64) / np.sqrt(charge.size) / npe)}
    results['response'].append(row)
reference = results['response'][0]['mean_response_per_pe']
for row in results['response']:
    row['relative_to_30pe'] = row['mean_response_per_pe'] / reference
    assert abs(row['relative_to_30pe'] - 1) < .003

# Full host readout, 1190 simultaneous detected photons on one PMT.
n = 1190
t = np.full(n, 70.)
sd, hs, ss = dig.digitize_and_decompose(
    sensor_idx=np.zeros(n, int), charge=np.full(n, .999999),
    t_true=t, t_reco=t, particle_idx=np.zeros(n, int),
    segment_idx=np.zeros(n, int), emission_process=np.zeros(n, int),
    n_sensors=1, model=model, rng=np.random.default_rng(19))
assert len(sd['PE']) == 1
results['full_digitizer_1190pe_flash'] = {'true_pe': float(hs['PE'].sum()), 'recorded_pe': float(sd['PE'][0])}

# Preserve Figure19 for direct visual confirmation of the high-charge sign.
import fitz
pdf = fitz.open(ROOT / 'audit_wc_20261006/digitizer/refs/SK_calibration_1307.0162.pdf')
page = pdf[23]
page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(
    ROOT / 'audit_wc_20261006/digitizer/refs/figure19_page24.png')
print(json.dumps(results, indent=2))
(ROOT / 'audit_wc_20261006/digitizer/high_charge_results.json').write_text(json.dumps(results, indent=2) + '\n')
print('PASS current SK_WAND remains linear at1190PE and5000PE; no calibrated compression applied')
