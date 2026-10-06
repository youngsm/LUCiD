"""Replay only saved ROOT event 5, preserving its original optical RNG keys."""
from pathlib import Path
import json
import os

assert os.environ.get('SLURM_JOB_PARTITION') in {'milano', 'roma'}
import h5py
import numpy as np
from lucid.production.run_job import _run_lucid
from lucid.sources import event_generation as generation

BASE = Path(__file__).resolve().parent
SOURCE_IDX = 5
prior = json.loads((BASE / 'results_bomb.json').read_text())
expected = next(row for row in prior['cases']['bomb']['events'] if row['source_event_idx'] == SOURCE_IDX)
with h5py.File(BASE / 'bomb/sensor/wc_sensor_0000.h5') as sf, h5py.File(BASE / 'bomb/hits/wc_hits_0000.h5') as hf:
    ev = next(k for k in sf if k.startswith('event_') and sf[k].attrs['source_event_idx'] == SOURCE_IDX)
    h, s = hf[ev], sf[ev]
    truth = np.bincount(h['digit_idx'][:], weights=h['PE'][:], minlength=len(s['PE']))
    digit_idx = int(truth.argmax())
    pmt_idx = int(s['sensor_idx'][digit_idx])
    target_truth = float(truth[digit_idx])
    target_reco = float(s['PE'][digit_idx])
    same_pmt_digits = int(np.count_nonzero(s['sensor_idx'][:] == pmt_idx))
    assert same_pmt_digits == 1, 'Need explicit digit-window matching if PMT has multiple saved digits'
    mask = h['digit_idx'][:] == digit_idx
    dark_pe = float(h['PE'][:][mask & (h['emission_process'][:] == 2)].sum())

real_keys = generation.derive_event_keys
real_reader = generation._read_event_raw
real_gather = generation.gather_photon_deposits

def replay_keys(master_seed, job_id, event_idx, **kwargs):
    return real_keys(master_seed, job_id, SOURCE_IDX, **kwargs)

def replay_reader(path, entry_index):
    return real_reader(path, SOURCE_IDX)

class CapturedReplay(Exception):
    pass

def capture_deposits(process_outputs):
    deposits = real_gather(process_outputs)
    valid = np.isfinite(deposits['t_true']) & np.isfinite(deposits['t_reco'])
    total = float(deposits['charge'][valid].sum())
    np.testing.assert_allclose(total, expected['detected_before_digitizer'], rtol=0., atol=1e-7)
    target = valid & (deposits['sensor_idx'] == pmt_idx)
    t = np.sort(deposits['t_true'][target])
    q = float(deposits['charge'][target].sum())
    np.testing.assert_allclose(q + dark_pe, target_truth, rtol=0., atol=5e-4)
    assert len(t) == round(target_truth - dark_pe)
    quantiles = np.quantile(t, [0., .05, .25, .5, .75, .95, 1.])
    windows = {}
    for width in (1., 5., 10., 20., 50., 100., 200.):
        counts = np.searchsorted(t, t + width, side='right') - np.arange(len(t))
        first = int(counts.argmax())
        windows[str(int(width))] = {
            'max_physical_photoelectrons': int(counts[first]),
            'fraction_of_digit_physics_pe': float(counts[first]/len(t)),
            'start_ns_in_g4_frame': float(t[first]),
        }
    report = {
        'job': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
        'source_event_idx': SOURCE_IDX, 'saved_event_group': ev,
        'digit_idx': digit_idx, 'pmt_idx': pmt_idx,
        'master_seed': 918273, 'production_job_id': 2718,
        'replay_total_detected_pe': total,
        'original_total_detected_pe': expected['detected_before_digitizer'],
        'exact_total_detected_match': total == expected['detected_before_digitizer'],
        'replay_physical_photoelectrons_on_pmt': len(t),
        'replay_physics_charge_on_pmt': q,
        'saved_digit_truth_pe': target_truth,
        'saved_digit_reconstructed_charge_pe': target_reco,
        'dark_pe_in_saved_digit': dark_pe,
        'same_pmt_saved_digit_count': same_pmt_digits,
        'arrival_quantiles_ns_in_g4_frame': dict(zip(('min','p05','p25','p50','p75','p95','max'), map(float, quantiles))),
        'full_true_arrival_span_ns': float(t[-1]-t[0]),
        'central_90_percent_true_arrival_span_ns': float(quantiles[5]-quantiles[1]),
        'max_sliding_window_counts_ns': windows,
        'timing_scope': 'Optical arrival times at the PMT from identical source event and optical random keys, before PMT transit-time spread or electronics response. This is not an anode-current waveform or a calibrated nonlinear-charge correction.',
    }
    np.savez(BASE / 'bright_digit_raw_deposits.npz',
             t_true=deposits['t_true'][target], t_reco=deposits['t_reco'][target],
             charge=deposits['charge'][target], segment_idx=deposits['segment_idx'][target],
             particle_idx=deposits['particle_idx'][target])
    (BASE / 'bright_digit_timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    raise CapturedReplay

generation.derive_event_keys = replay_keys
generation._read_event_raw = replay_reader
generation.gather_photon_deposits = capture_deposits
try:
    _run_lucid(root_file=BASE / 'bomb_source/bomb.root', output_dir=BASE / 'replay_only',
               config=json.loads((BASE / 'bomb_source/original_config.json').read_text()),
               file_index=0, n_events=1, master_seed=918273, job_id=2718, detector='SK_WAND')
except CapturedReplay:
    print('MATCHED_SAVED_EVENT_AND_CAPTURED_RAW_DEPOSITS', flush=True)
else:
    raise AssertionError('Replay did not reach deposit capture')
