"""Summarize saved bomb occupancy without generating or propagating events."""
from contextlib import ExitStack
from pathlib import Path
import json
import os

assert os.environ.get('SLURM_JOB_PARTITION') in {'milano', 'roma'}
import h5py
import numpy as np

BASE = Path(__file__).resolve().parent
summary = {'job': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
           'source_job': '39970274', 'threshold_truth_pe': 1000.,
           'digit_integration_window_ns': 200., 'events': [], 'bright_digits': []}
all_pmts, high_pmts, affected = set(), set(), set()
digit_n = high_digit_n = event_pmt_n = high_event_pmt_n = 0
true_sum = high_true_sum = reco_sum = high_reco_sum = 0.
phys_sum = high_phys_sum = 0.
rounded_high_n = 0

with ExitStack() as stack:
    files = {kind: stack.enter_context(h5py.File(BASE / 'bomb' / kind / f'wc_{kind}_0000.h5', 'r'))
             for kind in ('sensor', 'hits', 'step')}
    events = sorted(k for k in files['sensor'] if k.startswith('event_'))
    n_sensors = len(files['sensor']['config/sensor_positions'])
    for ev in events:
        s, h, g = (files[k][ev] for k in ('sensor', 'hits', 'step'))
        pmt = s['sensor_idx'][:].astype(int)
        charge = s['PE'][:].astype(float)
        nd = len(pmt)
        hd, hw, hp = h['digit_idx'][:], h['PE'][:].astype(float), h['emission_process'][:]
        true = np.bincount(hd, weights=hw, minlength=nd)
        phys = np.bincount(hd[hp == 0], weights=hw[hp == 0], minlength=nd)
        high = true >= summary['threshold_truth_pe']
        rounded_high_n += int(np.count_nonzero(np.rint(true) >= summary['threshold_truth_pe']))
        per_pmt = np.bincount(pmt, weights=true, minlength=n_sensors)
        pmt_high = per_pmt >= summary['threshold_truth_pe']
        digit_n += nd
        high_digit_n += int(high.sum())
        event_pmt_n += len(np.unique(pmt))
        high_event_pmt_n += int(pmt_high.sum())
        true_sum += float(true.sum())
        high_true_sum += float(true[high].sum())
        reco_sum += float(charge.sum())
        high_reco_sum += float(charge[high].sum())
        phys_sum += float(phys.sum())
        high_phys_sum += float(phys[high].sum())
        all_pmts.update(map(int, pmt))
        high_pmts.update(map(int, pmt[high]))
        if high.any():
            affected.add(ev)
        summary['events'].append({
            'event': ev, 'source_event_idx': int(s.attrs['source_event_idx']),
            'saved_digits': nd, 'distinct_pmts': len(np.unique(pmt)),
            'digits_ge_1000_truth_pe': int(high.sum()),
            'pmts_ge_1000_total_event_truth_pe': int(pmt_high.sum()),
            'total_truth_pe': float(true.sum()), 'total_reco_pe': float(charge.sum()),
            'bright_truth_pe': float(true[high].sum()), 'bright_reco_pe': float(charge[high].sum()),
            'bright_truth_fraction_of_event': float(true[high].sum()/true.sum()),
            'bright_reco_fraction_of_event': float(charge[high].sum()/charge.sum()),
            'max_digit_truth_pe': float(true.max(initial=0.)),
            'max_pmt_all_digits_truth_pe': float(per_pmt.max(initial=0.)),
        })
        sh = g['sensor_hits']
        for di in np.flatnonzero(high):
            rows = sh['digit_idx'][:] == di
            t = sh['T'][:][rows]
            w = sh['PE'][:][rows].astype(float)
            # Each row retains the FIRST arrival among this segment's photons
            # in this digit. These times are not all individual photon times.
            earliest = float(t.min())
            onset_bounds = {}
            for window in (5., 10., 20., 50., 100.):
                inside = t < earliest + window
                onset_bounds[str(int(window))] = {
                    'min_physics_pe_from_distinct_segment_first_arrivals': int(inside.sum()),
                    'max_physics_pe_from_groups_starting_before_window_end': float(w[inside].sum()),
                }
            summary['bright_digits'].append({
                'event': ev, 'source_event_idx': int(s.attrs['source_event_idx']),
                'digit_idx': int(di), 'pmt_idx': int(pmt[di]),
                'truth_pe': float(true[di]), 'physics_truth_pe': float(phys[di]),
                'dark_truth_pe': float(true[di]-phys[di]), 'reconstructed_charge_pe': float(charge[di]),
                'same_pmt_saved_digit_count': int(np.count_nonzero(pmt == pmt[di])),
                'same_pmt_all_digits_truth_pe': float(per_pmt[pmt[di]]),
                'contributing_segment_rows': len(t),
                'segment_first_arrival_span_ns': float(t.max() - t.min()),
                'first_arrival_window_bounds_ns': onset_bounds,
            })

summary.update({
    'saved_events': len(events), 'saved_digits': digit_n,
    'distinct_pmts_across_sample': len(all_pmts),
    'event_pmt_pairs': event_pmt_n,
    'digits_ge_1000_truth_pe': high_digit_n,
    'digits_ge_1000_after_rounding_truth_pe': rounded_high_n,
    'affected_events': len(affected), 'affected_event_names': sorted(affected),
    'distinct_pmts_with_bright_digits': len(high_pmts), 'bright_pmt_indices': sorted(high_pmts),
    'event_pmt_pairs_ge_1000_total_truth_pe': high_event_pmt_n,
    'total_truth_pe': true_sum, 'bright_digit_truth_pe': high_true_sum,
    'bright_fraction_of_truth_charge': high_true_sum / true_sum,
    'total_reco_pe': reco_sum, 'bright_digit_reco_pe': high_reco_sum,
    'bright_fraction_of_reconstructed_charge': high_reco_sum / reco_sum,
    'total_physics_truth_pe': phys_sum, 'bright_digit_physics_truth_pe': high_phys_sum,
    'bright_fraction_of_physics_truth_charge': high_phys_sum / phys_sum,
    'timing_limitation': 'Digits integrate 200 ns. Step rows retain only first arrival for each (segment,digit), not every photon arrival. The window bounds are conservative physics-PE bounds relative to the earliest segment arrival; they do not reconstruct an instantaneous waveform or establish an SK nonlinear response correction.',
})
(BASE / 'bomb_occupancy.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
