"""Independent real PhotonSim -> production SK_WAND check. Run on milano/roma."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import time

assert os.environ.get('SLURM_JOB_PARTITION') in {'milano', 'roma'}
import h5py
import numpy as np
from lucid.production.run_job import _run_lucid
from lucid.sources import event_generation as generation
from lucid.sources.root_reader import _read_event_raw
from lucid.geometry import generate_detector

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / 'photonsim_physics'
PMT_RADIUS_M = float(generate_detector(str(BASE.parents[1] / 'config/SK_WAND_geom_config.json')).S_radius)
CASES = [('electron', 5, 100.), ('muon', 5, 1000.), ('pion', 10, 1000.), ('gamma', 10, 2.2)]
RESULT_FILE = BASE / 'results.json'
BOMB_MODE = sys.argv[1:] == ['bomb']
if BOMB_MODE:
    SOURCE = BASE / 'bomb_source'
    CASES = [('bomb', 10, None)]
    RESULT_FILE = BASE / 'results_bomb.json'
RESULT = {'job': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
          'pmt_radius_m': PMT_RADIUS_M, 'cases': {}}
original_gather = generation.gather_photon_deposits
event_deposits = []


def distance_to_segment(points, start, end):
    delta = end - start
    length2 = np.sum(delta * delta, axis=1)
    u = np.clip(np.divide(np.sum((points - start) * delta, axis=1), length2,
                         out=np.zeros_like(length2), where=length2 > 0), 0., 1.)
    return np.linalg.norm(points - (start + u[:, None] * delta), axis=1)


def inspect_gather(process_outputs):
    deposits = original_gather(process_outputs)
    valid = np.isfinite(deposits['t_true']) & np.isfinite(deposits['t_reco'])
    pe = {key: value[valid] for key, value in deposits.items()}
    max_unit_deviation = float(np.max(np.abs(pe['charge'] - 1), initial=0.))
    print('MAX_FINITE_PHOTON_PE_DEVIATION', max_unit_deviation,
          'UNIQUE_NONUNIT', np.unique(pe['charge'][pe['charge'] != 1])[:20], flush=True)
    assert np.allclose(pe['charge'], 1., rtol=0., atol=2e-6), 'Finite-time data-mode photons must deposit one PE each within first-hit clipping epsilon'
    event_deposits.append({
        'detected_before_digitizer': float(pe['charge'].sum()),
        'orphan_segment_pe': int(np.count_nonzero(pe['segment_idx'] < 0)),
        'orphan_particle_pe': int(np.count_nonzero(pe['particle_idx'] < 0)),
        'first_detected_ns': float(pe['t_true'].min()) if pe['charge'].size else None,
        'last_detected_ns': float(pe['t_true'].max()) if pe['charge'].size else None,
        'max_pmt_total_pe_before_digitizer': float(np.bincount(pe['sensor_idx'], weights=pe['charge'], minlength=11096).max()),
        'infinite_time_rows_dropped_by_digitizer': int(np.count_nonzero(~valid)),
        'infinite_time_total_charge_dropped_by_digitizer': float(deposits['charge'][~valid].sum()),
        'max_finite_photon_pe_deviation_from_one': max_unit_deviation,
    })
    assert event_deposits[-1]['orphan_segment_pe'] == 0
    assert event_deposits[-1]['orphan_particle_pe'] == 0
    return deposits


generation.gather_photon_deposits = inspect_gather
for name, n_events, energy in CASES:
    started = time.monotonic()
    root_file = SOURCE / f'{name}.root'
    raw_summary = []
    raw_ancestors = []
    for i in range(n_events):
        raw = _read_event_raw(str(root_file), i)
        seg = raw['segments_raw']
        si = raw['photon_segment_index_raw']
        start = np.stack([seg[f'start_{d}_mm'] for d in 'xyz'], axis=1)[si] / 1000.
        end = np.stack([seg[f'end_{d}_mm'] for d in 'xyz'], axis=1)[si] / 1000.
        # Independent check against G4's mm segment truth catches an accidental
        # scale change in the raw-photon reader before any detector translation.
        gap = distance_to_segment(raw['photon_origins'], start, end)
        primary_rows = sorted((int(tid), int(info['pdg']), float(info['energy']))
                              for tid, info in raw['track_info_dict'].items()
                              if int(info['parent_id']) == 0)
        parents = {int(tid): int(info['parent_id']) for tid, info in raw['track_info_dict'].items()}
        ancestors = {}
        for tid in parents:
            cur, visited = tid, set()
            while parents[cur] != 0:
                assert cur not in visited, 'Cyclic raw Geant4 ancestry'
                visited.add(cur)
                cur = parents[cur]
                assert cur in parents, 'Missing ancestor in raw Geant4 track table'
            ancestors[tid] = cur
        raw_ancestors.append(ancestors)
        raw_summary.append({
            'source_event_idx': i,
            'emitted_photons': len(si),
            'max_photon_distance_from_g4_segment_m': float(gap.max(initial=0.)),
            'max_emission_radius_m': float(np.linalg.norm(raw['photon_origins'], axis=1).max(initial=0.)),
            'primary_ke_MeV': raw['primary_energy'],
            'primary_track_ids': [r[0] for r in primary_rows],
            'primary_pdgs': [r[1] for r in primary_rows],
            'primary_kinetic_energies_MeV': [r[2] for r in primary_rows],
            'total_primary_kinetic_energy_MeV': sum(r[2] for r in primary_rows),
        })
        assert gap.max(initial=0.) < 0.005
    del raw, start, end, gap
    config = {'name': f'wc_actual_audit_{name}', 'lucid_options': {
        'apply_smearing': True, 'apply_translation': True, 'apply_rotation': False}}
    if name == 'gamma':
        config['selection'] = {'mode': 'min_physics_hits', 'n': 3}
    if BOMB_MODE:
        config = json.loads((SOURCE / 'original_config.json').read_text())
    event_deposits.clear()
    output_dir = BASE / name
    _run_lucid(root_file=root_file, output_dir=output_dir, config=config,
               file_index=0, n_events=n_events, master_seed=918273,
               job_id=2718, detector='SK_WAND')
    rows = []
    with ExitStack() as stack:
        files = {kind: stack.enter_context(h5py.File(output_dir / kind / f'wc_{kind}_0000.h5', 'r'))
                 for kind in ('sensor', 'hits', 'step', 'labl')}
        keys = sorted(k for k in files['sensor'] if k.startswith('event_'))
        assert all(sorted(k for k in f if k.startswith('event_')) == keys for f in files.values())
        sensor_positions = files['sensor']['config/sensor_positions'][:]
        for ev in keys:
            s, h, g, l = (files[k][ev] for k in ('sensor', 'hits', 'step', 'labl'))
            seg_hits = g['sensor_hits']
            sid, q = s['sensor_idx'][:].astype(int), s['PE'][:]
            nd = len(sid)
            assert np.all(np.isfinite(q)) and np.all(q > 0)
            source_idx = int(s.attrs['source_event_idx'])
            row = dict(raw_summary[source_idx], **event_deposits[source_idx])
            if BOMB_MODE:
                pi = l['per_interaction']
                assert len(pi['n_primaries']) == 1, 'A bomb has one physical interaction'
                assert int(pi['n_primaries'][0]) == len(row['primary_track_ids'])
                assert np.array_equal(pi['primary_track_ids_data'][:], row['primary_track_ids'])
                assert np.array_equal(pi['primary_pdgs_data'][:], row['primary_pdgs'])
                assert np.allclose(pi['primary_energies_data'][:], row['primary_kinetic_energies_MeV'], rtol=2e-7)
                tr = l['per_track']
                expected_ancestors = [raw_ancestors[source_idx][int(tid)] for tid in tr['track_id'][:]]
                assert np.array_equal(tr['ancestor'][:], expected_ancestors), 'Persisted ancestry disagrees with raw G4 parents'
                assert np.all(tr['interaction'][:] == 0)
                assert np.all(l['per_particle/interaction_idx'][:] == 0)
            hit_counts = np.bincount(h['digit_idx'][:], weights=h['PE'][:], minlength=nd)
            phys = h['emission_process'][:] == 0
            dark = h['emission_process'][:] == 2
            hit_physics_counts = np.bincount(h['digit_idx'][:][phys], weights=h['PE'][:][phys], minlength=nd)
            seg_counts = np.bincount(seg_hits['digit_idx'][:], weights=seg_hits['PE'][:], minlength=nd)
            assert np.allclose(hit_physics_counts, seg_counts, rtol=2e-6, atol=2e-5), 'Particle and segment physics charge mismatch'
            for decomp in (h, seg_hits):
                di = decomp['digit_idx'][:]
                assert np.all((di >= 0) & (di < nd))
                assert np.array_equal(decomp['sensor_idx'][:], sid[di])
                assert np.allclose(decomp['PE'][:], np.rint(decomp['PE'][:]), rtol=2e-6, atol=2e-5)
            assert np.all(phys | dark), 'Unexpected water scintillation'
            if BOMB_MODE:
                # A segment's track identifies which categorized particle owns
                # its light. Check each (particle, digit) independently so light
                # cannot swap between primaries while conserving event totals.
                segment_idx = seg_hits['segment_idx'][:].astype(int)
                owning_tracks = g['track_idx'][:][segment_idx]
                owning_particles = l['per_track/particle_idx'][:][owning_tracks]
                n_particles = len(l['per_particle/category'])
                assert np.all((owning_particles >= 0) & (owning_particles < n_particles))
                step_keys = owning_particles * nd + seg_hits['digit_idx'][:]
                hit_keys = h['particle_idx'][:][phys] * nd + h['digit_idx'][:][phys]
                step_per_particle_digit = np.bincount(step_keys, weights=seg_hits['PE'][:], minlength=n_particles * nd)
                hit_per_particle_digit = np.bincount(hit_keys, weights=h['PE'][:][phys], minlength=n_particles * nd)
                assert np.allclose(step_per_particle_digit, hit_per_particle_digit, rtol=2e-6, atol=2e-5)
                row['max_per_particle_digit_charge_difference'] = float(np.max(np.abs(step_per_particle_digit - hit_per_particle_digit), initial=0.))
            # The time attached to a segment contribution must allow at least
            # straight-line travel from that segment to the PMT sphere.
            sh_idx = seg_hits['segment_idx'][:].astype(int)
            sh_valid = (sh_idx >= 0) & (seg_hits['emission_process'][:] == 0)
            idx = sh_idx[sh_valid]
            segment_start = np.stack([g[f'start_{d}'][:] for d in 'xyz'], axis=1)[idx]
            segment_end = np.stack([g[f'end_{d}'][:] for d in 'xyz'], axis=1)[idx]
            pmt = sensor_positions[seg_hits['sensor_idx'][:][sh_valid]]
            flight_distance = np.maximum(0., distance_to_segment(pmt, segment_start, segment_end) - PMT_RADIUS_M)
            causal_slack = seg_hits['T'][:][sh_valid] - g['time'][:][idx] - flight_distance / (0.299792458/1.33)
            assert np.min(causal_slack, initial=0.) >= -0.02, f'Unphysical early arrival {causal_slack.min()} ns'
            t0 = float(l['per_event/t0'][()])
            vertex = [float(l[f'per_interaction/vertex_{d}'][0]) for d in 'xyz']
            row.update({
                'vertex_m': vertex, 'digit_count': nd,
                'retained_physics_pe': float(h['PE'][:][phys].sum()),
                'retained_dark_pe': float(h['PE'][:][dark].sum()),
                'total_reconstructed_charge_pe': float(q.sum()),
                'max_digit_truth_pe': float(hit_counts.max(initial=0.)),
                'max_digit_charge_pe': float(q.max(initial=0.)),
                'max_particle_vs_segment_charge_difference': float(np.max(np.abs(hit_physics_counts-seg_counts), initial=0.)),
                'min_segment_causal_slack_ns': float(causal_slack.min()) if causal_slack.size else None,
                'physics_first_time_ns': float(h['T'][:][phys].min() - t0) if phys.any() else None,
                'physics_last_time_ns': float(h['T'][:][phys].max() - t0) if phys.any() else None,
                'reco_pe_per_primary_MeV': float(q.sum()/(energy if energy is not None else row['total_primary_kinetic_energy_MeV'])),
                'contained': bool(l['per_event/contained'][()]),
            })
            pw = l['per_window']
            off = pw['digit_offsets'][:]
            assert off[0] == 0 and off[-1] == nd and np.all(np.diff(off) >= 0)
            for k, (a, b) in enumerate(zip(off[:-1], off[1:])):
                ts = s['T'][a:b]
                assert np.all(ts >= pw['window_start'][k]) and np.all(ts <= pw['window_end'][k])
            rows.append(row)
    summary = {
        'mean_emitted_photons': float(np.mean([r['emitted_photons'] for r in rows])),
        'mean_detected_before_digitizer': float(np.mean([r['detected_before_digitizer'] for r in rows])),
        'mean_retained_physics_pe': float(np.mean([r['retained_physics_pe'] for r in rows])),
        'mean_reconstructed_charge_pe': float(np.mean([r['total_reconstructed_charge_pe'] for r in rows])),
        'mean_reco_pe_per_primary_MeV': float(np.mean([r['reco_pe_per_primary_MeV'] for r in rows])),
        'max_digit_truth_pe': max((r['max_digit_truth_pe'] for r in rows), default=0.),
        'max_pmt_total_pe_before_digitizer': max((r['max_pmt_total_pe_before_digitizer'] for r in rows), default=0.),
    }
    RESULT['cases'][name] = {'events_requested': n_events, 'events_kept': len(rows),
                              'elapsed_s': time.monotonic() - started,
                              'summary': summary, 'events': rows}
    RESULT_FILE.write_text(json.dumps(RESULT, indent=2) + '\n')
    print('AUDIT_CASE', name, json.dumps(RESULT['cases'][name]), flush=True)
print('ALL_END_TO_END_CHECKS_PASSED', flush=True)
