"""Full SK_WAND DATA simulation for a real translated muon, with source-bounds veto."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation import setup_event_simulator
from lucid.sources.event_builder import _trace_event_bucketed

BASE = Path(__file__).resolve().parent


def main():
    assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
    sample = np.load(BASE / 'mu1000_worst_placement.npz')
    origins, directions = sample['origins_m'], sample['directions']
    times, wavelengths = sample['times_ns'], sample['wavelengths_nm']
    segments, outside = sample['segment_ids'], sample['outside_mask']
    sim = setup_event_simulator(
        'config/SK_WAND_geom_config.json', 0, K=12, is_data=True,
        temperature=0.0, charge_resolution=None,
        physics_config='config/SK_WAND_physics_config.json',
        default_detector_params=True, hit_mode='per_segment', deposit_leg_bound=True,
    )
    detector = sim.det_geom.detector
    checked_inside = np.asarray(detector.bounds_check(jnp.asarray(origins)))
    assert np.array_equal(checked_inside, ~outside)

    # N is used only by the actual kernel's arange<N intensity mask.
    # Broadcast it to the photon axis in BOTH runs, then zero its entries outside.
    # Input ordering, bucket padding, keys, geometry and optical arrays stay identical.
    def wrapper(veto):
        def run(track, key, data):
            data = dict(data)
            n = data['N']
            source_inside = detector.bounds_check(data['photon_origins'] / 100.0)
            counts = jnp.full((data['photon_origins'].shape[0],), n, dtype=jnp.int32)
            data['N'] = jnp.where(source_inside | (not veto), counts, 0)
            return sim(track, key, data)
        return run

    key = jax.random.PRNGKey(80237)
    summaries, inside_records = [], []
    for veto in (False, True):
        tstart = time.monotonic()
        out = _trace_event_bucketed(
            wrapper(veto), origins, directions, times, wavelengths, segments,
            sim.det_geom.num_sensors, (32768,), key,
        )
        pe, tt, tr, weight, ttrue, treco, sensor, seg, gid = out
        keep = weight > 0
        from_outside = outside[gid]
        valid_inside = keep & ~from_outside
        inside_records.append((gid[valid_inside].copy(), weight[valid_inside].copy(), ttrue[valid_inside].copy(), sensor[valid_inside].copy()))
        summaries.append({'veto_exterior': veto, 'total_PE': float(pe.sum()), 'outside_source_PE': float(weight[keep & from_outside].sum()), 'detected_records': int(keep.sum()), 'sensors_hit': int((pe > 0).sum()), 'runtime_s': time.monotonic() - tstart, 'pe': pe.copy(), 'time': tt.copy()})
        print(json.dumps({k:v for k,v in summaries[-1].items() if k not in ('pe','time')}), flush=True)
        del out, weight, ttrue, treco, sensor, seg, gid
    all_run, veto_run = summaries
    unchanged_inside = all(np.array_equal(a,b) for a,b in zip(*inside_records))
    jointly_hit = (all_run['pe'] > 0) & (veto_run['pe'] > 0)
    dt = all_run['time'][jointly_hit] - veto_run['time'][jointly_hit]
    result = {
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'partition': os.environ['SLURM_JOB_PARTITION'],
        'source_energy_MeV': 1000, 'source_event': int(sample['event']), 'placement': int(sample['placement']),
        'source_photons': len(origins), 'outside_source_photons': int(outside.sum()), 'vertex_m': sample['vertex_m'].tolist(),
        'all_source_PE': all_run['total_PE'], 'veto_exterior_PE': veto_run['total_PE'], 'outside_source_PE': all_run['outside_source_PE'],
        'relative_extra_PE_vs_veto': (all_run['total_PE'] / veto_run['total_PE']) - 1,
        'inside_records_bitwise_identical': unchanged_inside,
        'all_sensors_hit': all_run['sensors_hit'], 'veto_sensors_hit': veto_run['sensors_hit'],
        'sensors_lost_after_veto': int(np.sum((all_run['pe'] > 0) & (veto_run['pe'] == 0))),
        'jointly_hit_sensors': int(jointly_hit.sum()), 'changed_first_times': int(np.count_nonzero(dt)),
        'largest_earlier_first_time_from_exterior_ns': float(-dt.min(initial=0.)),
    }
    assert unchanged_inside, result
    assert veto_run['outside_source_PE'] == 0, result
    np.savez_compressed(BASE / 'actual_source_mask_sensor_arrays.npz', all_pe=all_run['pe'], veto_pe=veto_run['pe'], all_time=all_run['time'], veto_time=veto_run['time'])
    print(json.dumps(result, indent=2), flush=True)
    (BASE / 'actual_source_mask_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
