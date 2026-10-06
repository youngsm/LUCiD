"""Screen retained Cherenkov histories for excursions beyond detector water.

Run only on a milano/roma allocation. An exposure flag is conservative: a
segment crossing a surface is marked from its START time. It establishes
possible material sensitivity, not a corrected detector prediction.
"""
from pathlib import Path
import json
import os

assert os.environ.get('SLURM_JOB_PARTITION') in {'milano', 'roma'}
import numpy as np
from lucid.sources.root_reader import _read_event_raw
from lucid.sources.writer import sample_translation_vector

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
AUDIT = REPO / 'audit_wc_20261006'
SOURCE = AUDIT / 'end_to_end/bomb_source/bomb.root'
VERTICES = json.loads((AUDIT / 'end_to_end/results_bomb.json').read_text())['cases']['bomb']['events']
ID = {'type': 'cylinder', 'radius': 16.96228024518598, 'height': 36.37842222222}
SCREENS = {'id': (ID['radius'], ID['height']/2),
           'nominal_tank': (19.65, 20.7),
           'inset_tank_sensitivity': (19.40, 20.45)}
EXTRA_PLACEMENTS = 16

def outside(pos, radius, halfheight):
    return ((pos[:, 0]**2 + pos[:, 1]**2) > radius**2) | (np.abs(pos[:, 2]) > halfheight)

report = {'job': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
          'source_root': str(SOURCE), 'independent_g4_events': 10,
          'extra_random_translations_per_event': EXTRA_PLACEMENTS,
          'screens_m': SCREENS, 'rows': [], 'witnesses': []}

for event_idx in range(10):
    raw = _read_event_raw(str(SOURCE), event_idx)
    seg = raw['segments_raw']
    tids = np.array(sorted(raw['track_info_dict']), dtype=np.int64)
    entries = [raw['track_info_dict'][int(t)] for t in tids]
    parents = np.array([x['parent_id'] for x in entries], dtype=np.int64)
    parent_idx = np.where(parents > 0, np.searchsorted(tids, parents), -1)
    assert np.all(parent_idx < np.arange(len(tids)))
    nonroot = parent_idx >= 0
    assert np.array_equal(tids[parent_idx[nonroot]], parents[nonroot])
    birth_pos = np.stack([x['position'] for x in entries])
    birth_time = np.array([x['time'] for x in entries])
    seg_track_idx = np.searchsorted(tids, seg['track_id'])
    assert np.array_equal(tids[seg_track_idx], seg['track_id'])
    starts = np.stack([seg[f'start_{d}_mm'] for d in 'xyz'], axis=1) / 1000.
    ends = np.stack([seg[f'end_{d}_mm'] for d in 'xyz'], axis=1) / 1000.
    seg_time = seg['time']
    photon_times = raw['photon_times']
    photon_track_idx = seg_track_idx[raw['photon_segment_index_raw']]
    production_vertex = np.asarray(VERTICES[event_idx]['vertex_m'])
    vertices = [production_vertex] + [sample_translation_vector(ID, np.random.default_rng([128731, event_idx, rep]))
                                      for rep in range(EXTRA_PLACEMENTS)]
    for placement, vertex in enumerate(vertices):
        ph_pos = raw['photon_origins'] + vertex
        st, en, born = starts + vertex, ends + vertex, birth_pos + vertex
        inside_id = ~outside(ph_pos, *SCREENS['id'])
        kept_10us = inside_id & (photon_times <= 1e4)
        kept_100us = inside_id & (photon_times <= 1e5)
        row = {'event_idx': event_idx, 'placement': placement,
               'is_original_production_vertex': placement == 0, 'vertex_m': vertex.tolist(),
               'emitted_photons': len(ph_pos), 'inside_id_photons': int(inside_id.sum()),
               'inside_id_photons_10us': int(kept_10us.sum()),
               'inside_id_photons_100us': int(kept_100us.sum())}
        for screen, dims in SCREENS.items():
            own_exit = np.where(outside(born, *dims), birth_time, np.inf)
            outside_segment = outside(st, *dims) | outside(en, *dims)
            np.minimum.at(own_exit, seg_track_idx[outside_segment], seg_time[outside_segment])
            first_exit = own_exit.copy()
            cause_idx = np.arange(len(tids))
            for ti, pi in enumerate(parent_idx):
                if pi >= 0 and first_exit[pi] < first_exit[ti]:
                    first_exit[ti] = first_exit[pi]
                    cause_idx[ti] = cause_idx[pi]
            exposed = photon_times >= first_exit[photon_track_idx]
            flag = exposed & kept_10us
            row[f'{screen}_history_exposed_10us_photons'] = int(flag.sum())
            row[f'{screen}_history_exposed_100us_photons'] = int((exposed & kept_100us).sum())
            row[f'{screen}_history_exposed_all_photons'] = int((exposed & inside_id).sum())
            row[f'{screen}_tracks_ever_outside'] = int(np.isfinite(own_exit).sum())
            if screen != 'id' and flag.any():
                chosen = np.flatnonzero(flag)
                np.savez_compressed(BASE / f'exposed_{screen}_ev{event_idx}_v{placement}.npz',
                                    photon_idx=chosen, vertex=vertex)
                for ph in chosen[:3]:
                    ti = photon_track_idx[ph]
                    ci = cause_idx[ti]
                    report['witnesses'].append({
                        'screen': screen, 'event_idx': event_idx, 'placement': placement,
                        'photon_idx': int(ph), 'position_m': ph_pos[ph].tolist(),
                        'photon_time_ns': float(photon_times[ph]),
                        'emitter_track_id': int(tids[ti]), 'emitter_pdg': int(entries[ti]['pdg']),
                        'first_external_ancestor_track_id': int(tids[ci]),
                        'first_external_ancestor_pdg': int(entries[ci]['pdg']),
                        'first_external_ancestor_time_ns_upper_screen': float(first_exit[ti]),
                    })
        report['rows'].append(row)
    print('EVENT_SCREENED', event_idx, 'photons', len(photon_times), flush=True)

report['summary'] = {}
for group, rows in [('actual_production_vertices', [r for r in report['rows'] if r['placement'] == 0]),
                    ('all_translated_placements', report['rows'])]:
    denom = sum(r['inside_id_photons_10us'] for r in rows)
    out = {'placements': len(rows), 'inside_id_photons_10us': denom}
    for screen in SCREENS:
        counts = [r[f'{screen}_history_exposed_10us_photons'] for r in rows]
        fractions = [n / max(1, r['inside_id_photons_10us']) for n, r in zip(counts, rows)]
        out[screen] = {'exposed_photons_10us': sum(counts),
                       'fraction_10us': sum(counts)/max(1, denom),
                       'placements_with_exposed_photons': sum(n > 0 for n in counts),
                       'max_placement_fraction_10us': max(fractions),
                       'max_placement_exposed_photons_10us': max(counts)}
    report['summary'][group] = out
(BASE / 'history_census.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report['summary'], indent=2), flush=True)
print('Witnesses:', len(report['witnesses']), flush=True)
