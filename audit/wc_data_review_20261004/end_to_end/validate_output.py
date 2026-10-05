from pathlib import Path
import json
import h5py
import numpy as np
from lucid.sources.root_reader import _read_event_raw

lane = Path('audit/wc_data_review_20261004/end_to_end')
results = {}
for directory, source in [('writer_output', lane / 'injected_primary.root'),
                           ('true_writer_output', Path('audit/wc_data_review_20261004/geant4_physics/e100tiny.root'))]:
    out = lane / directory
    with h5py.File(out / 'sensor/wc_sensor_0000.h5') as sensors, \
         h5py.File(out / 'hits/wc_hits_0000.h5') as hits, \
         h5py.File(out / 'step/wc_step_0000.h5') as steps, \
         h5py.File(out / 'labl/wc_labl_0000.h5') as labels:
        event_results = []
        for name in sorted(k for k in sensors if k.startswith('event_')):
            i = int(name.split('_')[1])
            s, h, g, l = sensors[name], hits[name], steps[name], labels[name]
            pe = s['PE'][:]
            rebuilt = np.bincount(h['digit_idx'][:], weights=h['PE'][:], minlength=len(pe))
            diff = float(np.max(np.abs(rebuilt - pe)))
            print(directory, name, "decomposition_diff", diff, "sensor_total", float(pe.sum()), "hits_total", float(h["PE"][:].sum()), "max_row", int(np.argmax(np.abs(rebuilt-pe))), "max_sensor", float(pe[np.argmax(np.abs(rebuilt-pe))]), "max_hits", float(rebuilt[np.argmax(np.abs(rebuilt-pe))]), flush=True)
            if directory == "writer_output":
                assert diff < 2e-5
            assert np.all(h['sensor_idx'][:] == s['sensor_idx'][:][h['digit_idx'][:]])
            real = h['emission_process'][:] == 0
            dark = h['emission_process'][:] == 2
            rebuilt_real = np.bincount(h['digit_idx'][:][real], weights=h['PE'][:][real], minlength=len(pe))
            rebuilt_steps = np.bincount(g['sensor_hits/digit_idx'][:], weights=g['sensor_hits/PE'][:], minlength=len(pe))
            assert np.max(np.abs(rebuilt_real - rebuilt_steps)) < 2e-5
            offsets = l['per_window/digit_offsets'][:]
            for j in range(len(offsets) - 1):
                t = s['T'][offsets[j]:offsets[j + 1]]
                assert np.all((t >= l['per_window/window_start'][j]) & (t <= l['per_window/window_end'][j]))
            raw = _read_event_raw(str(source), i)
            raw_primary = next(v for v in raw['track_info_dict'].values() if v['parent_id'] == 0)
            v = np.asarray([l['per_interaction/vertex_' + axis][0] for axis in 'xyz'])
            if directory == "true_writer_output":
                assert np.array_equal(l["per_interaction/primary_pdgs_data"][:], [11])
                assert np.array_equal(l["per_interaction/primary_energies_data"][:], [100.])
                assert np.array_equal(v, np.asarray([g["start_" + axis][0] for axis in "xyz"]))
            event_results.append({'event': i, 'digits': len(pe), 'total_pe': float(pe.sum()),
                'physics_pe': float(h['PE'][:][real].sum()), 'dark_pe': float(h['PE'][:][dark].sum()),
                'max_reco_vs_truth_digit_charge_difference_pe': diff,
                'source_primary_position_m': raw_primary['position'].tolist(),
                'written_vertex_m': v.tolist(), 'segments': len(g['start_x']),
                'first_segment_start_m': [float(g['start_' + axis][0]) for axis in 'xyz'],
                'written_primary_energy_MeV': l['per_interaction/primary_energies_data'][:].tolist(),
                'written_primary_pdg': l['per_interaction/primary_pdgs_data'][:].tolist()})
        results[directory] = event_results
print(json.dumps(results, indent=2), flush=True)
(lane / 'writer_validation_results.json').write_text(json.dumps(results, indent=2))
