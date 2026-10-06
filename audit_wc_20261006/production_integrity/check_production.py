"""Independent production-boundary checks; run only in Slurm milano/roma."""
import copy
import json
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from lucid.production import run_job
from lucid.sources.event_builder import (
    _derive_views_from_segments, derive_particle_idx_per_track,
    derive_track_ancestor_and_interaction,
)
from lucid.sources.event_generation import _drop_unwritten
from lucid.sources.seed_utils import derive_event_keys, derive_subprocess_seeds
from lucid.sources.writer import (
    _compute_contained, build_interaction_metadata, sample_translation_vector,
    save_sensor_event, save_hits_event, save_step_event, save_labl_event,
)

repo = Path(__file__).resolve().parents[2]
report = {}

# Exercise actual production forwarding, replacing only expensive simulation.
cfg = json.loads((repo / 'lucid/production/configs/GeV/01_pbomb.json').read_text())
geom = SimpleNamespace(sensor_points=np.zeros((11096, 3)),
                       detector_type='cylinder', medium=SimpleNamespace(material='water'))
with patch('lucid.geometry.detector_geometry.DetectorGeometry.from_config', return_value=geom), \
     patch('lucid.simulation.setup_event_simulator', return_value='sim') as setup, \
     patch('lucid.sources.event_generation.generate_events_from_photonsim_particles', return_value=[]) as generate:
    run_job._run_lucid(root_file=Path('/tmp/audit_unused.root'), output_dir=Path('/tmp'),
                      config=cfg, file_index=0, n_events=7, master_seed=19,
                      job_id=1, detector='SK_WAND')
    kw = setup.call_args.kwargs
    assert kw['is_data'] and kw['temperature'] == 0 and kw['K'] == 12
    assert kw['charge_resolution'] is None and kw['deposit_leg_bound'] is True
    assert kw['hit_mode'] == 'per_segment'
    kw = generate.call_args.kwargs
    assert kw['digitizer'] == {'model': 'ski'} and kw['trigger']['n_thr'] == 25
    assert kw['min_physics_hits'] is None and kw['apply_smearing']
    assert kw['apply_translation'] and not kw['apply_rotation']
    report['production_forwarding'] = 'SK_WAND data, ski, threshold25, leg bound, one digitizer smearing'

# Independent hierarchy: mu -> decay e; pi0 -> gamma -> conversion e.
tracks = {}
for tid, pid, pdg, proc in [(1, 0, 13, ''), (2, 1, 11, 'Decay'),
                            (3, 0, 111, ''), (4, 3, 22, 'Decay'), (5, 4, 11, 'conv')]:
    tracks[tid] = dict(track_id=tid, parent_id=pid, pdg=pdg, energy=100.,
                       creator_process=proc, position=np.zeros(3), direction=np.array([1.,0.,0.]))
seg = dict(n_segments=6, track_id=np.array([1,1,2,3,4,5]),
           n_cherenkov=np.array([3,2,4,0,0,6]), time=np.arange(6.),
           edep=np.ones(6), beta_start=np.ones(6),
           dir_x=np.ones(6), dir_y=np.zeros(6), dir_z=np.zeros(6))
for pre in ('start','end'):
    for axis in 'xyz':
        seg[f'{pre}_{axis}_mm'] = np.zeros(6)
seg['end_x_mm'][2] = 20000.  # only decay electron exits R=16 m.
raw = dict(segments_raw=seg, track_info_dict=tracks, photon_segment_index_raw=np.array([0,1,2,5]),
           photon_origins=np.zeros((4,3)), photon_directions=np.tile([1.,0.,0.], (4,1)),
           photon_times=np.arange(4.), photon_wavelengths=np.full(4,400.),
           primary_energy=100., rootracker_entry_id=-1, neutrino_pdg=0, neutrino_energy_MeV=0.)
view = _derive_views_from_segments(raw)
assert [p['genealogy'] for p in view['particles']] == [[1], [1,2], [3,4]]
meta = build_interaction_metadata(view, t0=-100.125, vertex_xyz=[0,0,0], source_type_code=0)
ev = dict(view, source_event_idx=0, primary_to_interaction={1:0,3:0}, interaction_metadata=[meta])
np.testing.assert_array_equal(derive_particle_idx_per_track(ev), [0,1,-1,2,2])
ancestor, interaction = derive_track_ancestor_and_interaction(ev)
np.testing.assert_array_equal(ancestor, [1,1,3,3,3])
np.testing.assert_array_equal(interaction, [0,0,0,0,0])
cont = _compute_contained(ev, dict(type='cylinder', radius=16., height=36.))
np.testing.assert_array_equal(cont['per_particle'], [True, False, True])
assert not cont['overall']
ev.update(contained=cont['overall'], contained_per_segment=cont['per_segment'],
          contained_per_particle=cont['per_particle'], contained_per_interaction=cont['per_interaction'])

# Sparse production writers must preserve repeated sensors, negative times and PE.
ev['sensor_digits'] = dict(sensor_idx=np.array([11095,0,0]), PE=np.array([.5,2.,3.]),
                           T=np.array([-100.125,0.,1000.375]))
ev['hits_sparse'] = dict(particle_idx=np.array([0,1,2]), sensor_idx=np.array([11095,0,0]),
                         PE=np.array([.5,2.,3.]), T=np.array([-100.125,0.,1000.375]),
                         T_reco=np.array([-100.125,0.,1000.375]), digit_idx=np.arange(3),
                         emission_process=np.zeros(3,dtype=np.int8))
ev['segment_sensor_hits'] = dict(ev['hits_sparse'], segment_idx=np.array([0,2,5]))
_drop_unwritten(ev)
with tempfile.TemporaryDirectory() as td:
    for name, writer in [('sensor',save_sensor_event), ('hits',save_hits_event),
                         ('step',save_step_event), ('labl',save_labl_event)]:
        with h5py.File(Path(td)/f'{name}.h5', 'w') as f:
            writer(f, ev, 0)
            g=f['event_000']
            if name in ('sensor','hits'):
                np.testing.assert_array_equal(g['PE'][:], [.5,2.,3.])
                np.testing.assert_array_equal(g['T'][:], [-100.125,0.,1000.375])
                np.testing.assert_array_equal(g['sensor_idx'][:], [11095,0,0])
            if name=='step':
                np.testing.assert_array_equal(g['track_idx'][:], [0,0,1,2,3,4])
                assert g['end_x'][2] == 20.
            if name=='labl':
                np.testing.assert_array_equal(g['per_track/ancestor'][:], [1,1,3,3,3])
                np.testing.assert_array_equal(g['per_interaction/primary_pdgs_data'][:], [13,111])
report['ancestry_containment_writer'] = 'muon/decay e, pi0/gamma/conversion e; sparse rows/times/PE preserved'

# Draw hierarchy uniqueness and cylinder-volume distribution, independently expected moments.
key_rows=[]
for job in (1,2):
    for event in (0,1):
        for vertex in (0,1):
            keys=derive_event_keys(17,job,event,vertex)
            key_rows.append(tuple(np.asarray(keys['sim_key']).tolist()))
assert len(set(key_rows))==8
assert derive_subprocess_seeds(17,1) != derive_subprocess_seeds(17,2)
rng=np.random.default_rng(42)
points=np.array([sample_translation_vector(dict(type='cylinder',radius=16.,height=36.),rng)
                 for _ in range(30000)])
r2=np.sum(points[:,:2]**2,axis=1)/(16.*.9)**2
z=points[:,2]/(18.*.9)
assert abs(r2.mean()-.5)<.01 and abs((z*z).mean()-1/3)<.01
assert np.max(r2)<=1 and np.max(abs(z))<=1
report['vertex_moments'] = dict(mean_r2_fraction=float(r2.mean()),mean_z2_fraction=float((z*z).mean()))
report['distinct_seed_streams'] = len(set(key_rows))
print(json.dumps(report,indent=2))
runpy.run_path(str(Path(__file__).with_name('repro_pileup_group_ids.py')), run_name='__main__')
