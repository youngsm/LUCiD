"""Controlled injected photons in real ROOT TTrees through the production writer.

The primary creator-process field is scalar string for uproot writer compatibility.
It tests transport/readout/storage; true Geant4 vector<string> schema is separate.
"""
from pathlib import Path
import json
import awkward as ak
import h5py
import numpy as np
import uproot
from lucid.geometry import generate_detector
from lucid.production.run_job import _run_lucid
from lucid.sources.root_reader import _read_event_raw

out = Path('audit/wc_data_review_20261004/end_to_end')
root_path = out / 'injected_primary.root'
det = generate_detector('config/SK_WAND_geom_config.json')
centers = np.asarray(det.all_points)
pick = np.linspace(0, len(centers) - 1, 64, dtype=int)
origin_m = np.asarray([5., -2., 3.])
origins = np.repeat(origin_m[None, :], 8192, axis=0)
directions = np.repeat(centers[pick] - origin_m, 128, axis=0)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
event = {'EventID': np.asarray([0], np.int32), 'NOpticalPhotons': np.asarray([8192], np.int32),
         'NSegments': np.asarray([1], np.int32), 'PrimaryEnergy': np.asarray([1000.]),
         'RooTrackerEntryID': np.asarray([-1], np.int32), 'IncomingNuPdg': np.asarray([0], np.int32),
         'IncomingNuKE': np.asarray([0.]), 'TrackInfo_CreatorProcess': ['Primary']}
for name, val in {'TrackID': 1, 'ParentTrackID': 0, 'PDG': 11}.items():
    event['TrackInfo_' + name] = ak.Array([[val]])
for name, val in {'Energy': 1000., 'Time': 0., 'PosX': 5000., 'PosY': -2000., 'PosZ': 3000.,
                  'DirX': 0., 'DirY': 0., 'DirZ': 1.}.items():
    event['TrackInfo_' + name] = ak.Array([[val]])
for name, val in {'StartX': 5000., 'StartY': -2000., 'StartZ': 3000.,
                  'EndX': 5000., 'EndY': -2000., 'EndZ': 3001.,
                  'DirX': 0., 'DirY': 0., 'DirZ': 1., 'Edep': 1., 'Time': 0., 'BetaStart': .99}.items():
    event['Segment_' + name] = ak.Array([[val]])
event['Segment_TrackID'] = ak.Array([[1]])
event['Segment_NCherenkov'] = ak.Array([[8192]])
event['Photon_SegmentIndex'] = ak.Array([np.zeros(8192, np.int32)])
raw = {'EventID': np.asarray([0], np.int32), 'ChunkStartID': np.asarray([0], np.int64)}
for i, axis in enumerate('XYZ'):
    raw['PhotonPos' + axis] = ak.Array([origins[:, i] * 1000.])
    raw['PhotonDir' + axis] = ak.Array([directions[:, i]])
raw['PhotonTime'] = ak.Array([np.full(8192, 7.)])
raw['PhotonWavelength'] = ak.Array([np.full(8192, 400.)])
with uproot.recreate(root_path) as f:
    for name, data in [('OpticalPhotons', event), ('OpticalPhotonsRaw', raw)]:
        schema = {}
        for key, value in data.items():
            if key == 'TrackInfo_CreatorProcess':
                schema[key] = 'string'
            elif isinstance(value, ak.Array):
                schema[key] = 'var * int32' if key in ('TrackInfo_TrackID', 'TrackInfo_ParentTrackID', 'TrackInfo_PDG', 'Segment_TrackID', 'Segment_NCherenkov', 'Photon_SegmentIndex') else 'var * float64'
            else:
                schema[key] = value.dtype
        f.mktree(name, schema).extend(data)
read = _read_event_raw(str(root_path), 0)
assert np.allclose(read['photon_origins'], origins)
print('Injected fixture meter boundary validated:', read['photon_origins'][0].tolist(), flush=True)
_run_lucid(root_file=root_path, output_dir=out / 'writer_output',
           config={'name': 'wc_audit_injected_primary', 'lucid_options': {
               'apply_smearing': False, 'apply_translation': False, 'pad_size_buckets': [8192]}},
           file_index=0, n_events=1, master_seed=984, job_id=1, detector='SK_WAND')
for path in sorted((out / 'writer_output').rglob('*.h5')):
    with h5py.File(path) as f:
        print('Written', path, 'attrs', dict(f.attrs), 'keys', list(f.keys()), flush=True)
