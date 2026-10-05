from pathlib import Path
import h5py
import json
import numpy as np
from lucid.production.run_job import _run_lucid
from lucid.sources.root_reader import _read_event_raw
from lucid.production.verify_output import verify_batch

root = Path('audit/wc_data_review_20261004/geant4_physics/e100tiny.root')
out = Path('audit/wc_data_review_20261004/end_to_end/true_writer_output')
for i in range(2):
    r = _read_event_raw(str(root), i)
    print('G4 read', i, 'photons', len(r['photon_times']), 'primary_energy', r['primary_energy'], 'tracks', len(r['track_info_dict']), flush=True)
_run_lucid(root_file=root, output_dir=out, config={'name': 'wc_audit_true_e100',
    'lucid_options': {'pad_size_buckets': [8192]}}, file_index=0, n_events=2,
    master_seed=984, job_id=1, detector='SK_WAND')
print('Structural verifier', verify_batch(out, 0, expected_dataset_name='wc_audit_true_e100'), flush=True)
