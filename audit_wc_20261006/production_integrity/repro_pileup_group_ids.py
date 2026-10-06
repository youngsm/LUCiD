"""Lower-impact truth regression: pile-up silently discards segment grouping.

Run only on milano/roma. This is not a loss of sensor photons.
"""
import tempfile
from pathlib import Path

import h5py
import numpy as np

from lucid.sources.event_generation import _merge_pileup_streams
from lucid.sources.writer import save_step_event
from lucid.simulation.digitizer import resolve_model_config


def stream(tid):
    seg = {k: np.zeros(2) for k in ('start_x','start_y','start_z','end_x','end_y','end_z',
                                   'dir_x','dir_y','dir_z','edep','time','beta_start','n_cherenkov')}
    seg.update(n_segments=2, group_id=np.array([0,0], dtype=np.int32))
    return dict(particles=[{'genealogy':[tid]}],
                meaningful_tracks={tid:dict(track_id=tid, parent_id=0, n_segments=2)},
                segments=seg, deposits={}, t0=0.,
                interaction_meta={'primary_track_ids':[tid]})


merged = _merge_pileup_streams([stream(1),stream(2)], n_sensors=1,
    apply_smearing=False, digitizer_model=resolve_model_config('basic'),
    digi_rng=np.random.default_rng(0), detector_bounds={'type':'cylinder','radius':16.,'height':36.})
merged['source_event_idx']=0
with tempfile.TemporaryDirectory() as td:
    with h5py.File(Path(td)/'step.h5','w') as f:
        save_step_event(f,merged,0)
        actual=f['event_000/group_id'][:]
expected=np.array([0,0,1,1])
print('Expected group IDs:',expected.tolist())
print('Actual group IDs:  ',actual.tolist())
print('Segment count unchanged; merged groups incorrectly change from 2 to 4.')
assert not np.array_equal(actual,expected), 'Bug no longer present; convert to equality regression.'
