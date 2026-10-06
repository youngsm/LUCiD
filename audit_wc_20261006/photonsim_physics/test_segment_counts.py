"""Failing physical truth conservation oracle using the first actual gamma event."""
from pathlib import Path
import numpy as np
import uproot
with uproot.open(Path(__file__).resolve().parent/'gamma.root') as f:
    t = f['OpticalPhotons']
    expected = t['Segment_NCherenkov'].array(entry_stop=1, library='np')[0]
    idx = t['Photon_SegmentIndex'].array(entry_stop=1, library='np')[0]
    actual = np.bincount(idx, minlength=len(expected))
    print(f'first event: {len(idx)} photons; segment-count absolute mismatch={abs(actual-expected).sum()}')
    assert np.array_equal(actual, expected), 'Photon assigned to a different creation step'
