from pathlib import Path
import os
import numpy as np
import pytest
import uproot

BASE = Path(__file__).resolve().parent

@pytest.fixture
def suffix():
    return os.environ.get('PHOTONSIM_TEST_SUFFIX', '')

def counts(name, suffix):
    with uproot.open(BASE / f'{name}{suffix}.root') as f:
        return f['OpticalPhotons']['NOpticalPhotons'].array(library='np')

def test_late_gamma_yield_is_time_translation_invariant(suffix):
    prompt=counts('gamma_prompt',suffix)
    delayed=counts('gamma_delayed',suffix)
    assert len(prompt)==len(delayed)==30
    assert delayed.sum() >= .95*prompt.sum()

def test_genie_isotropic_directions_are_uniform_on_sphere(suffix):
    with uproot.open(BASE / f'genie_isotropic{suffix}.root') as f:
        z=np.array([v[0] for v in f['OpticalPhotons']['TrackInfo_DirZ'].array(library='np')])
    assert len(z)==20000
    assert abs(z.mean())<.02
    assert abs(np.mean(z>0)-.5)<.02
    assert abs(np.mean(z*z)-1/3)<.02

def test_pion_deflection_preserves_position(suffix):
    with uproot.open(BASE / f'pion_deflection{suffix}.root') as f:
        t=f['OpticalPhotons']
        a=t.arrays(['TrackInfo_ParentTrackID','TrackInfo_CreatorProcess',
            'TrackInfo_PosX','TrackInfo_PosY','TrackInfo_PosZ',
            'Segment_TrackID','Segment_EndX','Segment_EndY','Segment_EndZ'],library='np')
        gaps=[]
        for e in range(t.num_entries):
            for i,p in enumerate(a['TrackInfo_CreatorProcess'][e]):
                if str(p).startswith('Deflection_'):
                    parent=a['TrackInfo_ParentTrackID'][e][i]
                    j=np.flatnonzero(a['Segment_TrackID'][e]==parent)[-1]
                    old=np.array([a[f'Segment_End{k}'][e][j] for k in 'XYZ'])
                    new=np.array([a[f'TrackInfo_Pos{k}'][e][i] for k in 'XYZ'])
                    gaps.append(np.linalg.norm(new-old))
        assert gaps
        assert max(gaps)<1e-6
