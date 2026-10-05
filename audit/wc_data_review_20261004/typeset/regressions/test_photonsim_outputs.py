"""Physical contracts for newly generated PhotonSim output, not a live Python test."""
import os
from pathlib import Path
import numpy as np
import pytest
import uproot


@pytest.fixture(scope='module')
def source_outputs():
    directory = os.environ.get('PHOTONSIM_OUTPUT_DIR')
    if not directory:
        pytest.skip('Regenerate PhotonSim outputs and set PHOTONSIM_OUTPUT_DIR.')
    return Path(directory), os.environ.get('PHOTONSIM_TEST_SUFFIX', '')


def optical_arrays(outputs, name, branches):
    directory, suffix = outputs
    with uproot.open(directory / f'{name}{suffix}.root') as f:
        return f['OpticalPhotons'].arrays(branches, library='np')


def test_delayed_gamma_yield_is_time_translation_invariant(source_outputs):
    name = 'NOpticalPhotons'
    prompt = optical_arrays(source_outputs, 'gamma_prompt', [name])[name]
    delayed = optical_arrays(source_outputs, 'gamma_delayed', [name])[name]
    assert len(prompt) == len(delayed) == 30
    assert prompt.sum() > 0
    assert delayed.sum() >= .95 * prompt.sum()


def test_genie_isotropic_directions_are_uniform(source_outputs):
    a = optical_arrays(source_outputs, 'genie_isotropic', ['TrackInfo_DirZ'])
    z = np.array([v[0] for v in a['TrackInfo_DirZ']])
    assert len(z) == 20000
    assert abs(z.mean()) < .02
    assert abs(np.mean(z > 0.) - .5) < .02
    assert abs(np.mean(z*z) - 1./3.) < .02


def test_pion_replacement_preserves_endpoint(source_outputs):
    branches = ['TrackInfo_ParentTrackID', 'TrackInfo_CreatorProcess', 'Segment_TrackID']
    branches += [f'TrackInfo_Pos{k}' for k in 'XYZ'] + [f'Segment_End{k}' for k in 'XYZ']
    a = optical_arrays(source_outputs, 'pion_deflection', branches)
    gaps = []
    for e in range(len(a['TrackInfo_ParentTrackID'])):
        for i, process in enumerate(a['TrackInfo_CreatorProcess'][e]):
            if str(process).startswith('Deflection_'):
                parent = a['TrackInfo_ParentTrackID'][e][i]
                j = np.flatnonzero(a['Segment_TrackID'][e] == parent)[-1]
                old = np.array([a[f'Segment_End{k}'][e][j] for k in 'XYZ'])
                new = np.array([a[f'TrackInfo_Pos{k}'][e][i] for k in 'XYZ'])
                gaps.append(np.linalg.norm(new-old))
    assert gaps
    assert max(gaps) < 1e-6


def test_thermal_water_capture_clock(source_outputs):
    directory, _ = source_outputs
    filename = os.environ.get('PHOTONSIM_THERMAL_ROOT', 'thermal_neutron_fixed.root')
    with uproot.open(directory / filename) as f:
        a = f['OpticalPhotons'].arrays(
            ['TrackInfo_PDG', 'TrackInfo_CreatorProcess', 'TrackInfo_Time'], library='np')
    times = []
    for pdg, process, t in zip(a['TrackInfo_PDG'], a['TrackInfo_CreatorProcess'], a['TrackInfo_Time']):
        capture = (pdg == 22) & np.array([str(p).startswith('nCapture') for p in process])
        if capture.any():
            times.append(t[capture].min() / 1000.)
    assert len(a['TrackInfo_PDG']) == 1000 and len(times) >= 990
    t = np.array(times)
    n_h = 2. * 6.02214076e23 * 1e6 / 18.01528
    tau_us = 1e6 / (n_h * .3326e-28 * 2200.)
    assert abs(t.mean() - tau_us) < 4. * t.std(ddof=1) / np.sqrt(len(t))
    assert np.mean(t > 1000.) < .02
