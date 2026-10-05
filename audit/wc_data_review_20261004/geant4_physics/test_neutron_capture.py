from pathlib import Path
import os
import numpy as np
import pytest
import uproot

@pytest.fixture
def capture_times_us():
    path=Path(__file__).resolve().parent/os.environ.get('PHOTONSIM_THERMAL_ROOT','thermal_neutron_fixed.root')
    with uproot.open(path) as f:
        a=f['OpticalPhotons'].arrays(['TrackInfo_PDG','TrackInfo_CreatorProcess','TrackInfo_Time'],library='np')
    times=[]
    for pdg,process,t in zip(a['TrackInfo_PDG'],a['TrackInfo_CreatorProcess'],a['TrackInfo_Time']):
        mask=(pdg==22)&np.array([str(p).startswith('nCapture') for p in process])
        if mask.any(): times.append(t[mask].min()/1000)
    assert len(a['TrackInfo_PDG'])==1000
    assert len(times)>=990
    return np.array(times)

def test_thermal_water_capture_clock(capture_times_us):
    t=capture_times_us
    n_h=2*6.02214076e23*1e6/18.01528
    tau_us=1e6/(n_h*.3326e-28*2200)
    assert abs(t.mean()-tau_us)<4*t.std(ddof=1)/np.sqrt(len(t))
    assert np.mean(t>1000)<.02
