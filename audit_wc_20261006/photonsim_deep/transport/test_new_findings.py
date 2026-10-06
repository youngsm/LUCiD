"""Tests physical oracles and actual process presence, not implementation mirrors."""
from pathlib import Path
import json
import numpy as np

b=Path(__file__).resolve().parent
capture=json.loads((b/'capture_results.json').read_text())
for particle in ['pi','mu']:
    assert not capture[f'{particle}_minus_cold_baseline']['passes_water_oracle']
    assert capture[f'{particle}_minus_cold_water']['passes_water_oracle']

baseline=(b/'gamma25_baseline.log').read_text()
fixed=(b/'gamma25_extra.log').read_text()
assert 'GammaGeneralProc{nuclear=NULL}' in baseline
assert 'GammaGeneralProc{nuclear=photonNuclear}' in fixed
for channel in ['electronNuclear','positronNuclear','muonNuclear']:
    assert channel not in baseline
    assert channel in fixed
a=np.genfromtxt(b/'gamma25_baseline.csv',delimiter=',',names=True)
c=np.genfromtxt(b/'gamma25_extra.csv',delimiter=',',names=True)
assert a['gamma_nuclear'].sum()==0
assert c['gamma_nuclear'].sum()>0
# Independent measured oxygen photonuclear giant-resonance channel exists:
# the injected gamma must sometimes cease being a pure EM shower.
assert np.mean(c['photons']<1000)>0.01
assert np.mean(a['photons']<1000)==0

out={'negative_muon_water_oracle':'baseline fails / water selector passes',
     'negative_pion_water_oracle':'baseline fails / measured water selector passes',
     'em_nuclear_processes':'baseline missing / G4EmExtraPhysics present',
     'muon_csda_range_checks':{}}
# PDG liquid-water CSDA values; stochastic transport and range straggling
# differ slightly, so this detects scale errors rather than 1% model errors.
for energy,expected_cm in [(200,78.94),(400,179.6),(1000,473.2),(2000,935.3)]:
    d=np.genfromtxt(b/f'muon{energy}_baseline.csv',delimiter=',',names=True)
    mean_cm=float(d['primary_range_mm'][d['primary_stop_ns']>=0].mean()/10)
    ratio=mean_cm/expected_cm
    assert abs(ratio-1)<.03
    out['muon_csda_range_checks'][str(energy)]={'measured_cm':mean_cm,'pdg_csda_cm':expected_cm,'ratio':ratio}
(b/'test_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
