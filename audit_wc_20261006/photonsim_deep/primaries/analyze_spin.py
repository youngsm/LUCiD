import os, json
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
B=Path(__file__).resolve().parent
def load(name): return np.atleast_1d(np.genfromtxt(B/f'{name}.tsv',delimiter='\t',names=True))
def vectors(a,cols): return np.column_stack([a[c] for c in cols])
base=load('spin_baseline'); fixed=load('spin_repaired'); beam=load('spin_beam_repaired'); missing=load('spin_missing_repaired')
spins=('spin_x','spin_y','spin_z'); momenta=('px_MeV','py_MeV','pz_MeV')
assert np.array_equal(vectors(base,momenta),vectors(fixed,momenta))
assert np.all(vectors(base,spins)==0)
mu=np.abs(fixed['pdg'])==13; s=vectors(fixed,spins)[mu]; p=vectors(fixed,momenta)[mu]
norm=np.linalg.norm(s,axis=1); helicity=np.sum(s*p,axis=1)/np.linalg.norm(p,axis=1)
expected=np.where(fixed['pdg'][mu]==13,-1.,1.)
assert np.max(np.abs(norm-1))<1e-12
assert np.max(np.abs(helicity-expected))<1e-12
assert np.all(vectors(fixed,spins)[~mu]==0)
unit=np.array([100.,200.,300.]); unit/=np.linalg.norm(unit)
mu_beam=np.abs(beam['pdg'])==13
assert np.allclose(vectors(beam,spins)[mu_beam],np.where(beam['pdg'][mu_beam,None]==13,-1.,1.)*unit)
assert np.all(vectors(missing,spins)==0)
assert len(missing)==20, 'Expected four imported primaries in each of five events'
assert np.array_equal(missing['event'],np.repeat(np.arange(5),4))
assert np.array_equal(missing['rootracker_entry'],np.repeat(np.arange(5),4)), 'GENIE input must not fall back to default electrons'
assert np.array_equal(missing['pdg'],np.tile([13,2212,211,22],5))
r={'events':256,'muon_primaries':int(mu.sum()),'baseline_muon_spin_norm_min_max':[0.,0.],
   'repaired_muon_spin_norm_min_max':[float(norm.min()),float(norm.max())],
   'max_repaired_helicity_error':float(np.max(np.abs(helicity-expected))),
   'same_momenta_before_after':True,'nonrandom_direction_preserved':True,'missing_optional_branch_keeps_zero_spin':True,
   'missing_optional_branch_has_expected_five_entries_and_twenty_primaries':True}
(B/'spin_results.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
