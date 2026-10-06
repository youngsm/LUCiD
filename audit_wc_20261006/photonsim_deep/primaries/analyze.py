"""Independent momentum/energy and distribution oracles, allowed Slurm CPUs only."""
import json, os
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
B=Path(__file__).resolve().parent
def read(name):
    return np.atleast_1d(np.genfromtxt(B/f'{name}.tsv',names=True,delimiter='\t'))
def momentum(a):
    return np.column_stack([a[f'{c}_MeV'] for c in ('px','py','pz')])
def snapshot(a):
    return {n:a[n].tolist() for n in ('pdg','kinetic_MeV','event_true_MeV')}
r={}
a=read('bomb'); e=a['event'].astype(int)
counts=np.bincount(e)
freq=np.bincount(counts,minlength=6)[1:6]
assert len(counts)==20000 and np.all((counts>=1)&(counts<=5))
assert np.max(np.abs(freq-4000))<450
pool=np.array([11,-11,13,-13,211,-211,111,22])
assert set(np.unique(a['pdg']))==set(pool)
pf=np.array([np.sum(a['pdg']==x) for x in pool])
assert np.max(np.abs(pf-len(a)/8))<7*np.sqrt(len(a)*7/64)
cfg=json.loads((B.parents[2]/'lucid/production/configs/GeV/01_pbomb.json').read_text())
bp=cfg['bomb']
name_to_pdg={'e-':11,'e+':-11,'mu-':13,'mu+':-13,'pi+':211,'pi-':-211,'pi0':111,'gamma':22}
limits={name_to_pdg[c['type']]:(c['energy_min_MeV'],c['energy_max_MeV']) for c in bp['candidates']}
lo=np.array([limits[int(p)][0] for p in a['pdg']]); hi=np.array([limits[int(p)][1] for p in a['pdg']])
u=(a['kinetic_MeV']-lo)/(hi-lo)
assert np.all((u>=0)&(u<=1)) and abs(u.mean()-.5)<.01
p=momentum(a); directions=p/np.linalg.norm(p,axis=1)[:,None]
assert np.max(np.abs(directions.mean(axis=0)))<.015
assert np.max(np.abs((directions**2).mean(axis=0)-1/3))<.015
totals=np.bincount(e,weights=a['kinetic_MeV'])
err=np.max(np.abs(a['event_true_MeV']-totals[e]))
assert err<1e-8
assert np.all(a['x_mm']==0) and np.all(a['y_mm']==0) and np.all(a['z_mm']==0) and np.all(a['t_ns']==0)
repeat=read('bomb_repeat'); newseed=read('bomb_newseed')
assert np.array_equal(a[e<10],repeat)
assert not np.array_equal(repeat,newseed)
r['bomb']={'events':len(counts),'primaries':len(a),'multiplicity_counts_1_to_5':freq.tolist(),
    'pdgs':pool.tolist(),'species_counts':pf.tolist(),'normalized_energy_mean':u.mean(),
    'direction_means':directions.mean(axis=0).tolist(),
    'direction_squared_means':(directions**2).mean(axis=0).tolist(),
    'max_true_energy_sum_error_MeV':err,'same_seeds_identical':True,'changed_seeds_differ':True}
a=read('correlated'); p=momentum(a).reshape(20000,4,3)
expected=np.array([[100,200,300],[-400,300,100],[50,-60,70],[0,0,50.]])
gram=p@p.transpose(0,2,1); oracle=expected@expected.T
gram_err=np.max(np.abs(gram-oracle))
assert gram_err<1e-7
assert np.all(a['pdg'].reshape(20000,4)==[13,2212,211,22])
assert np.array_equal(a['rootracker_entry'].reshape(20000,4)[:,0],np.arange(20000))
assert np.all(np.linalg.det(p[:,:3,:])*np.linalg.det(expected[:3,:])>0)
mass=np.array([105.6583715,938.27208816,139.57039,0])
ke=np.sqrt((expected**2).sum(axis=1)+mass**2)-mass
ke_err=np.max(np.abs(a['kinetic_MeV'].reshape(20000,4)-ke))
assert ke_err<1e-4
dirs=p/np.linalg.norm(p,axis=2)[:,:,None]
assert np.max(np.abs(dirs.mean(axis=0)))<.025
r['genie_correlated']={'events':20000,'max_gram_error_MeV2':gram_err,
    'max_kinetic_energy_error_MeV':ke_err,'chirality_preserved':True,
    'entry_sequence_correct':True,'direction_means':dirs.mean(axis=0).tolist()}
a=read('multigun')
assert np.all(a['pdg'].reshape(5,3)==[13,-11,22])
assert np.allclose(a['kinetic_MeV'].reshape(5,3),[500,25,10])
r['multigun']=snapshot(a[:3])
r['ion']=snapshot(read('ion'))
r['ion_after_explicit_GetIon']=snapshot(read('ion_preload'))
r['missing_input']=snapshot(read('missing'))
r['existing_wrong_tree']=snapshot(read('wrong_tree'))
for kind in ('missing_input','existing_wrong_tree'):
    assert r[kind]['pdg']==[11.] and np.allclose(r[kind]['kinetic_MeV'],[5.])
(B/'results.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
