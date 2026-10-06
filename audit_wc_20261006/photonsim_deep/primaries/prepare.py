"""Prepare independent primary-injection fixtures; allowed Slurm CPU only."""
from pathlib import Path
import json, os
import numpy as np
import uproot
from lucid.production.generate_macro import generate_macro

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
B=Path(__file__).resolve().parent
REPO=B.parents[2]

def commands(name,cfg,rootracker=None,seeds=(42631,98711)):
    macro=generate_macro(cfg,str(B/f'{name}.root'),20000,genie_rootracker=rootracker,photonsim_seeds=seeds)
    (B/f'{name}.full.mac').write_text(macro)
    lines=[x for x in macro.splitlines() if x.startswith(('/gun/','/random/'))]
    (B/f'{name}.commands').write_text('\n'.join(lines)+'\n')

bomb=json.loads((REPO/'lucid/production/configs/GeV/01_pbomb.json').read_text())
commands('bomb',bomb)
commands('bomb_newseed',bomb,seeds=(42632,98711))
commands('multigun',dict(name='audit_multigun',material='water',energy_distribution='monoenergetic',
    particles=[dict(type='mu-',energy_MeV=500.),dict(type='e+',energy_MeV=25.),dict(type='gamma',energy_MeV=10.)]))

def rootracker(name,pdg,status,mom,masses,n):
    p4=np.column_stack((np.asarray(mom),np.sqrt(np.sum(np.asarray(mom)**2,axis=1)+np.asarray(masses)**2)))/1000.
    count=len(pdg)
    with uproot.recreate(B/f'{name}.gtrac.root') as f:
        tree=f.mktree('gRooTracker',{'StdHepN':'int32','StdHepPdg':f'{count} * int32',
            'StdHepStatus':f'{count} * int32','StdHepP4':f'{count} * 4 * float64'})
        tree.extend({'StdHepN':np.full(n,count,np.int32),
            'StdHepPdg':np.tile(np.asarray(pdg,np.int32),(n,1)),
            'StdHepStatus':np.tile(np.asarray(status,np.int32),(n,1)),
            'StdHepP4':np.tile(p4,(n,1,1))})
    cfg=dict(name=name,material='water',primary_source='genie',genie=dict(direction='isotropic'))
    commands(name,cfg,rootracker=str(B/f'{name}.gtrac.root'))

rootracker('correlated',[14,13,2212,211,22],[0,1,1,1,1],
            [[0,0,2000],[100,200,300],[-400,300,100],[50,-60,70],[0,0,50]],
            [0,105.6583715,938.27208816,139.57039,0],20000)
rootracker('ion',[14,11,1000080160],[0,1,1],
            [[0,0,2000],[0,0,100],[0,0,200]], [0,.51099895,14895.08],1)

missing=dict(name='missing',material='water',primary_source='genie',genie={})
commands('missing',missing,rootracker=str(B/'does_not_exist.gtrac.root'))
with uproot.recreate(B/'wrong_tree.gtrac.root') as f:
    f['unrelated_tree']={'x':np.array([1],dtype=np.int32)}
commands('wrong_tree',missing,rootracker=str(B/'wrong_tree.gtrac.root'))

configs={}
for path in sorted((REPO/'lucid/production/configs/GeV').rglob('*.json')):
    cfg=json.loads(path.read_text())
    configs[str(path.relative_to(REPO))]=dict(source=cfg.get('primary_source','particles'),
        particle_count=len(cfg.get('particles',[])),decays_disabled=cfg.get('disable_decays',False),
        store_photons=cfg.get('store_individual_photons',False))
(B/'config_inventory.json').write_text(json.dumps(configs,indent=2))
