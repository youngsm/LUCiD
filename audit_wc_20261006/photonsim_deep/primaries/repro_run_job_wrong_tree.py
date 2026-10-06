"""Real production runner + real PhotonSim; run only on allowed Slurm CPUs."""
import os, json, subprocess, sys
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
B=Path(__file__).resolve().parent
cfg={'name':'audit GENIE input validation','material':'water','primary_source':'genie',
     'genie':{'probe_pdg':14,'energy_min_GeV':0.9,'energy_max_GeV':1.1,
              'target':'water','tune':'G18_10a_02_11b','direction':'isotropic'},
     'store_individual_photons':True,'disable_decays':False,'cleanup_root_files':False}
cfgpath=B/'production_genie_config.json'; cfgpath.write_text(json.dumps(cfg,indent=2))
env=dict(os.environ,PHOTONSIM_BIN=str(B/'photonsim_wrapper.sh'))
results={}
for case,marker in [('wrong_tree_without_marker',False),('wrong_tree_with_marker',True)]:
    out=B/case; out.mkdir(exist_ok=True)
    fixture=out/'gntp_job_000001.gtrac.root'
    with uproot.recreate(fixture) as f:
        f['unrelated_tree']={'x':np.array([1],dtype=np.int32)}
    if marker: (out/'gntp_job_000001.primaries.txt').write_text('2\n')
    args=[sys.executable,'-m','lucid.production.run_job','--config',str(cfgpath),
          '--output-dir',str(out),'--job-id','1','--n-events','2','--detector','SK_WAND',
          '--master-seed','1701','--skip-genie','--skip-lucid','--keep-root']
    p=subprocess.run(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'run_job.log').write_text(p.stdout)
    assert p.returncode==0, (case,p.returncode,p.stdout[-4000:])
    with uproot.open(out/'output_job_000001.root') as f:
        a=f['OpticalPhotons'].arrays(['PrimaryEnergy','NOpticalPhotons','IncomingNuPdg',
            'RooTrackerEntryID','TrackInfo_PDG','TrackInfo_Energy','TrackInfo_ParentTrackID'])
        primary=a['TrackInfo_ParentTrackID']==0
        pdg=ak.to_list(a['TrackInfo_PDG'][primary]); energy=ak.to_list(a['TrackInfo_Energy'][primary])
        assert pdg==[[11],[11]],pdg
        assert np.allclose(ak.to_numpy(a['PrimaryEnergy']),5.)
        assert np.allclose(energy,5.)
        assert np.all(ak.to_numpy(a['NOpticalPhotons'])>0)
        assert np.all(ak.to_numpy(a['IncomingNuPdg'])==0)
        assert np.all(ak.to_numpy(a['RooTrackerEntryID'])==-1)
        results[case]={'runner_exit':p.returncode,'events':len(a),'primary_pdgs':pdg,
            'primary_kinetic_MeV':energy,'created_photons':ak.to_list(a['NOpticalPhotons']),
            'rootracker_entries':ak.to_list(a['RooTrackerEntryID']),
            'wrong_tree_file_existed':fixture.is_file(),'count_marker_present':marker}
(B/'production_wrong_tree_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
