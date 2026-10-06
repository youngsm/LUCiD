from pathlib import Path
import json
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
CASES=[('gamma25','gamma',25.,2000,True),('gamma2000','gamma',2000.,400,True),
       ('electron2000','e-',2000.,400,True),('muon2000','mu+',2000.,1000,True),
       ('mu_minus_rest','mu-',.000001,5000,False),('mu_plus_rest','mu+',.000001,5000,False),
       ('muon200','mu+',200.,200,False),('muon400','mu+',400.,200,False),('muon1000','mu+',1000.,200,False),
       ('proton200','proton',200.,200,False),('proton1000','proton',1000.,100,False),
       ('pionminus_rest','pi-',.000001,200,False),('kaonplus_rest','kaon+',.000001,200,False),
       ('neutron30','neutron',30.,200,False)]
def prepare():
    run_lines=[]
    for name,species,energy,n,both in CASES:
        for variant in (['baseline','extra'] if both else ['baseline']):
            key=f'{name}_{variant}'
            (BASE/f'{key}.mac').write_text(
                f'/output/filename {BASE}/{key}.root\n/random/setSeeds 314159 271828\n'
                '/photon/storeIndividual false\n/gun/clearPrimaries\n'
                f'/gun/particle {species}\n/gun/energy {energy} MeV\n/gun/direction 0 0 1\n/run/beamOn {n}\n')
            run_lines.append(f'{key} {variant}')
    (BASE/'cases.txt').write_text('\n'.join(run_lines)+'\n')

def analyze():
    results={}
    for name,species,_,expected_n,both in CASES:
        for variant in (['baseline','extra'] if both else ['baseline']):
            key=f'{name}_{variant}'
            d=np.genfromtxt(BASE/f'{key}.csv',delimiter=',',names=True)
            assert len(d)==expected_n, (key,len(d),expected_n)
            for column in d.dtype.names:
                assert np.all(np.isfinite(d[column])), (key,column)
            r={'n':len(d),'mean_cherenkov':float(d['photons'].mean()),
               'mean_cherenkov_se':float(d['photons'].std(ddof=1)/np.sqrt(len(d))),
               'cherenkov_quantiles':np.quantile(d['photons'],[0,.01,.05,.5,.95,1.]).tolist(),
               'mean_edep_MeV':float(d['edep_MeV'].mean())}
            for col in ['gamma_nuclear','electron_nuclear','muon_nuclear','neutrons','protons','charged_pions','secondary_muons','primary_nuclear','primary_decay_e']:
                r[col+'_sum']=int(d[col].sum())
                r[col+'_event_fraction']=float(np.mean(d[col]>0))
            stopped=d['primary_stop_ns']>=0
            if stopped.any():
                r['stopped_fraction']=float(stopped.mean())
                r['mean_stopped_primary_range_mm']=float(d['primary_range_mm'][stopped].mean())
                if species=='mu+':
                    r['mean_stopped_primary_lifetime_ns']=float((d['primary_death_ns'][stopped]-d['primary_stop_ns'][stopped]).mean())
            results[key]=r
    for name,_,_,_,both in CASES:
        if not both: continue
        baseline=results[name+'_baseline'];extra=results[name+'_extra']
        results[name+'_comparison']={'relative_mean_cherenkov_change':extra['mean_cherenkov']/baseline['mean_cherenkov']-1.,
            'mean_difference_standard_errors':(extra['mean_cherenkov']-baseline['mean_cherenkov'])/
                np.hypot(extra['mean_cherenkov_se'],baseline['mean_cherenkov_se'])}
    # Only a kept-daughter diagnostic: negative-muon endpoints need the
    # pre-stacking model-ID observer in capture_ledger_main.cc. Its parent
    # clock stays at stopping time while delayed daughter times advance.
    for sign,tau in [('minus',1795.4),('plus',2196.9811)]:
        d=np.genfromtxt(BASE/f'mu_{sign}_rest_baseline.csv',delimiter=',',names=True)
        results[f'mu_{sign}_rest_baseline']['visible_decay_probability_before_10us']=float(np.mean(d['primary_decay_e']>0))
        if sign=='plus':
            life=d['primary_death_ns']-d['primary_stop_ns']
            results[f'mu_{sign}_rest_baseline']['lifetime_se_ns']=float(life.std(ddof=1)/np.sqrt(len(life)))
    (BASE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__': {'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
