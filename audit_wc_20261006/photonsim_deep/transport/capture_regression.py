"""Independent water disappearance oracle; run only inside milano/roma allocation."""
from pathlib import Path
import json
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
CASES=[('mu_minus_cold_baseline','mu-',1e-6,10000,'mu_baseline'),
       ('mu_minus_cold_water','mu-',1e-6,10000,'mu_water'),
       ('mu_plus_cold','mu+',1e-6,10000,'mu_baseline'),
       ('mu_minus200_baseline','mu-',200.,2000,'mu_baseline'),
       ('mu_minus200_water','mu-',200.,2000,'mu_water'),
       ('pi_minus_cold_baseline','pi-',1e-6,3000,'pi_baseline'),
       ('pi_minus_cold_water','pi-',1e-6,3000,'pi_water'),
       ('pi_minus200_baseline','pi-',200.,1000,'pi_baseline'),
       ('pi_minus200_water','pi-',200.,1000,'pi_water')]

def prepare():
    for name,species,energy,n,variant in CASES:
        (BASE/f'{name}.mac').write_text(
            f'/output/filename {BASE}/{name}.root\n/random/setSeeds 314159 271828\n'
            '/photon/storeIndividual false\n/gun/clearPrimaries\n'
            f'/gun/particle {species}\n/gun/energy {energy} MeV\n/gun/direction 0 0 1\n/run/beamOn {n}\n')
    (BASE/'capture_cases.txt').write_text('\n'.join(f'{c[0]} {c[4]}' for c in CASES)+'\n')

def analyze():
    results={}
    for name,species,_,n,_ in CASES:
        d=np.genfromtxt(BASE/f'{name}.csv',delimiter=',',names=True)
        assert len(d)==n
        if species=='pi-':
            captures=d['pion_captures'].sum()
            hydrogen=d['pion_hydrogen'].sum()
            p=hydrogen/captures
            r={'n':n,'pion_captures':int(captures),'pion_hydrogen':int(hydrogen),
               'hydrogen_capture_fraction':float(p),
               'hydrogen_fraction_se':float(np.sqrt(p*(1-p)/captures)),
               'pi0_from_stopped_capture':int(d['pion_pi0'].sum()),
               'hard_gamma_from_stopped_capture':int(d['pion_hard_gamma'].sum()),
               'mean_created_cherenkov':float(d['photons'].mean()),
               'mean_created_cherenkov_se':float(d['photons'].std(ddof=1)/np.sqrt(n)),
               'cherenkov_quantiles':np.quantile(d['photons'],[0,.1,.5,.9,1.]).tolist(),
               'zero_light_fraction':float(np.mean(d['photons']==0))}
            # PRA75,034501 (2007), H capture probability for pi- in water.
            r['hydrogen_z_from_water']=float((p-.00445)/np.hypot(r['hydrogen_fraction_se'],.00024))
            r['passes_water_oracle']=bool(abs(r['hydrogen_z_from_water'])<4)
            results[name]=r
            continue
        endpoint=d['endpoint_decay']+d['endpoint_capture']
        assert np.all(endpoint==1), (name,np.unique(endpoint,return_counts=True))
        stopped=d['primary_stop_ns']>=0
        ns=int(stopped.sum())
        life=(d['endpoint_time_ns']-d['primary_stop_ns'])[stopped]
        assert np.all(life>=0)
        r={'n':len(d),'n_stopped':ns,'capture_fraction':float(d['endpoint_capture'][stopped].mean()),
           'capture_fraction_se':float(d['endpoint_capture'][stopped].std(ddof=1)/np.sqrt(ns)),
           'endpoint_lifetime_mean_ns':float(life.mean()),
           'endpoint_lifetime_se_ns':float(life.std(ddof=1)/np.sqrt(ns)),
           'oxygen_target_fraction':float(np.mean(d['target_z']==8)),
           'mean_created_cherenkov':float(d['photons'].mean()),
           'mean_created_cherenkov_se':float(d['photons'].std(ddof=1)/np.sqrt(n)),
           'visible_michel_fraction_before_10us':float(np.mean(d['primary_decay_e']>0)),
           'endpoint_fraction_after_10us':float(np.mean(d['endpoint_time_ns']>10000.)),
           'mean_neutrons':float(d['neutrons'].mean())}
        for z in [1,8]:
            s=(d['target_z']==z)&stopped
            if s.any():
                r[f'target_{z}_capture_fraction']=float(d['endpoint_capture'][s].mean())
                r[f'target_{z}_lifetime_ns']=float((d['endpoint_time_ns'][s]-d['primary_stop_ns'][s]).mean())
        if species=='mu-':
            # Water measurements reported by Super-K PRD110,082008 (2024).
            # Endpoints are observed at birth before PhotonSim's accepted cut,
            # so these fractions and lifetimes require no truncation correction.
            r['capture_z_from_water']=float((r['capture_fraction']-.184)/np.hypot(r['capture_fraction_se'],.001))
            r['lifetime_z_from_water']=float((r['endpoint_lifetime_mean_ns']-1795.4)/np.hypot(r['endpoint_lifetime_se_ns'],2.0))
            r['passes_water_oracle']=bool(abs(r['capture_z_from_water'])<4 and abs(r['lifetime_z_from_water'])<4)
        results[name]=r
    assert not results['mu_minus_cold_baseline']['passes_water_oracle']
    assert results['mu_minus_cold_water']['passes_water_oracle']
    assert not results['mu_minus200_baseline']['passes_water_oracle']
    assert results['mu_minus200_water']['passes_water_oracle']
    assert not results['pi_minus_cold_baseline']['passes_water_oracle']
    assert results['pi_minus_cold_water']['passes_water_oracle']
    plus=results['mu_plus_cold']
    assert plus['capture_fraction']==0
    assert abs(plus['endpoint_lifetime_mean_ns']-2196.9811)<4*plus['endpoint_lifetime_se_ns']
    (BASE/'capture_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': {'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
