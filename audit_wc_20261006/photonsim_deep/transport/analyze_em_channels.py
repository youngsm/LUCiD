from pathlib import Path
import json
import numpy as np

base=Path(__file__).resolve().parent
results={}
for label,expected in [('gamma25',2000),('gamma2000',400),('electron2000',400),('muon2000',1000)]:
    variants={}
    for variant in ['baseline','extra']:
        path=base/f'{label}_{variant}.csv'
        assert path.exists(), path
        d=np.genfromtxt(path,delimiter=',',names=True)
        assert len(d)==expected, (path,len(d),expected)
        for column in d.dtype.names:
            assert np.all(np.isfinite(d[column])), (path,column)
        r={'n':len(d),'mean_photons':float(d['photons'].mean()),
           'mean_photons_se':float(d['photons'].std(ddof=1)/np.sqrt(len(d))),
           'min_photons':float(d['photons'].min()),
           'mean_neutrons':float(d['neutrons'].mean()),
           'fraction_with_neutrons':float(np.mean(d['neutrons']>0)),
           'mean_edep_MeV':float(d['edep_MeV'].mean())}
        for key in ['gamma_nuclear','electron_nuclear','muon_nuclear','primary_nuclear']:
            r[key+'_count']=int(d[key].sum())
            r[key+'_event_fraction']=float(np.mean(d[key]>0))
        variants[variant]=r
    assert len(variants)==2
    if len(variants)==2:
        a,b=variants['baseline'],variants['extra']
        variants['comparison']={'relative_photon_change':b['mean_photons']/a['mean_photons']-1,
            'mean_photon_difference_sigmas':(b['mean_photons']-a['mean_photons'])/np.hypot(a['mean_photons_se'],b['mean_photons_se'])}
    results[label]=variants
(base/'em_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
