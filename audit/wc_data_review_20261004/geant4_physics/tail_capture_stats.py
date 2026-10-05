import numpy as np
import uproot
for n in ['thermal_neutron_fixed','thermal_neutron_noelastic']:
    a=uproot.open(n+'.root')['OpticalPhotons'].arrays(['TrackInfo_PDG','TrackInfo_CreatorProcess','TrackInfo_Time','TrackInfo_Energy'],library='np')
    ts=[]; uncaptured=[]
    for ev,(p,c,t,e) in enumerate(zip(a['TrackInfo_PDG'],a['TrackInfo_CreatorProcess'],a['TrackInfo_Time'],a['TrackInfo_Energy'])):
        m=(p==22)&(c=='nCapture')
        if m.any(): ts.append(float(t[m].min())/1000)
        else: uncaptured.append({'event':ev,'pdg':p.tolist(),'process':c.tolist(),'time_ns':t.tolist(),'energy_mev':e.tolist()})
    ts=np.array(ts)
    print(n, 'frac>1ms',float(np.mean(ts>1000)),'frac>2ms',float(np.mean(ts>2000)),'p95us',float(np.quantile(ts,.95)),'maxus',float(ts.max()),'uncaptured',uncaptured)
