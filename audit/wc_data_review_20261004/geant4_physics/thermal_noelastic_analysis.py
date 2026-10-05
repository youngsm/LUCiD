from pathlib import Path
import json
import numpy as np
import uproot
with uproot.open('thermal_neutron_noelastic.root') as f:
    a=f['OpticalPhotons'].arrays(['TrackInfo_PDG','TrackInfo_CreatorProcess','TrackInfo_Time','TrackInfo_Energy'],library='np')
    times=[]; gamma_energies=[]
    for p,c,t,e in zip(a['TrackInfo_PDG'],a['TrackInfo_CreatorProcess'],a['TrackInfo_Time'],a['TrackInfo_Energy']):
        m=(p==22)&(c=='nCapture')
        if np.any(m):
            times.append(float(np.min(t[m]))/1000)
            gamma_energies.extend(e[m].tolist())
    times=np.array(times)
    rho=1e6 # g/m3, Geant4G4_WATERdensity
    nH=2*6.02214076e23*rho/18.01528
    sigma=.3326e-28
    rate=nH*sigma*2200
    result={'events':len(a['TrackInfo_PDG']),'captures':len(times),'mean_us':float(times.mean()),'sample_sd_us':float(times.std(ddof=1)),'mean_standard_error_us':float(times.std(ddof=1)/np.sqrt(len(times))),'median_us':float(np.median(times)),'rate_oracle_us':float(1e6/rate),'hydrogen_density_m3':nH,'sigma_2200_barn':.3326,'gamma_energy_MeV_min_max':[min(gamma_energies),max(gamma_energies)],'histogram_times_100us_bins':np.histogram(times,bins=np.arange(0,2100,100))[0].tolist()}
    Path('thermal_noelastic_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
