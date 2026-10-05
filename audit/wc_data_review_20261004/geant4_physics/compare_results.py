import json
from pathlib import Path
import numpy as np
import uproot
from test_source_physics import counts
BASE=Path(__file__).resolve().parent
out={}
for suffix in ('','_fixed'):
    with uproot.open(BASE/f'genie_isotropic{suffix}.root') as f:
        z=np.array([v[0] for v in f['OpticalPhotons']['TrackInfo_DirZ'].array(library='np')])
    gaps=[]
    with uproot.open(BASE/f'pion_deflection{suffix}.root') as f:
        t=f['OpticalPhotons']
        a=t.arrays(['TrackInfo_ParentTrackID','TrackInfo_CreatorProcess','TrackInfo_Time',
        'TrackInfo_PosX','TrackInfo_PosY','TrackInfo_PosZ','Segment_TrackID',
        'Segment_EndX','Segment_EndY','Segment_EndZ'],library='np')
        for e in range(t.num_entries):
            for i,p in enumerate(a['TrackInfo_CreatorProcess'][e]):
                if str(p).startswith('Deflection_'):
                    parent=a['TrackInfo_ParentTrackID'][e][i]
                    j=np.flatnonzero(a['Segment_TrackID'][e]==parent)[-1]
                    old=np.array([a[f'Segment_End{k}'][e][j] for k in 'XYZ'])
                    new=np.array([a[f'TrackInfo_Pos{k}'][e][i] for k in 'XYZ'])
                    gaps.append(float(np.linalg.norm(new-old)))
    pp=counts('gamma_prompt',suffix);pd=counts('gamma_delayed',suffix)
    nn=counts('neutron',suffix)
    with uproot.open(BASE/f'neutron{suffix}.root') as f:
        t=f['OpticalPhotons']
        a=t.arrays(['TrackInfo_PDG','TrackInfo_CreatorProcess','TrackInfo_Time'],library='np')
        capture_gamma_times=[float(t) for pdg,pr,tm in zip(a['TrackInfo_PDG'],a['TrackInfo_CreatorProcess'],a['TrackInfo_Time']) for p,c,t in zip(pdg,pr,tm) if p==22 and c=='nCapture']
    out[suffix or 'original']={'isotropy':{'n':len(z),'mean_cos_theta':float(z.mean()),'second_moment':float(np.mean(z*z)),'positive_z_fraction':float(np.mean(z>0))},'late_gamma':{'prompt_total':int(pp.sum()),'delayed_total':int(pd.sum()),'loss':float(1-pd.sum()/pp.sum())},'pion':{'replacement_count':len(gaps),'max_gap_mm':max(gaps),'median_gap_mm':float(np.median(gaps))},'neutrons':{'events':len(nn),'photons_total':int(nn.sum()),'events_with_any_photons':int(np.sum(nn>0)),'capture_gamma_count':len(capture_gamma_times),'capture_gamma_mean_us':float(np.mean(capture_gamma_times)/1000) if capture_gamma_times else None}}
print(json.dumps(out,indent=2))
(BASE/'comparison_results.json').write_text(json.dumps(out,indent=2)+'\n')
