"""Actual 2.2 MeV gamma photons: compare time-linked segment to source position."""
from pathlib import Path
import json
import numpy as np
import uproot
base = Path(__file__).resolve().parent
f = uproot.open(base/'gamma.root')
d = f['OpticalPhotons'].arrays(library='np')
r = f['OpticalPhotonsRaw'].arrays(library='np')
examples = []
bad = checked = 0
for ev in range(len(d['EventID'])):
    chunks = np.flatnonzero(r['EventID'] == d['EventID'][ev])
    pos = np.concatenate([np.stack([r[f'PhotonPos{k}'][chunk] for k in 'XYZ'],axis=1) for chunk in chunks])
    segments = d['Photon_SegmentIndex'][ev]
    start = np.stack([d[f'Segment_Start{k}'][ev] for k in 'XYZ'],axis=1)
    end = np.stack([d[f'Segment_End{k}'][ev] for k in 'XYZ'],axis=1)
    delta = end-start
    for p, (origin, assigned) in enumerate(zip(pos, segments)):
        candidates = np.flatnonzero(d['Segment_TrackID'][ev] == d['Segment_TrackID'][ev][assigned])
        v = delta[candidates]
        norm2 = np.einsum('ij,ij->i',v,v)
        t = np.einsum('ij,ij->i',origin-start[candidates],v)/np.maximum(norm2,1e-30)
        distance = np.linalg.norm(origin-start[candidates]-np.clip(t,0,1)[:,None]*v,axis=1)
        nearest_local = np.argmin(distance)
        assigned_local = np.flatnonzero(candidates==assigned)[0]
        checked += 1
        if distance[assigned_local] > .001 and distance[nearest_local] < .001:
            bad += 1
            examples.append({'event':ev,'photon':p,'assigned_segment':int(assigned),
                'correct_position_segment':int(candidates[nearest_local]),
                'assigned_distance_mm':float(distance[assigned_local]),
                'correct_distance_mm':float(distance[nearest_local]),
                'source_mm':origin.tolist(),'assigned_start_mm':start[assigned].tolist(),
                'assigned_end_mm':end[assigned].tolist(),
                'correct_start_mm':start[candidates[nearest_local]].tolist(),
                'correct_end_mm':end[candidates[nearest_local]].tolist()})
examples.sort(key=lambda x:x['assigned_distance_mm'], reverse=True)
result = {'source':'gamma.root', 'checked_photons':checked,
          'wrong_segment_by_source_position':bad, 'wrong_fraction':bad/checked,
          'examples':examples[:5]}
(base/'segment_ownership_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert bad > 0, 'No displaced segment ownership reproduced'
