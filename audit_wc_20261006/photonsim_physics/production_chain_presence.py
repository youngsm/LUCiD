"""Identify affected visible pi->mu->Michel chains in actual 1 GeV pion events."""
from pathlib import Path
import json
import numpy as np
import uproot
base = Path(__file__).resolve().parent
d = uproot.open(base/'pion.root')['OpticalPhotons'].arrays(library='np')
chains = []
for ev in range(len(d['EventID'])):
    ids, pdg, parents, proc = (d[k][ev] for k in ('TrackInfo_TrackID','TrackInfo_PDG','TrackInfo_ParentTrackID','TrackInfo_CreatorProcess'))
    for mi in np.flatnonzero(pdg == -13):
        pi = np.flatnonzero(ids == parents[mi])
        if not len(pi) or pdg[pi[0]] != 211 or str(proc[mi]) != 'Decay': continue
        electrons = np.flatnonzero((pdg == -11) & (parents == ids[mi]))
        for ei in electrons:
            if str(proc[ei]) != 'Decay': continue
            chains.append({'event':ev,'muon_birth_KE_MeV':float(d['TrackInfo_Energy'][ev][mi]),
                           'michel_birth_KE_MeV':float(d['TrackInfo_Energy'][ev][ei]),
                           'muon_cherenkov_count':int(d['Segment_NCherenkov'][ev][d['Segment_TrackID'][ev]==ids[mi]].sum()),
                           'muon_track':int(ids[mi]),'pion_track':int(parents[mi])})
result={'events':len(d['EventID']),'pion_muon_michel_chains':chains,
        'visible_muon_chains':sum(c['muon_cherenkov_count']>0 for c in chains)}
(base/'production_chain_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
