"""Pion decay at rest -> mu+ stopping in water -> Michel angular correlation.

Cherenkov is disabled ONLY to reduce optical I/O; charged-particle propagation,
pion/muon decay, material and all other registered physics remain unchanged.
Run inside milano/roma only. The candidate fix is isolated under this audit folder.
"""
from pathlib import Path
import json
import sys
from collections import Counter
import numpy as np
import uproot

BASE = Path(__file__).resolve().parent

def prepare():
    src = (BASE/'upstream/src/PhysicsList.cc').read_text()
    src = src.replace('#include "G4DecayPhysics.hh"', '#include "G4DecayPhysics.hh"\n#include "G4SpinDecayPhysics.hh"')
    src = src.replace('RegisterPhysics(new G4DecayPhysics(0));',
                      'RegisterPhysics(new G4DecayPhysics(0));\n  RegisterPhysics(new G4SpinDecayPhysics(0));')
    (BASE/'PhysicsList_spin.cc').write_text(src)
    for variant in ('baseline', 'spin'):
        (BASE/f'pion_rest_{variant}.mac').write_text(
            f'/output/filename {BASE}/pion_rest_{variant}.root\n/run/initialize\n'
            '/random/setSeeds 314159 271828\n/photon/storeIndividual true\n'
            '/process/inactivate Cerenkov\n/gun/particle pi+\n/gun/energy 0.000001 MeV\n'
            '/gun/direction 0 0 1\n/run/beamOn 2000\n')

def analyze():
    results = {}
    variants = ['baseline','spin'] + (['water'] if (BASE/'pion_rest_water.root').exists() else [])
    for variant in variants:
        d = uproot.open(BASE/f'pion_rest_{variant}.root')['OpticalPhotons'].arrays(library='np')
        cosines, energies, muon_energies = [],[],[]
        positron_processes, muon_processes = Counter(), Counter()
        for ev in range(len(d['EventID'])):
            tid = d['TrackInfo_TrackID'][ev]
            pdg = d['TrackInfo_PDG'][ev]
            parent = d['TrackInfo_ParentTrackID'][ev]
            proc = d['TrackInfo_CreatorProcess'][ev]
            for ei in np.flatnonzero(pdg == -11):
                if str(proc[ei]) not in ('Decay','DecayWithSpin'): continue
                mi = np.flatnonzero(tid == parent[ei])
                if not len(mi) or pdg[mi[0]] != -13: continue
                mi = mi[0]
                pi = np.flatnonzero(tid == parent[mi])
                if not len(pi) or pdg[pi[0]] != 211: continue
                mu_direction = np.array([d[f'TrackInfo_Dir{k}'][ev][mi] for k in 'XYZ'])
                e_direction = np.array([d[f'TrackInfo_Dir{k}'][ev][ei] for k in 'XYZ'])
                cosines.append(float(mu_direction @ e_direction))
                energies.append(float(d['TrackInfo_Energy'][ev][ei]))
                muon_energies.append(float(d['TrackInfo_Energy'][ev][mi]))
                positron_processes[str(proc[ei])] += 1
                muon_processes[str(proc[mi])] += 1
        c, e = np.asarray(cosines), np.asarray(energies)
        high = e > 40
        results[variant] = {'events':len(d['EventID']), 'michel_count':len(c),
            'mean_cos_mu_birth_to_positron':float(c.mean()),
            'mean_cos_standard_error':float(c.std(ddof=1)/np.sqrt(len(c))),
            'positron_same_mu_hemisphere_fraction':float(np.mean(c>0)),
            'high_energy_count':int(high.sum()),
            'high_energy_mean_cos':float(c[high].mean()),
            'high_energy_same_hemisphere_fraction':float(np.mean(c[high]>0)),
            'positron_energy_mean_MeV':float(e.mean()),
            'muon_birth_kinetic_energy_mean_MeV':float(np.mean(muon_energies)),
            'positron_creator_processes':dict(positron_processes),
            'muon_creator_processes':dict(muon_processes),
            'cosine_histogram_10bins':np.histogram(c,bins=np.linspace(-1,1,11))[0].tolist()}
    (BASE/'spin_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    assert results['baseline']['michel_count'] > 1500
    assert abs(results['baseline']['mean_cos_mu_birth_to_positron']) < .04
    assert results['spin']['mean_cos_mu_birth_to_positron'] < -.06

if __name__ == '__main__':
    {'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
