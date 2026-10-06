"""Run only inside a milano/roma allocation: exact PhotonSim source contract checks."""
from pathlib import Path
import json
import sys
import numpy as np
import uproot

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
sys.path.insert(0, str(REPO))

def prepare():
    from lucid.production.generate_macro import generate_macro
    cases = [('electron', 'e-', 100., 5), ('electron_10gev', 'e-', 10000., 5), ('muon', 'mu-', 1000., 5),
             ('pion', 'pi+', 1000., 20), ('gamma', 'gamma', 2.2, 10)]
    for name, species, energy, n in cases:
        config = {'name': name, 'config_number': 1, 'material': 'water',
                  'energy_distribution': 'monoenergetic',
                  'store_individual_photons': True, 'disable_decays': False,
                  'particles': [{'type': species, 'energy_MeV': energy}],
                  'fixed_direction_z': True}
        (BASE / f'{name}.mac').write_text(generate_macro(config, str(BASE / f'{name}.root'), n,
                                                       photonsim_seeds=(314159, 271828)))
    n = 20000
    p4 = np.zeros((n, 2, 4), np.float64)
    p4[:, :, 2:] = .001
    with uproot.recreate(BASE / 'forward.gtrac.root') as f:
        tree = f.mktree('gRooTracker', {'StdHepN':'int32', 'StdHepPdg':'2 * int32',
                                     'StdHepStatus':'2 * int32', 'StdHepP4':'2 * 4 * float64'})
        tree.extend({'StdHepN':np.full(n,2,np.int32),
                     'StdHepPdg':np.tile(np.array([14,14],np.int32),(n,1)),
                     'StdHepStatus':np.tile(np.array([0,1],np.int32),(n,1)), 'StdHepP4':p4})
    config = {'name':'genie_isotropy', 'config_number':2, 'material':'water',
              'primary_source':'genie', 'genie':{'direction':'isotropic'},
              'store_individual_photons':True}
    (BASE/'isotropy.mac').write_text(generate_macro(config, str(BASE/'isotropy.root'), n,
          genie_rootracker=str(BASE/'forward.gtrac.root'), photonsim_seeds=(314159,271828)))

def analyze():
    result = {'photonsim_ref':'v1.0.2', 'photonsim_sha':'9232a7dae46ae36d9c81dd5f1b4e9f7653803514'}
    for name in ('electron','muon','pion','gamma'):
        f = uproot.open(BASE/f'{name}.root')
        d = f['OpticalPhotons'].arrays(library='np')
        raw = f['OpticalPhotonsRaw'].arrays(library='np')
        gaps, mismatched, invalid, wavelengths = [],0,0,[]
        primary_energy = []
        for ev in range(len(d['EventID'])):
            idx = d['Photon_SegmentIndex'][ev]
            invalid += int(np.count_nonzero(idx < 0))
            hist = np.bincount(idx[idx >= 0], minlength=len(d['Segment_NCherenkov'][ev]))
            mismatched += int(np.abs(hist-d['Segment_NCherenkov'][ev]).sum())
            primary = d['TrackInfo_ParentTrackID'][ev] == 0
            primary_energy.extend(d['TrackInfo_Energy'][ev][primary].tolist())
            for row, proc in enumerate(d['TrackInfo_CreatorProcess'][ev]):
                if not str(proc).startswith('Deflection_'): continue
                parent = d['TrackInfo_ParentTrackID'][ev][row]
                last = np.flatnonzero(d['Segment_TrackID'][ev] == parent)[-1]
                end = np.array([d[f'Segment_End{k}'][ev][last] for k in 'XYZ'])
                born = np.array([d[f'TrackInfo_Pos{k}'][ev][row] for k in 'XYZ'])
                gaps.append(float(np.linalg.norm(end-born)))
        wl = np.concatenate(raw['PhotonWavelength'])
        counts = d['NOpticalPhotons']
        result[name] = {'events':len(counts), 'photons':int(counts.sum()),
                        'raw_photons':len(wl), 'segment_cherenkov_photons':int(sum(map(np.sum,d['Segment_NCherenkov']))),
                        'invalid_segment_indices':invalid,'absolute_segment_count_mismatch':mismatched,
                        'wavelength_min_nm':float(wl.min()),'wavelength_max_nm':float(wl.max()),
                        'primary_kinetic_energies_MeV':primary_energy,
                        'pion_replacements':len(gaps),'max_pion_endpoint_gap_mm':max(gaps,default=0.)}
        assert len(wl) == counts.sum()
        assert invalid == 0
        assert sum(map(np.sum,d['Segment_NCherenkov'])) == counts.sum()
        expected_ke = {'electron':100., 'muon':1000., 'pion':1000., 'gamma':2.2}[name]
        assert np.allclose(primary_energy, expected_ke, rtol=1e-10, atol=1e-10)
        assert max(gaps,default=0.) < 1e-6
        assert wl.min() >= 1239.84/4.51-.01 and wl.max() <= 1239.85/1.84+.01
    z = np.array([x[0] for x in uproot.open(BASE/'isotropy.root')['OpticalPhotons']['TrackInfo_DirZ'].array(library='np')])
    result['genie_isotropy'] = {'events':len(z), 'mean_cos_theta':float(z.mean()),
          'mean_cos_theta_squared':float(np.mean(z*z)), 'positive_z_fraction':float(np.mean(z>0))}
    assert abs(z.mean()) < .02 and abs(np.mean(z*z)-1/3) < .02 and abs(np.mean(z>0)-.5)<.02
    # Independent Frank-Tamm source yield check on energetic primary muon
    # steps. Step chords approximate true length extremely well here; restrict
    # beta>0.99 to avoid stopping-track scattering and use a 3% tolerance.
    d = uproot.open(BASE/'muon.root')['OpticalPhotons'].arrays(library='np')
    grid = np.linspace(1.84,4.51,10001)
    refractive_index = np.interp(grid, [1.84,2.07,2.48,2.76,3.10,3.31,3.54,3.81,4.13,4.51],
                         [1.33110,1.33306,1.33680,1.33966,1.34356,1.34624,1.34944,1.35360,1.35915,1.36679])
    observed, expected, length_m = 0,0.,0.
    for ev in range(len(d['EventID'])):
        primary = d['TrackInfo_TrackID'][ev][d['TrackInfo_ParentTrackID'][ev]==0][0]
        mask = (d['Segment_TrackID'][ev] == primary) & (d['Segment_BetaStart'][ev]>.99)
        beta = d['Segment_BetaStart'][ev][mask]
        ends = np.stack([d[f'Segment_End{k}'][ev][mask] for k in 'XYZ'],axis=1)
        starts = np.stack([d[f'Segment_Start{k}'][ev][mask] for k in 'XYZ'],axis=1)
        length = np.linalg.norm(ends-starts,axis=1)*.001
        integral = np.trapezoid(1.-1./(beta[:,None]*refractive_index[None,:])**2,grid,axis=1)
        expected += float(np.sum(length*integral/(137.035999084*1.973269804e-7)))
        observed += int(d['Segment_NCherenkov'][ev][mask].sum())
        length_m += float(length.sum())
    result['frank_tamm'] = {'selected_primary_muon_chord_m':length_m,
         'observed_photons':observed,'expected_using_pre_step_beta':expected,
         'observed_over_expected':observed/expected}
    assert abs(observed/expected-1.) < .03
    (BASE/'source_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    {'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
