"""Real Geant4 primary Cherenkov count comparison, with only water RINDEX varied."""
import json
import os
from pathlib import Path

import numpy as np
import uproot

BASE = Path(__file__).resolve().parent


def count_primary(path):
    with uproot.open(path) as f:
        t = f['OpticalPhotons']
        a = t.arrays(['PrimaryEnergy', 'Segment_TrackID', 'Segment_NCherenkov', 'NOpticalPhotons'], library='np')
        counts = np.array([nc[tid == 1].sum() for tid, nc in zip(a['Segment_TrackID'], a['Segment_NCherenkov'])])
        return {'events': t.num_entries, 'primary_energy_MeV': float(a['PrimaryEnergy'][0]), 'primary_cherenkov_total': int(counts.sum()), 'primary_cherenkov_mean': float(counts.mean()), 'primary_cherenkov_per_event': counts.tolist(), 'all_photons_total': int(a['NOpticalPhotons'].sum())}


def main():
    assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
    rows = []
    for energy in (52, 54):
        old = count_primary(BASE.parent / 'geant4_physics' / f'mu{energy}.root')
        new = count_primary(BASE / f'mu{energy}_measured.root')
        assert old['events'] == new['events'] == 100
        assert old['primary_energy_MeV'] == new['primary_energy_MeV'] == energy
        rows.append({'kinetic_MeV': energy, 'original': old, 'measured_phase_index': new, 'primary_cherenkov_loss_fraction': 1 - old['primary_cherenkov_total'] / new['primary_cherenkov_total']})
    assert rows[0]['original']['primary_cherenkov_total'] == 0
    assert rows[0]['measured_phase_index']['primary_cherenkov_total'] > 0
    assert rows[1]['primary_cherenkov_loss_fraction'] > .5
    result = {'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'partition': os.environ['SLURM_JOB_PARTITION'], 'rows': rows}
    print(json.dumps(result, indent=2), flush=True)
    (BASE / 'index_ab_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
