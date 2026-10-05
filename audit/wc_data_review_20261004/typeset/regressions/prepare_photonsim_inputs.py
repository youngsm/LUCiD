"""Create portable deterministic source macros and a forward GENIE input."""
import argparse
import os
from pathlib import Path
import numpy as np
import uproot

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', type=Path)
args = parser.parse_args()
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
n = 20000
pdg = np.tile(np.array([14, 14], np.int32), (n, 1))
status = np.tile(np.array([0, 1], np.int32), (n, 1))
p4 = np.zeros((n, 2, 4), np.float64); p4[:, :, 2:] = .001
with uproot.recreate(output / 'forward_neutrinos.gtrac.root') as f:
    t = f.mktree('gRooTracker', {'StdHepN': 'int32', 'StdHepPdg': '2 * int32',
        'StdHepStatus': '2 * int32', 'StdHepP4': '2 * 4 * float64'})
    t.extend({'StdHepN': np.full(n, 2, np.int32), 'StdHepPdg': pdg,
              'StdHepStatus': status, 'StdHepP4': p4})
macros = {
    'gamma_prompt': '/gun/addPrimary gamma 2.2 MeV\n/gun/time 0 ns\n/run/beamOn 30',
    'gamma_delayed': '/gun/addPrimary gamma 2.2 MeV\n/gun/time 200000 ns\n/run/beamOn 30',
    'pion_deflection': '/gun/addPrimary pi+ 1000 MeV\n/run/beamOn 20',
    'genie_isotropic': f'/gun/genieInput {output / "forward_neutrinos.gtrac.root"}\n/gun/genieIsotropic true\n/run/beamOn {n}',
    'thermal_neutron': '/gun/addPrimary neutron 0.025 eV\n/run/beamOn 1000'}
for name, body in macros.items():
    header = (f'/output/filename {output / (name + ".root")}\n/run/initialize\n'
        '/random/setSeeds 314159 271828\n/photon/storeIndividual true\n'
        '/gun/clearPrimaries\n/gun/randomDirection false\n')
    (output / (name + '.mac')).write_text(header + body + '\n')
print(output)
