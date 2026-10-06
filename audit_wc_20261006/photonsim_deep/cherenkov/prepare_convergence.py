import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') in {'milano','roma'}
base=Path(__file__).resolve().parent
for name,step in [('default',10),('small',.1),('smaller',.01)]:
    (base/f'conv_{name}.mac').write_text(f'''/output/filename {base}/conv_{name}.root
/process/optical/cerenkov/setMaxBetaChange {step}
/run/initialize
/random/setSeeds 493821 128394
/photon/storeIndividual false
/gun/clearPrimaries
/gun/addPrimary e- 0.3 MeV
/gun/position 0 0 0 m
/gun/direction 0 0 1
/run/beamOn 10000
''')
