"""Physical oracle; baseline must FAIL, calibrated water variant must PASS.

Run on milano/roma after run_audit.sbatch and run_spin_water.sbatch:
  python test_spin_physics.py baseline
  python test_spin_physics.py water
"""
from pathlib import Path
import json
import math
import sys
results = json.loads((Path(__file__).resolve().parent/'spin_results.json').read_text())
r = results[sys.argv[1]]
# Michel formula integrated above total positron energy 40 MeV + its rest mass.
# mu+ spin points opposite its birth direction for pion decay at rest.
retention = .718
x = (40. + .51099895) / ((105.6583715**2 + .51099895**2)/(2.*105.6583715))
asymmetry = (1./3. + 2.*x**3/3. - x**4)/(1.-2.*x**3+x**4)
expected = .5 - retention*asymmetry/4.
observed = r['high_energy_same_hemisphere_fraction']
se = math.sqrt(expected*(1.-expected)/r['high_energy_count'])
print(f'{sys.argv[1]}: observed high-energy forward fraction={observed:.6f}; '
      f'water Michel oracle={expected:.6f}; statistical sigma={se:.6f}')
assert abs(observed-expected) < 4*se + .01, 'Missing physical pion/muon spin correlation'
