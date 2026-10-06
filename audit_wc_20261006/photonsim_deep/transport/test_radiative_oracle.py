"""All numbers come from capture CSV and independent primary measurements."""
from pathlib import Path
import json
import math
import numpy as np

b=Path(__file__).resolve().parent
d=np.genfromtxt(b/'pi_minus_cold_water.csv',delimiter=',',names=True)
n_o=int((d['pion_captures']-d['pion_hydrogen']).sum())
n_h=int(d['pion_hydrogen'].sum())
hard=int(d['pion_hard_gamma'].sum())
assert hard==0
# Bistirlich PRC5,1867 printedp24 reports two hard components separately:
# ground-state transition0.15+/-0.03%, giant-resonance0.25+/-0.06%.
# Both exceed100MeV and do not depend on extrapolation below50MeV.
hard_lines=.0015+.0025
# The two absolute rates share normalization uncertainty. Sum their errors
# as a conservative bound; quadrature would assume unknown independence.
hard_lines_sigma=.0003+.0006
conservative=hard_lines-3*hard_lines_sigma
prob_zero=(1-conservative)**n_o
assert prob_zero<.05
# This physical assertion fails even after fixing target selection.
physical_assertion_passes=hard>0
assert not physical_assertion_passes
panofsky=1.546
gamma_h=1/(1+panofsky)
baseline=np.genfromtxt(b/'pi_minus_cold_baseline.csv',delimiter=',',names=True)
nh_base=int(baseline['pion_hydrogen'].sum())
results={
    'oxygen_captures_after_target_fix':n_o,
    'hard_capture_gamma_observed':hard,
    'zero_count_95pct_upper_probability':1-.05**(1/n_o),
    'measured_hard_lines_probability':hard_lines,
    'measured_hard_lines_uncertainty_upper_bound':hard_lines_sigma,
    'expected_hard_line_events_only':n_o*hard_lines,
    'zero_probability_at_3sigma_low_measured_lines':prob_zero,
    'hydrogen_baseline_captures':nh_base,
    'hydrogen_em_capture_family_probability_expected':gamma_h,
    'hydrogen_em_capture_family_events_expected_for_baseline_H_choices':nh_base*gamma_h,
    'hydrogen_em_capture_family_probability_per_real_water_stop':.00445*gamma_h,
    'panofsky_convention':'Radiative family includes small internal-conversion channel; real-gamma-only branching is slightly lower.',
    'target_only_fix_still_fails_radiative_oracle':not physical_assertion_passes,
}
(b/'radiative_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
