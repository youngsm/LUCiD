"""A/B replacement diagnostics; run on milano/roma."""
import json,os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
out=Path(__file__).resolve().parent
result={"job":os.environ["SLURM_JOB_ID"],"cases":{}}
for pdg in (211,-211):
  for energy in (2,200,1000,2000):
    on=pd.read_csv(out/f'on_{pdg}_{energy}.csv');off=pd.read_csv(out/f'off_{pdg}_{energy}.csv')
    expected=500 if energy==2 else 300
    assert len(on)==len(off)==expected, (pdg,energy,len(on),len(off))
    assert (on.replacements==on.resumed).all(), (pdg,energy,"unresumed child")
    assert (on.chain_tracks==1+on.replacements).all(), (pdg,energy,"broken ancestry")
    assert (on.terminals==1).all() and (off.terminals==1).all(), (pdg,energy,"terminal count")
    assert on.max_birth_position_error_mm.max()==0, (pdg,energy,"position discontinuity")
    assert on.max_birth_time_error_ns.max()==0, (pdg,energy,"time discontinuity")
    assert on.max_birth_energy_error_mev.max()<1e-9, (pdg,energy,"energy discontinuity")
    case={"n_on":len(on),"n_off":len(off),"replacement_summary":{}}
    for key in ["replacements","resumed","resumed_stopped","replace_alive","replace_stopbutalive","replace_suspend","replace_suspendwait","replace_postpone"]:
      case["replacement_summary"][key]=int(on[key].sum())
    case["replacement_summary"]["missing_resumed_children"]=int((on.replacements-on.resumed).sum())
    for key in ["max_birth_position_error_mm","max_birth_time_error_ns","max_birth_energy_error_mev"]:
      case["replacement_summary"][key]=float(on[key].max())
    case["terminal_process_on"]=on.terminal_process.value_counts().to_dict()
    case["terminal_process_off"]=off.terminal_process.value_counts().to_dict()
    case["events_without_exactly_one_terminal_on"]=int((on.terminals!=1).sum())
    case["events_without_exactly_one_terminal_off"]=int((off.terminals!=1).sum())
    case["comparison"]={}
    for key in ["photons","chain_photons","chain_length_mm","end_time_ns","end_radius_mm","max_radius_mm","decays","captures","inelastic","muon_births","secondary_births","pi_plus_births","pi_minus_births"]:
      a=on[key].to_numpy(dtype=float);b=off[key].to_numpy(dtype=float)
      se=np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b))
      delta=a.mean()-b.mean()
      case["comparison"][key]={"on_mean":float(a.mean()),"off_mean":float(b.mean()),"on_minus_off":float(delta),"independent_se":float(se),"z":float(delta/se) if se else 0.,"ks_p":float(ks_2samp(a,b).pvalue)}
    result["cases"][f'{pdg}_{energy}']=case
(out/"results.json").write_text(json.dumps(result,indent=2))
lines=[f"Milano job {result['job']}: all continuity, ancestry, resume, and terminal assertions passed.",
       "Differences below are replacement ON minus OFF; intervals are approximate 95% independent-ensemble intervals."]
for name,case in result["cases"].items():
  r=case["replacement_summary"]; c=case["comparison"]; p=c["photons"]
  scale=100/p["off_mean"] if p["off_mean"] else 0
  worst=max(c,key=lambda key:abs(c[key]["z"]))
  lines.append(f"{name}: n={case['n_on']} each; replacements={r['replacements']} (alive={r['replace_alive']}, stopped={r['replace_stopbutalive']}, suspended={r['replace_suspend']}); no missing children.")
  lines.append(f"  photons={p['on_mean']:.3f} vs {p['off_mean']:.3f}; change={scale*p['on_minus_off']:.3f}% +/- {scale*1.96*p['independent_se']:.3f}% (z={p['z']:.3f}). Largest compared |z|: {worst} {c[worst]['z']:.3f}.")
  lines.append(f"  mean pion-chain length={c['chain_length_mm']['on_mean']:.6g} vs {c['chain_length_mm']['off_mean']:.6g} mm; terminal time={c['end_time_ns']['on_mean']:.6g} vs {c['end_time_ns']['off_mean']:.6g} ns.")
  lines.append(f"  terminal processes: ON {case['terminal_process_on']}; OFF {case['terminal_process_off']}.")
(out/"summary.txt").write_text("\n".join(lines)+"\n")
print(json.dumps(result,indent=2),flush=True)
