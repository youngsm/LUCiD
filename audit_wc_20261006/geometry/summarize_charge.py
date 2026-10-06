import json, math, os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
out=Path(__file__).resolve().parent
data=json.loads((out/"charge_bias_results_30seeds.json").read_text())
keys=["total_pe","pe_after150ns","pe_after300ns"]
summary={"photon_draws":data["n_per_seed"]*len(data["seeds"]),"job":data["job"],"source":data["source"]}
for i,name in enumerate(keys):
    baseline=sum(r[name] for r in data["runs"]["baseline"])
    fixed=sum(r[name] for r in data["runs"]["fixed"])
    delta=sum(r["paired_delta"][i] for r in data["runs"]["fixed"])
    sigma=math.sqrt(sum(r["paired_sigma"][i]**2 for r in data["runs"]["fixed"]))
    summary[name]={"baseline":baseline,"fixed":fixed,"delta":delta,"paired_sigma":sigma,"relative_delta":delta/baseline,"significance":delta/sigma}
(out/"charge_bias_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
