from pathlib import Path
import json
import h5py
from lucid.production.run_job import _run_lucid

HERE = Path(__file__).resolve().parent
ROOT = HERE / "subthreshold.root"
config = {"name": "SK_WAND_run_job_audit", "primary_source": "particles",
          "lucid_options": {"apply_translation": False, "apply_smearing": True,
                            "pad_size_buckets": [256]}}
_run_lucid(root_file=ROOT, output_dir=HERE / "run_job_output", config=config,
           file_index=0, n_events=1, master_seed=19, job_id=1, detector="SK_WAND")
def attrs(folder):
    with h5py.File(HERE / folder / "sensor/wc_sensor_0000.h5") as f:
        a = f["config"].attrs
        return {"digitizer_model": a["digitizer_model"], "trigger": a["trigger"],
                "n_events": int(a["n_events"]),
                "n_hits": [int(f[e].attrs["n_hits"]) for e in f if e.startswith("event_")]}
result = {"direct_cli": attrs("cli_output"), "run_job": attrs("run_job_output")}
assert result["direct_cli"]["digitizer_model"] == "basic"
assert result["direct_cli"]["trigger"] == "none"
assert result["direct_cli"]["n_events"] == 1
assert result["run_job"]["digitizer_model"] == "ski"
assert result["run_job"]["trigger"] == "sliding_window"
assert result["run_job"]["n_events"] == 0
(HERE / "cli_comparison_results.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
