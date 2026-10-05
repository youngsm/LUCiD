from pathlib import Path
import json
import h5py
import numpy as np
from lucid.sources.root_reader import _read_event_raw
from lucid.production.run_job import _run_lucid

HERE = Path(__file__).resolve().parent
ROOT = HERE / "displaced_shifted.root"
raw = _read_event_raw(ROOT, 0)
primary = [t for t in raw["track_info_dict"].values() if t["parent_id"] == 0]
assert len(primary) == 1
position = primary[0]["position"]
assert np.allclose(position, [5., -2., 3.])
_run_lucid(
    root_file=ROOT, output_dir=HERE/"displaced_output",
    config={"name": "SK_WAND_displaced_audit",
            "lucid_options": {"apply_translation": False, "apply_smearing": True,
                              "pad_size_buckets": [2048]}},
    file_index=0, n_events=1, master_seed=19, job_id=1, detector="SK_WAND")
with h5py.File(HERE/"displaced_output/labl/wc_labl_0000.h5") as f:
    grp = f["event_000/per_interaction"]
    vertex = np.array([grp["vertex_"+axis][0] for axis in "xyz"])
with h5py.File(HERE/"displaced_output/step/wc_step_0000.h5") as f:
    grp = f["event_000"]
    step = np.array([grp["start_"+axis][0] for axis in "xyz"])
assert np.allclose(step, position)
assert np.array_equal(vertex, [0., 0., 0.])
result = {"source_primary_position_m": position.tolist(),
          "first_step_position_m": step.tolist(),
          "saved_interaction_vertex_m": vertex.tolist(),
          "vertex_error_m": float(np.linalg.norm(position-vertex)),
          "source_photons": int(raw["photon_origins"].shape[0])}
(HERE/"vertex_truth_results.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
