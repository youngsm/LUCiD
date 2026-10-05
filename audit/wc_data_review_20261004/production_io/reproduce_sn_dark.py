from pathlib import Path
import json
import numpy as np
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation.digitizer import resolve_model_config
from lucid.sources.event_generation import _merge_pileup_streams, _group_interactions_by_gap
from reproduce_host import stream

HERE = Path(__file__).resolve().parent
geometry = DetectorGeometry.from_config("config/SK_WAND_geom_config.json")
n_sensors = len(geometry.sensor_points)
model = resolve_model_config({"model": "ski"})
streams = [stream(1, 0.0, 1100.0), stream(2, 1000.0, 100.0)]
for s in streams:
    for name, value in s["deposits"].items():
        s["deposits"][name] = np.repeat(value, 3)
    s["deposits"]["sensor_idx"] = np.array([0, 1, 2])
    s["segments"]["n_cherenkov"][:] = 3
    for t in s["meaningful_tracks"].values():
        t["n_cherenkov"] = 3
split_dark, joint_dark = [], []
for seed in range(256):
    dark = 0.0
    for group in _group_interactions_by_gap(np.array([0., 1000.]), 800.):
        merged = _merge_pileup_streams(
            [streams[i] for i in group], n_sensors=n_sensors, apply_smearing=False,
            digitizer_model=model, digi_rng=np.random.default_rng(seed*4 + group[0]),
            detector_bounds=None, readout_pad_ns=500.0)
        h = merged["hits_sparse"]
        dark += float(h["PE"][h["emission_process"] == 2].sum())
    split_dark.append(dark)
    merged = _merge_pileup_streams(
        streams, n_sensors=n_sensors, apply_smearing=False, digitizer_model=model,
        digi_rng=np.random.default_rng(seed*4 + 2), detector_bounds=None,
        readout_pad_ns=500.0)
    h = merged["hits_sparse"]
    joint_dark.append(float(h["PE"][h["emission_process"] == 2].sum()))
expected_single = n_sensors * 4.2e-6 * 1000.0
results = {"n_sensors": n_sensors, "n_ensembles": 256,
    "expected_dark_one_span": expected_single,
    "mean_split_dark": float(np.mean(split_dark)),
    "mean_joint_dark": float(np.mean(joint_dark)),
    "dark_ratio": float(np.mean(split_dark)/np.mean(joint_dark))}
assert 1.85 < results["dark_ratio"] < 2.15
(HERE / "sn_dark_results.json").write_text(json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
