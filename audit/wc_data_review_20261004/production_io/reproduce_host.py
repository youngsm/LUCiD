"""Executed CPU checks of real host-side event generation and HDF5 APIs."""
from pathlib import Path
import json

import h5py
import numpy as np

from lucid.simulation.digitizer import resolve_model_config
from lucid.sources.event_generation import (
    _group_interactions_by_gap,
    _merge_pileup_streams,
    _concat_triggered_chunks,
    _keep_chunk_min_physics_hits,
)
from lucid.sources.reader import read_sensor_event, read_labl_event
from lucid.sources.writer import save_sensor_event, save_labl_event

HERE = Path(__file__).resolve().parent


def stream(tid, t0, arrival):
    particle = {"genealogy": [tid], "extended_genealogy": [tid],
                "track_info": {"category": 0}}
    track = {"track_id": tid, "parent_id": 0, "pdg": 11,
             "initial_energy": 10.0, "n_cherenkov": 1, "n_segments": 1}
    seg = {key: np.array([0.0]) for key in (
        "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
        "dir_x", "dir_y", "edep", "beta_start")}
    seg.update(dir_z=np.array([1.0]), time=np.array([t0]),
               n_cherenkov=np.array([1]), n_segments=1)
    dep = {"sensor_idx": np.array([0]), "charge": np.array([1.0]),
           "t_true": np.array([arrival]), "t_reco": np.array([arrival]),
           "particle_idx": np.array([0]), "segment_idx": np.array([0]),
           "emission_process": np.array([0])}
    meta = {"t0": t0, "vertex_xyz": np.zeros(3), "source_type": 2,
            "neutrino_pdg": 12, "neutrino_energy_MeV": 10.0,
            "primary_track_ids": [tid], "primary_pdgs": [11],
            "primary_energies": [10.0]}
    return {"particles": [particle], "meaningful_tracks": {tid: track},
            "segments": seg, "deposits": dep, "t0": t0,
            "interaction_meta": meta}


def merge(streams):
    return _merge_pileup_streams(
        streams, n_sensors=1, apply_smearing=False,
        digitizer_model=resolve_model_config({"model": "ski", "dark_rate_khz": 0}),
        digi_rng=np.random.default_rng(9),
        detector_bounds={"type": "cylinder", "radius": 16.9, "height": 36.2},
    )


def test_sn_chunk_overlap():
    # A delayed photon from interaction 0 arrives together with prompt light
    # from interaction 1. Both are within the retained 100-us readout bound.
    streams = [stream(1, 0.0, 1100.0), stream(2, 1000.0, 100.0)]
    groups = _group_interactions_by_gap(np.array([0.0, 1000.0]), 800.0)
    chunks = []
    for group in groups:
        part = merge([streams[i] for i in group])
        sd, hs, sh, pw = _keep_chunk_min_physics_hits(part, 1)
        part.update(sensor_digits=sd, hits_sparse=hs,
                    segment_sensor_hits=sh, per_window=pw)
        chunks.append(part)
    split = _concat_triggered_chunks(chunks)
    joint = merge(streams)
    assert groups == [[0], [1]]
    assert split["sensor_digits"]["PE"].tolist() == [1.0, 1.0]
    assert split["sensor_digits"]["T"].tolist() == [1100.0, 1100.0]
    assert joint["sensor_digits"]["PE"].tolist() == [2.0]
    assert joint["sensor_digits"]["T"].tolist() == [1100.0]

    # The real writer and reader also preserve the duplicated digits and windows.
    split["source_event_idx"] = 0
    with h5py.File(HERE / "split_sensor.h5", "w") as sf:
        save_sensor_event(sf, split, 0)
    with h5py.File(HERE / "split_labl.h5", "w") as lf:
        save_labl_event(lf, split, 0)
    written = read_sensor_event(HERE / "split_sensor.h5", 0)
    labels = read_labl_event(HERE / "split_labl.h5", 0)
    assert written["PE"].tolist() == [1.0, 1.0]
    assert labels["per_window"]["window_start"].tolist() == [1100.0, 1100.0]
    return {"groups": groups,
            "split_PE": split["sensor_digits"]["PE"].tolist(),
            "split_T": split["sensor_digits"]["T"].tolist(),
            "joint_PE": joint["sensor_digits"]["PE"].tolist(),
            "joint_T": joint["sensor_digits"]["T"].tolist(),
            "written_n_windows": int(labels["per_window"]["n_windows"])}


def test_time_and_ancestry_roundtrip():
    streams = [stream(7, 1e10 + 0.25, 80.125), stream(19, 1e10 + 40.25, 40.625)]
    merged = merge(streams)
    merged["source_event_idx"] = 4
    with h5py.File(HERE / "roundtrip_sensor.h5", "w") as sf:
        save_sensor_event(sf, merged, 0)
    with h5py.File(HERE / "roundtrip_labl.h5", "w") as lf:
        save_labl_event(lf, merged, 0)
    sd = read_sensor_event(HERE / "roundtrip_sensor.h5", 0)
    labl = read_labl_event(HERE / "roundtrip_labl.h5", 0)
    assert sd["T"].tolist() == [1e10 + 80.375]
    assert sd["T"].dtype == np.float64
    assert sd["PE"].tolist() == [2.0]
    assert labl["per_track"]["ancestor"].tolist() == [7, 19]
    assert labl["per_track"]["interaction"].tolist() == [0, 1]
    assert labl["per_particle"]["interaction_idx"].tolist() == [0, 1]
    assert labl["per_interaction"]["primary_track_ids_data"].tolist() == [7, 19]
    return {"T_dtype": str(sd["T"].dtype), "T": sd["T"].tolist(),
            "PE": sd["PE"].tolist(),
            "interaction": labl["per_track"]["interaction"].tolist()}


if __name__ == "__main__":
    results = {"sn_chunk_overlap": test_sn_chunk_overlap(),
               "time_ancestry_roundtrip": test_time_and_ancestry_roundtrip()}
    (HERE / "host_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
