"""Independent WC ROOT/chunk/units/time invariants; execute only in Slurm CPU.

This is a coverage diagnostic, not a claimed failing regression. The ROOT
oracle starts from chosen dimensional values, while the actual transport
oracle uses analytic radial ray/sphere entry distances.
"""
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import uproot

from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources.event_builder import _trace_event_bucketed, gather_photon_deposits
from lucid.sources.event_generation import _random_rotation_matrix, _rotate_event_raw
from lucid.sources.root_reader import _read_photons_for_event

BASE = Path(__file__).resolve().parent


def root_chunk_check():
    # Event 0 chunks deliberately appear noncontiguously and in reverse order.
    # Distinct physical mm/ns/nm data catch unit, order, and branch mix-ups.
    n = 7
    origins_m = np.arange(21, dtype=np.float32).reshape(n, 3) - 12
    directions = np.eye(3, dtype=np.float32)[np.arange(n) % 3]
    times_ns = np.arange(n, dtype=np.float32) * 37.5
    wavelengths_nm = np.linspace(290, 660, n, dtype=np.float32)
    selected = [np.arange(4, 7), np.arange(0, 1), np.arange(0, 4)]
    branches = {"EventID": "int32", "ChunkStartID": "int32"}
    cols = {}
    for axis, suffix in enumerate("XYZ"):
        cols[f"PhotonPos{suffix}"] = origins_m[:, axis] * 1000
        cols[f"PhotonDir{suffix}"] = directions[:, axis]
    cols["PhotonTime"] = times_ns
    cols["PhotonWavelength"] = wavelengths_nm
    branches.update({k: "var * float32" for k in cols})
    path = BASE / "synthetic_chunks.root"
    with uproot.recreate(path) as f:
        f.mktree("OpticalPhotonsRaw", branches)
        f["OpticalPhotonsRaw"].extend({
            "EventID": np.array([0, 1, 0], np.int32),
            "ChunkStartID": np.array([4, 0, 0], np.int32),
            **{k: ak.Array([v[s] for s in selected]) for k, v in cols.items()},
        })
    with uproot.open(path) as f:
        got = _read_photons_for_event(f["OpticalPhotonsRaw"], 0)
        empty = _read_photons_for_event(f["OpticalPhotonsRaw"], 2)
    for actual, expected in zip(got, (origins_m, directions, times_ns, wavelengths_nm)):
        np.testing.assert_array_equal(actual, expected)
    assert [a.shape for a in empty] == [(0, 3), (0, 3), (0,), (0,)]
    return {"photons": n, "noncontiguous_reverse_chunks": "pass", "mm_to_m": "pass",
            "directions_ns_nm_unchanged": "pass", "missing_event_empty": "pass"}


def flatten_and_padding_check():
    # Uniquely tag each input, and return positive data only in later flat slabs.
    # This catches losing all but the first step/candidate and including padding.
    n = 523
    pos = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 100
    dirs = np.eye(3, dtype=np.float32)[np.arange(n) % 3]
    times = np.arange(n, dtype=np.float32) + 5
    wavelengths = np.arange(n, dtype=np.float32) / 10 + 300
    segments = np.arange(n, dtype=np.int32) + 900
    seen = []

    def capture(_track, _key, data):
        k = int(data["N"])
        seen.append({name: np.asarray(data[name])[:k] for name in
                     ("photon_origins", "photon_directions", "photon_times", "wavelengths")})
        b = data["photon_origins"].shape[0]
        # Six (step,candidate) slabs in kernel C order; only slab 5 active.
        weight = np.zeros((6, b), np.float32)
        weight[5, :] = 1.0
        time = np.broadcast_to(np.asarray(data["photon_times"]), (6, b)).copy()
        seg = np.broadcast_to(np.asarray(data["photon_segment_index"]), (6, b)).copy()
        sensor = np.zeros((6, b), np.int32)
        return (jnp.array([k], dtype=jnp.float32), jnp.array([1.]), jnp.array([1.]),
                jnp.asarray(weight.ravel()), jnp.asarray(time.ravel()), jnp.asarray(time.ravel()),
                jnp.asarray(sensor.ravel()), jnp.asarray(seg.ravel()))

    result = _trace_event_bucketed(capture, pos, dirs, times, wavelengths, segments,
                                   1, (8, 256), jax.random.PRNGKey(43))
    _, _, _, weights, observed_t, _, _, observed_seg, gid = result
    for field, expected in (("photon_origins", pos * 100), ("photon_directions", dirs),
                            ("photon_times", times), ("wavelengths", wavelengths)):
        np.testing.assert_array_equal(np.concatenate([c[field] for c in seen]), expected)
    positive = weights > 0
    np.testing.assert_array_equal(np.sort(gid[positive]), np.arange(n))
    np.testing.assert_array_equal(observed_seg[positive], segments[gid[positive]])
    np.testing.assert_array_equal(observed_t[positive], times[gid[positive]])
    # Categorization sentinels must not kill the actual PE stream.
    deposits = gather_photon_deposits([{"particle_data": {"photon_records_filtered": {
        "qe_weight": np.ones(2), "qe_time": np.array([20., 21.]),
        "sensor_idx": np.array([0, 0]), "particle_idx": np.array([-1, 0]),
        "seg_idx_filtered": np.array([-1, 0]),
    }}}])
    assert deposits["charge"].sum() == 2
    return {"photons": n, "chunks": len(seen), "later_step_slab_preserved": "pass",
            "m_to_cm_boundary": "pass", "padding_omitted": "pass", "orphan_pe_retained": "pass"}


def transport_check():
    # Perfect optics isolate input handoff from stochastic optical calibration.
    sim = setup_event_simulator(
        "config/SK_WAND_geom_config.json", 0, K=12, is_data=True,
        temperature=0., wavelength_mode=False, charge_resolution=None,
        hit_mode="per_segment", deposit_leg_bound=True,
        default_detector_params=DetectorParams.from_flat(
            num_sensors=11096, qe=1., tts=0., scatter_length=1e12,
            mie_scatter_length=1e12, absorption_length=1e12,
            wall_reflection_rate=0., sensor_reflection_rate=0.))
    g = sim.det_geom
    centers = np.asarray(g.sensor_points, np.float64)
    targets = np.argsort(np.linalg.norm(centers, axis=1))[:513]
    direction = centers[targets] / np.linalg.norm(centers[targets], axis=1)[:, None]
    source_distance_m = np.linspace(.3, 1.2, len(targets))
    origins = centers[targets] - source_distance_m[:, None] * direction
    times = np.linspace(0., 200000., len(targets)).astype(np.float32)
    wls = np.linspace(280., 670., len(targets)).astype(np.float32)
    result = _trace_event_bucketed(sim, origins, direction, times, wls,
        np.arange(len(targets), dtype=np.int32), g.num_sensors, (256,), jax.random.PRNGKey(92))
    pe, _, _, weights, observed_t, _, sensor, seg, gid = result
    kept = weights > 0
    np.testing.assert_array_equal(np.sort(gid[kept]), np.arange(len(targets)))
    np.testing.assert_array_equal(sensor[kept], targets[gid[kept]])
    np.testing.assert_array_equal(seg[kept], gid[kept])
    flight_ns = (source_distance_m - g.sensor_radius) / g.speed_of_light
    residual = observed_t[kept] - (times[gid[kept]].astype(float) + flight_ns[gid[kept]])
    assert abs(pe.sum() - len(targets)) < .002
    assert max(abs(residual)) < .02
    # The initial source mask must also work through the public host boundary.
    outside = centers[targets] + 2 * direction
    out_result = _trace_event_bucketed(sim, outside, -direction, times, wls,
        np.arange(len(targets), dtype=np.int32), g.num_sensors, (256,), jax.random.PRNGKey(92))
    assert np.count_nonzero(out_result[3]) == 0
    return {"photons": len(targets), "distinct_target_pmts": len(targets),
            "observed_PE": float(pe.sum()), "one_detected_record_per_input": "pass",
            "max_independent_arrival_residual_ns": float(max(abs(residual))),
            "external_sources_detected": int(np.count_nonzero(out_result[3]))}


def rotation_check():
    rng = np.random.default_rng(303)
    axes = np.stack([_random_rotation_matrix(rng)[:, 2] for _ in range(10000)])
    assert abs(axes.mean(axis=0)).max() < .025
    assert abs((axes ** 2).mean(axis=0) - 1/3).max() < .025
    p = rng.normal(size=(20, 3))
    d = rng.normal(size=(20, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    raw = {"photon_origins": p.copy(), "photon_directions": d.copy(),
           "segments_raw": {**{f"start_{a}_mm": p[:, i]*1000 for i, a in enumerate("xyz")},
                            **{f"end_{a}_mm": (p[:, i]+d[:, i])*1000 for i, a in enumerate("xyz")},
                            **{f"dir_{a}": d[:, i].copy() for i, a in enumerate("xyz")}},
           "track_info_dict": {1: {"position": p[0].copy(), "direction": d[0].copy()}}}
    R = _random_rotation_matrix(rng)
    _rotate_event_raw(raw, R)
    np.testing.assert_allclose(raw["photon_origins"], p @ R.T, atol=4e-7)
    np.testing.assert_allclose(raw["photon_directions"], d @ R.T, atol=2e-7)
    for kind, original in (("start", p), ("end", p+d)):
        got = np.stack([raw["segments_raw"][f"{kind}_{a}_mm"] for a in "xyz"], axis=1)/1000
        np.testing.assert_allclose(got, original @ R.T, atol=1e-12)
    np.testing.assert_allclose(raw["track_info_dict"][1]["direction"], R @ d[0], atol=1e-12)
    return {"coherent_photon_segment_track_transform": "pass", "SO3_axis_mean": axes.mean(0).tolist(),
            "SO3_axis_second_moment": (axes**2).mean(0).tolist()}


if __name__ == "__main__":
    output = {"slurm_job": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
              "root_chunks": root_chunk_check(), "flat_handoff": flatten_and_padding_check(),
              "rotation": rotation_check(), "actual_data_transport": transport_check()}
    print(json.dumps(output, indent=2), flush=True)
    (BASE / "handoff_results.json").write_text(json.dumps(output, indent=2) + "\n")
