"""Independent deterministic input-photon conservation check on SK_WAND geometry."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import DetectorParams
from lucid.simulation import setup_event_simulator
from lucid.sources.event_builder import _trace_event_bucketed
from lucid.sources.event_generation import _random_rotation_matrix, _rotate_event_raw


BASE = Path(__file__).resolve().parent


def main():
    assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
    simulator = setup_event_simulator(
        "config/SK_WAND_geom_config.json", 0, K=12, is_data=True,
        temperature=0.0, wavelength_mode=False, charge_resolution=None,
        hit_mode="per_segment", deposit_leg_bound=True,
        default_detector_params=DetectorParams.from_flat(num_sensors=11096,
            qe=1.0, tts=0.0, scatter_length=1e12,
            mie_scatter_length=1e12, absorption_length=1e12,
            wall_reflection_rate=0.0, sensor_reflection_rate=0.0,
        ),
    )
    geometry = simulator.det_geom
    centers = np.asarray(geometry.sensor_points, dtype=np.float64)
    target = np.argmin(np.linalg.norm(centers, axis=1))
    center = centers[target]
    direction = center / np.linalg.norm(center)
    origin = center - 1.0 * direction
    flight_time = (1.0 - geometry.sensor_radius) / geometry.speed_of_light
    n = 513
    origins = np.broadcast_to(origin, (n, 3)).copy()
    directions = np.broadcast_to(direction, (n, 3)).copy()
    times = np.linspace(0.0, 200_000.0, n).astype(np.float32)
    wavelengths = np.linspace(280.0, 670.0, n).astype(np.float32)
    segment_ids = np.arange(n, dtype=np.int32)
    result = _trace_event_bucketed(
        simulator, origins, directions, times, wavelengths, segment_ids,
        geometry.num_sensors, (256,), jax.random.PRNGKey(31415),
    )
    pe, t, tr, w, tt, ttr, si, segi, gid = result
    kept = w > 0
    observed_gid = gid[kept]
    residual = tt[kept] - (times[observed_gid] + flight_time)
    summary = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "active_photons": n,
        "total_PE": float(pe.sum()),
        "detected_flat_records": int(kept.sum()),
        "all_source_ids_exactly_once": bool(np.array_equal(np.sort(observed_gid), np.arange(n))),
        "all_segment_ids_preserved": bool(np.array_equal(segi[kept], segment_ids[observed_gid])),
        "all_sensor_ids_are_target": bool(np.all(si[kept] == target)),
        "max_emission_time_preserved_ns": float(times[observed_gid].max()),
        "max_arrival_residual_ns": float(np.abs(residual).max()),
        "expected_flight_ns": float(flight_time),
        "emission_time_endpoints_ns": [float(times.min()), float(times.max())],
    }
    assert abs(pe.sum() - n) < 0.002, summary
    assert summary["all_source_ids_exactly_once"], summary
    assert summary["all_segment_ids_preserved"], summary
    assert summary["all_sensor_ids_are_target"], summary
    assert np.abs(residual).max() < 0.032, summary

    # The vertex rotation must preserve cone opening angles and distances.
    rng = np.random.default_rng(13459)
    nrot = 20_000
    rotated_axes = np.stack([_random_rotation_matrix(rng)[:, 2] for _ in range(nrot)])
    rotation_moments = {
        "draws": nrot,
        "mean_axis": rotated_axes.mean(axis=0).tolist(),
        "second_moment_axis": np.mean(rotated_axes**2, axis=0).tolist(),
        "expected_mean": [0.0, 0.0, 0.0],
        "expected_second_moment": [1.0 / 3] * 3,
    }
    assert np.abs(rotated_axes.mean(axis=0)).max() < 0.02
    assert np.abs(np.mean(rotated_axes**2, axis=0) - 1.0 / 3).max() < 0.02
    R = _random_rotation_matrix(rng)
    raw = {
        "photon_origins": origins[:3].copy(),
        "photon_directions": directions[:3].copy(),
        "segments_raw": {
            **{f"start_{k}_mm": origins[:3, i] * 1000 for i, k in enumerate("xyz")},
            **{f"end_{k}_mm": (origins[:3, i] + directions[:3, i]) * 1000 for i, k in enumerate("xyz")},
            **{f"dir_{k}": directions[:3, i].copy() for i, k in enumerate("xyz")},
        },
        "track_info_dict": {1: {"position": origin.copy(), "direction": direction.copy()}},
    }
    _rotate_event_raw(raw, R)
    expected_origins = origins[:3] @ R.T
    expected_dirs = directions[:3] @ R.T
    np.testing.assert_allclose(raw["photon_origins"], expected_origins, atol=2e-6)
    np.testing.assert_allclose(raw["photon_directions"], expected_dirs, atol=2e-7)
    actual_seg_origins = np.stack([raw["segments_raw"][f"start_{k}_mm"] for k in "xyz"], axis=1) / 1000
    np.testing.assert_allclose(actual_seg_origins, expected_origins, atol=2e-6)
    np.testing.assert_allclose(raw["track_info_dict"][1]["position"], expected_origins[0], atol=2e-6)
    summary["rotation_moments"] = rotation_moments
    summary["rotation_photon_segment_track_coherence_passed"] = True
    print(json.dumps(summary, indent=2), flush=True)
    (BASE / "source_boundary_results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
