"""Run on Slurm milano/roma only: exact all-sphere oracle versus SK propagation."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.geometry.detector_geometry import DetectorGeometry


def exact_first_sphere(origins, directions, centers, radius):
    first = np.full(len(origins), -1, dtype=np.int32)
    times = np.full(len(origins), np.inf)
    for start in range(0, len(origins), 128):
        o, d = origins[start:start + 128], directions[start:start + 128]
        delta = centers[None, :, :] - o[:, None, :]
        projection = np.einsum("nsi,ni->ns", delta, d)
        discriminant = radius**2 - np.sum(delta**2, axis=-1) + projection**2
        entry = projection - np.sqrt(np.maximum(discriminant, 0.0))
        valid = (discriminant > 0.0) & (entry > 0.0)
        entry = np.where(valid, entry, np.inf)
        arg = np.argmin(entry, axis=1)
        t = entry[np.arange(len(o)), arg]
        first[start:start + len(o)] = np.where(np.isfinite(t), arg, -1)
        times[start:start + len(o)] = t
    return first, times


def main():
    assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
    geometry = DetectorGeometry.from_config(
        "config/SK_geom_config.json", temperature=0.0, deposit_leg_bound=True)
    detector = geometry.detector
    centers = np.asarray(geometry.sensor_points, dtype=np.float64)
    radius = geometry.sensor_radius
    barrel = np.flatnonzero(np.abs(centers[:, 2]) < detector.H / 4)
    target = barrel[np.argmin(np.abs(centers[barrel, 2]))]
    center = centers[target]
    radial = center.copy()
    radial[2] = 0.0
    radial /= np.linalg.norm(radial)
    direction = np.array([0.0, 0.0, 1.0])
    for offset in (0.4, 0.35, 0.45, 0.6, 0.8, 1.1):
        origin = center - 0.1 * radial - np.array([0.0, 0.0, offset])
        if np.min(np.linalg.norm(centers - origin, axis=1)) > radius:
            truth_idx, truth_t = exact_first_sphere(origin[None], direction[None], centers, radius)
            if truth_idx[0] == target:
                break
    assert np.min(np.linalg.norm(centers - origin, axis=1)) > radius
    result = geometry.propagator(jnp.array(origin[None]), jnp.array(direction[None]))
    truth_idx, truth_t = exact_first_sphere(origin[None], direction[None], centers, radius)
    assert truth_idx[0] == target
    candidates = np.asarray(result["sensor_indices"])[:, 0].tolist()
    returned_t = np.linalg.norm(np.asarray(result["positions"])[0] - origin)
    minimal = dict(origin_m=origin.tolist(), direction=direction.tolist(),
                   expected_first_sensor=int(target), expected_entry_m=float(truth_t[0]),
                   returned_candidates=candidates, returned_stop_m=float(returned_t),
                   expected_sensor_in_candidates=bool(target in candidates),
                   charge_weight_sum=float(np.sum(np.asarray(result["sensor_weights"]))))
    assert target not in candidates, minimal
    print(json.dumps({"minimal_reproduction": minimal}), flush=True)

    rng = np.random.default_rng(1037)
    n = 12000
    metrics = []
    for origin_radius in (0.0, 8.0, 15.0, 16.5, float(detector.r - 0.1)):
        origin_point = np.array([origin_radius, 0.0, 0.0])
        for offset in (0.0, 0.2, 0.4, 0.6, 0.8):
            origin_point[2] = offset
            if np.min(np.linalg.norm(centers - origin_point, axis=1)) > radius:
                break
        assert np.min(np.linalg.norm(centers - origin_point, axis=1)) > radius
        origins = np.broadcast_to(origin_point, (n, 3)).copy()
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        truth_idx, truth_t = exact_first_sphere(origins, directions, centers, radius)
        result = geometry.propagator(jnp.asarray(origins), jnp.asarray(directions))
        cand_idx = np.asarray(result["sensor_indices"])
        cand_inside = np.asarray(result["inside_sensor"])
        cand_times = np.asarray(result["times"]).squeeze(-1)
        masked_times = np.where(cand_inside, cand_times, np.inf)
        slot = np.argmin(masked_times, axis=0)
        found = cand_idx[slot, np.arange(n)]
        found = np.where(np.any(cand_inside, axis=0), found, -1)
        true_hit = truth_idx >= 0
        false_negative = true_hit & (found < 0)
        wrong_first = true_hit & (found >= 0) & (found != truth_idx)
        row = dict(origin_radius_m=origin_radius, origin_m=origin_point.tolist(), total_rays=n,
                   exact_hit_rays=int(np.sum(true_hit)),
                   propagator_hit_rays=int(np.sum(found >= 0)),
                   missing_any_hit=int(np.sum(false_negative)),
                   wrong_first_sensor=int(np.sum(wrong_first)),
                   lost_fraction_of_true_hits=float(np.sum(false_negative) / np.sum(true_hit)),
                   wrong_fraction_of_true_hits=float(np.sum(wrong_first) / np.sum(true_hit)))
        metrics.append(row)
        print(json.dumps(row), flush=True)
    out = dict(slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               partition=os.environ["SLURM_JOB_PARTITION"],
               sensor_radius_m=radius, detector_radius_m=float(detector.r),
               minimal=minimal, population_metrics=metrics)
    Path("audit/wc_data_20261004/transport/geometry_results.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
