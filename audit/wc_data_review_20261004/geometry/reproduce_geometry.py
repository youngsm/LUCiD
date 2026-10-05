"""Compare the production SK_WAND propagator with a bounded all-PMT oracle."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.geometry.detector_geometry import DetectorGeometry


def cylinder_exit(origins, directions, radius, height):
    """Independent float64 finite-cylinder first exit for interior origins."""
    a = np.sum(directions[:, :2] ** 2, axis=1)
    b = np.sum(origins[:, :2] * directions[:, :2], axis=1)
    c = np.sum(origins[:, :2] ** 2, axis=1) - radius**2
    wall = np.divide(-b + np.sqrt(np.maximum(b*b - a*c, 0)), a,
                     out=np.full(len(a), np.inf), where=a > 0)
    wall_z = origins[:, 2] + wall * directions[:, 2]
    wall = np.where(np.abs(wall_z) <= height/2, wall, np.inf)
    cap_z = np.where(directions[:, 2] >= 0, height/2, -height/2)
    cap = np.divide(cap_z-origins[:, 2], directions[:, 2],
                    out=np.full(len(a), np.inf), where=directions[:, 2] != 0)
    cap_xy = origins[:, :2] + cap[:, None]*directions[:, :2]
    cap = np.where(np.sum(cap_xy**2, axis=1) <= radius**2, cap, np.inf)
    return np.minimum(wall, cap)


def exact_first_hit(origins, directions, centers, sensor_radius, t_exit):
    """Intersect every sphere; require the entry to be on the water leg."""
    first = np.full(len(origins), -1, dtype=np.int32)
    distance = np.full(len(origins), np.inf)
    unbounded = np.full(len(origins), -1, dtype=np.int32)
    for start in range(0, len(origins), 128):
        o, d = origins[start:start+128], directions[start:start+128]
        delta = centers[None, :, :] - o[:, None, :]
        projection = np.einsum('nsi,ni->ns', delta, d)
        perpendicular = delta-projection[:, :, None]*d[:, None, :]
        disc = sensor_radius**2-np.sum(perpendicular**2, axis=-1)
        entry = projection-np.sqrt(np.maximum(disc, 0))
        positive = (disc > 0) & (entry > 0)
        line_entry = np.where(positive, entry, np.inf)
        line_arg = np.argmin(line_entry, axis=1)
        unbounded[start:start+len(o)] = np.where(
            np.isfinite(line_entry[np.arange(len(o)), line_arg]), line_arg, -1)
        valid = positive & (entry <= t_exit[start:start+len(o), None])
        entry = np.where(valid, entry, np.inf)
        arg = np.argmin(entry, axis=1)
        time = entry[np.arange(len(o)), arg]
        first[start:start+len(o)] = np.where(np.isfinite(time), arg, -1)
        distance[start:start+len(o)] = time
    return first, distance, unbounded


def analyze(geometry, origins, directions, label):
    detector = geometry.detector
    centers = np.asarray(geometry.sensor_points, dtype=np.float64)
    directions = directions/np.linalg.norm(directions, axis=1, keepdims=True)
    exits = cylinder_exit(origins, directions, detector.r, detector.H)
    truth_idx, truth_dist, unbounded = exact_first_hit(
        origins, directions, centers, geometry.sensor_radius, exits)
    out = geometry.propagator(jnp.asarray(origins), jnp.asarray(directions))
    cand_idx = np.asarray(out['sensor_indices'])
    cand_inside = np.asarray(out['inside_sensor'])
    cand_times = np.asarray(out['times']).squeeze(-1)
    weights = np.asarray(out['sensor_weights'])
    masked = np.where(cand_inside, cand_times, np.inf)
    slot = np.argmin(masked, axis=0)
    found = np.where(np.any(cand_inside, axis=0), cand_idx[slot, np.arange(len(origins))], -1)
    candidate_contains = np.any(cand_idx == truth_idx[None, :], axis=0)
    hit = truth_idx >= 0
    missed = hit & (found < 0)
    wrong = hit & (found >= 0) & (found != truth_idx)
    false = (~hit) & (found >= 0)
    positive_weight = np.sum(weights, axis=0) > 0.5
    positions = np.asarray(out['positions'])
    normals = np.asarray(out['normals'])
    found_distance = np.sum((positions-origins)*directions, axis=1)
    correct = hit & (found == truth_idx)
    center = centers[np.maximum(found, 0)]
    expected_normal = -(positions-center)/geometry.sensor_radius
    delta_normal = np.linalg.norm(expected_normal-normals, axis=1)
    row = dict(label=label, rays=len(origins), true_hits=int(np.sum(hit)),
               unbounded_hits=int(np.sum(unbounded >= 0)), found_hits=int(np.sum(found >= 0)),
               missed=int(np.sum(missed)), wrong_first=int(np.sum(wrong)), false_hits=int(np.sum(false)),
               missing_first_candidate=int(np.sum(hit & ~candidate_contains)),
               missed_with_candidate=int(np.sum(missed & candidate_contains)),
               positive_weight_no_geometry=int(np.sum(positive_weight & (found < 0))),
               positive_weight_no_oracle=int(np.sum(positive_weight & ~hit)),
               lost_fraction_true_hits=float(np.sum(missed)/max(1,np.sum(hit))),
               wrong_fraction_true_hits=float(np.sum(wrong)/max(1,np.sum(hit))),
               correct_stop_max_error_m=float(np.max(np.abs(found_distance[correct]-truth_dist[correct]))) if correct.any() else None,
               found_normal_max_error=float(np.max(delta_normal[found >= 0])) if np.any(found >= 0) else None)
    examples = []
    for i in np.flatnonzero(missed | wrong | false)[:5]:
        examples.append(dict(origin=origins[i].tolist(), direction=directions[i].tolist(),
                             expected_sensor=int(truth_idx[i]), expected_distance=float(truth_dist[i]),
                             found_sensor=int(found[i]), found_distance=float(found_distance[i]),
                             cylinder_exit=float(exits[i]), candidates=cand_idx[:, i].tolist(),
                             weights=weights[:, i].tolist()))
    print(json.dumps(row), flush=True)
    return row, examples


def main():
    assert os.environ['SLURM_JOB_PARTITION'] in ('milano', 'roma')
    geometry = DetectorGeometry.from_config('config/SK_WAND_geom_config.json',
                                            temperature=0.0, deposit_leg_bound=True)
    detector = geometry.detector
    centers = np.asarray(geometry.sensor_points, dtype=np.float64)
    base = dict(job_id=os.environ['SLURM_JOB_ID'], partition=os.environ['SLURM_JOB_PARTITION'],
                jax_version=jax.__version__, numpy_version=np.__version__,
                radius_m=detector.r, height_m=detector.H, sensor_radius_m=geometry.sensor_radius,
                n_sensors=geometry.num_sensors,
                grid=[detector._n_cap, detector._n_angular, detector._n_height],
                snap_max_m=float(np.max(np.linalg.norm(detector.raw_positions-centers, axis=1))),
                pmt_id_unique=len(np.unique(detector.pmt_id)),
                directions_norm_range=[float(np.linalg.norm(detector.pmt_directions,axis=1).min()),
                                       float(np.linalg.norm(detector.pmt_directions,axis=1).max())])
    rows, examples = [], {}
    target = 4105
    origin = centers[target]-np.array([0, 0, .4])
    origin[:2] *= (detector.r-.1)/detector.r
    direction = np.array([[0., 0., 1.]])
    row, sample = analyze(geometry, origin[None], direction, 'minimal_tangent_barrel')
    assert row['true_hits'] == 1 and row['wrong_first'] == 1 and row['missing_first_candidate'] == 1, sample
    rows.append(row)
    examples[row['label']] = sample
    rng = np.random.default_rng(1042026)
    n = 20000
    for radial in [0.,8.,detector.r-2.,15.,16.5,detector.r-.1]:
        origin = np.array([radial, 0., 0.])
        for z in [0., .2, .4, .6, .8]:
            origin[2] = z
            if np.min(np.linalg.norm(centers-origin, axis=1)) > geometry.sensor_radius:
                break
        assert np.min(np.linalg.norm(centers-origin, axis=1)) > geometry.sensor_radius
        directions = rng.normal(size=(n,3))
        row, sample = analyze(geometry,np.broadcast_to(origin,(n,3)).copy(),directions,f'isotropic_r{radial:.6f}')
        rows.append(row)
        examples[row['label']] = sample
    raw = np.load('config/sk_geometry.npz')
    original = raw['positions_mm']*.001
    geofile = np.loadtxt('config/geofile_SuperK.txt',skiprows=5)
    file_ids = geofile[:, 0].astype(int)
    file_positions = geofile[:, 3:6]*.01
    lookup = {pid:i for i,pid in enumerate(file_ids)}
    paired = np.array([lookup[int(pid)] for pid in raw['pmt_id']])
    base['npz_geofile_max_position_discrepancy_m'] = float(np.max(np.linalg.norm(original-file_positions[paired],axis=1)))
    base['npz_geofile_id_sets_equal'] = bool(set(file_ids)==set(raw['pmt_id']))
    base['surface_counts'] = {str(k):int(v) for k,v in zip(*np.unique(raw['surfaces'],return_counts=True))}
    base['metrics'] = rows
    base['examples'] = examples
    Path('audit/wc_data_review_20261004/geometry/results.json').write_text(json.dumps(base,indent=2)+'\n')


if __name__ == '__main__':
    main()
