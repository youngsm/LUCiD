"""Local, data-only all-PMT candidate-selection prototype; production untouched."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.propagation.base import compute_sensor_intersections_base
from lucid.overlap import create_overlap_prob

assert os.environ['SLURM_JOB_PARTITION'] in ('milano', 'roma')
g = DetectorGeometry.from_config('config/SK_WAND_geom_config.json', temperature=0.0, deposit_leg_bound=True)
results = json.loads(Path('audit/wc_data_review_20261004/geometry/results.json').read_text())
examples = [sample for batch in results['examples'].values() for sample in batch if sample['expected_sensor'] >= 0]
o = jnp.asarray([sample['origin'] for sample in examples])
d = jnp.asarray([sample['direction'] for sample in examples])
_, t_exit, _, _ = g.detector.intersect_ray(o, d)

@jax.jit
def complete_first_candidate(origins, directions, exits):
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    delta = g.sensor_points[None] - origins[:, None]
    projection = jnp.sum(delta * directions[:, None], axis=-1)
    perpendicular = delta - projection[:, :, None] * directions[:, None]
    discriminant = g.sensor_radius**2 - jnp.sum(perpendicular**2, axis=-1)
    entry = projection - jnp.sqrt(jnp.maximum(discriminant, 0))
    valid = (discriminant > 0) & (entry > 0) & (entry <= exits[:, None])
    bounded = jnp.where(valid, entry, jnp.inf)
    sensor = jnp.argmin(bounded, axis=1)
    first_t = jnp.min(bounded, axis=1)
    return jnp.where(jnp.isfinite(first_t), sensor, -1), first_t

fixed, distance = complete_first_candidate(o,d,t_exit)
expected = np.asarray([sample['expected_sensor'] for sample in examples])
assert np.array_equal(np.asarray(fixed), expected), (fixed,expected)
minimal = examples[0]
fixed_base = compute_sensor_intersections_base(fixed[:1],g.sensor_points,g.sensor_radius,o[:1],d[:1],g.detector.bounds_check,create_overlap_prob(None,g.sensor_radius),t_geometry=t_exit[:1])
assert int(fixed_base[2][0]) == 4105 and bool(fixed_base[4][0])
assert abs(float(fixed_base[1][0,0])-minimal['expected_distance']) < 1e-5

center = np.asarray(g.sensor_points[4105])
radial = center.copy(); radial[2]=0; radial /= np.linalg.norm(radial)
outside = jnp.asarray((center + 2*radial)[None])
inward = jnp.asarray((-radial)[None])
external = g.propagator(outside,inward)
external_row = dict(origin_m=np.asarray(outside)[0].tolist(),direction=np.asarray(inward)[0].tolist(),
                    origin_inside=bool(g.detector.bounds_check(outside)[0]),
                    candidates=np.asarray(external['sensor_indices'])[:,0].tolist(),
                    inside_sensor=np.asarray(external['inside_sensor'])[:,0].tolist(),
                    weights=np.asarray(external['sensor_weights'])[:,0].tolist(),
                    times_m=np.asarray(external['times'])[:,0,0].tolist(),
                    stop_m=np.asarray(external['positions'])[0].tolist())
assert not external_row['origin_inside']
assert sum(external_row['weights']) > .99
out = dict(job_id=os.environ['SLURM_JOB_ID'], tested_problem_rays=len(examples),
           complete_candidates_match_independent_oracle=True,
           corrected_minimal_sensor=int(fixed_base[2][0]),corrected_minimal_distance_m=float(fixed_base[1][0,0]),
           outside_origin_geometry=external_row)
Path('audit/wc_data_review_20261004/geometry/fix_and_boundary_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
