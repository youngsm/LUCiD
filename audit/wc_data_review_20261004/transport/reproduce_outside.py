"""Actual SK_WAND DATA front/back incidence audit; run in Slurm only."""
import os
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from lucid.simulation.simulator import setup_event_simulator
from lucid.detector_params import ParticleParams

n = 100_000
sim = setup_event_simulator('config/SK_WAND_geom_config.json', 0, K=12,
    is_data=True, temperature=0.0, physics_config='config/SK_WAND_physics_config.json',
    default_detector_params=True, hit_mode='per_segment', deposit_leg_bound=True)
geom = sim.det_geom
p = np.asarray(geom.sensor_points)
ids = np.flatnonzero((np.linalg.norm(p[:, :2], axis=1) > geom.detector.r - .01)
                    & (np.abs(p[:, 2]) < 1.0))
sensor_id = int(ids[0])
center = p[sensor_id]
normal = np.array([center[0], center[1], 0.0])
normal /= np.linalg.norm(normal)
particle = ParticleParams.from_cartesian(1000.0, [0.,0.,0.], [1.,0.,0.])
rows = []
for name, origin, direction in (
    ('inside_front', center - .5 * normal, normal),
    ('outside_back', center + .5 * normal, -normal)):
    data = {'photon_origins': jnp.tile(jnp.asarray(origin * 100.0), (n, 1)),
            'photon_directions': jnp.tile(jnp.asarray(direction), (n, 1)),
            'photon_times': jnp.zeros(n), 'wavelengths': jnp.full(n, 400.),
            'N': jnp.asarray(n), 'apply_rotation': jnp.asarray(False),
            'rotation_axis': jnp.array([0.,0.,1.]), 'rotation_angle': jnp.asarray(0.),
            'photon_segment_index': jnp.zeros(n, dtype=jnp.int32)}
    out = sim(particle, jax.random.PRNGKey(12341), data)
    q = np.asarray(out[0])
    result = {'source': name, 'N': n, 'origin_m': origin.tolist(),
        'direction': direction.tolist(), 'origin_inside': bool(geom.detector.bounds_check(jnp.asarray(origin)[None])[0]),
        'sensor_id': sensor_id, 'sensor_center_m': center.tolist(),
        'total_pe': float(q.sum()), 'target_sensor_pe': float(q[sensor_id]),
        'fraction_detected': float(q.sum() / n),
        'nonzero_sensor_count': int(np.count_nonzero(q)),
        'expected_if_outside_black_liner_is_opaque': 0.0 if name == 'outside_back' else None}
    prop = geom.propagator(jnp.asarray(origin)[None], jnp.asarray(direction)[None])
    result['first_surface_distance_m'] = float(np.linalg.norm(np.asarray(prop['positions'])[0] - origin))
    result['inside_sensor_candidates'] = np.asarray(prop['inside_sensor'])[:, 0].tolist()
    result['candidate_weights'] = np.asarray(prop['sensor_weights'])[:, 0].tolist()
    result['candidate_sensor_ids'] = np.asarray(prop['sensor_indices'])[:, 0].tolist()
    rows.append(result)
    print(json.dumps(result), flush=True)
assert rows[1]['total_pe'] > 10_000
Path('audit/wc_data_review_20261004/transport/outside_results.json').write_text(
    json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'partition':os.environ['SLURM_JOB_PARTITION'],'results':rows}, indent=2) + '\n')
