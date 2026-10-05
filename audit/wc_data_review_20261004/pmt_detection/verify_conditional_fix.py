"""Run only in Slurm milano/roma: verify the minimal scalar conditional-QE fix."""
import json
import os
import jax
import jax.numpy as jnp
from lucid.detector_params import load_physics_config
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection
from lucid.simulation.sensor_response import make_hits_per_photon
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
dp, _, _ = load_physics_config('config/SK_WAND_physics_config.json', num_sensors=1)
refl = ScalarMixReflection(dp.reflection.wall_reflection_rate, dp.reflection.sensor_reflection_rate,
                          dp.reflection.wall_fspec, dp.reflection.sensor_fspec)
n = 100_000
def step(key):
    return photon_iteration_sample(jnp.zeros(3), jnp.array([1., 0., 0.]), 0., 1.,
        jnp.array([1., 0., 0.]), 1e20, 1e20, .95, refl, 1e20, True, 400., key, .225,
        reflection_fn=scalar_mix_reflection)[3]
weights = jax.jit(jax.vmap(step))(jax.random.split(jax.random.PRNGKey(76), n))
q = make_hits_per_photon(weights, jnp.zeros(n, jnp.int32), jnp.ones(n), 1,
    qe=dp.response.qe/(1-dp.reflection.sensor_reflection_rate),
    qe_corrections=jnp.ones(1), rng_key=jax.random.PRNGKey(70))[0].sum()
expected = float(dp.response.qe)
observed = float(q)/n
assert abs(observed-expected) < .003
print(json.dumps({'job_id':os.environ['SLURM_JOB_ID'], 'partition':os.environ['SLURM_JOB_PARTITION'],
    'conditional_qe':float(dp.response.qe/(1-dp.reflection.sensor_reflection_rate)),
    'expected_incident_qe':expected, 'observed_incident_probability':observed}), flush=True)
