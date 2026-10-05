"""Run only under milano/roma Slurm allocation using uv.

Validate stochastic rotational invariants; illustrate the unpolarized approximation.
The polarization comparison is a fidelity limit, not a claimed implementation defect.
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.optics import compute_scatter_direction, create_local_frame
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
n = 250_000
keys = jax.random.split(jax.random.PRNGKey(62409), n)
axes = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 2, 3], [-4, 1, -2]], dtype=np.float32)
axes /= np.linalg.norm(axes, axis=1, keepdims=True)
sample = jax.jit(jax.vmap(compute_scatter_direction, in_axes=(None, 0)))
results = []
for axis in axes:
    directions = np.asarray(sample(jnp.array(axis), keys))
    frame = np.asarray(create_local_frame(jnp.array(axis)))
    local = directions @ frame.T
    mean = local.mean(axis=0)
    second = (local * local).mean(axis=0)
    norm_error = float(np.max(np.abs(np.linalg.norm(directions, axis=1) - 1)))
    assert np.max(np.abs(mean)) < .005
    assert np.max(np.abs(second - [0.3, 0.3, 0.4])) < .005
    assert norm_error < 2e-6
    results.append(dict(axis=axis.tolist(), mean=mean.tolist(), second=second.tolist(), max_norm_error=norm_error))

# Independent electric-dipole expectation for incident d=(0,0,1), polarization=(1,0,0).
# The normalized density is 3/(8pi) * (1 - d_out.x**2).
# Spherical isotropic moments E[x²]=1/3, E[x⁴]=1/5, E[x²y²]=1/15
# give E[x²]=.2, E[y²]=.4, E[z²]=.4 after polarization weighting.
d = np.asarray(sample(jnp.array([0., 0., 1.]), keys))
unpolarized_second = (d * d).mean(axis=0)
physical_polarized_second = [0.2, 0.4, 0.4]

# Reflecting off either a wall outward normal or the inverted sensor normal
# must point back into the water hemisphere for every specular/diffuse branch.
normal = jnp.array([0., 0., 1.])
incident = jnp.array([0.6, 0., 0.8])
r = ScalarMixReflection(jnp.array(.05), jnp.array(.25), jnp.array(.55), jnp.array(.9))
refl_sample = jax.jit(jax.vmap(scalar_mix_reflection, in_axes=(None, None, None, None, None, 0)))
reflection_results = []
for hit_sensor in [False, True]:
    rate, reflected, _ = refl_sample(incident, normal, hit_sensor, r, jnp.array(400.), keys)
    reflected = np.asarray(reflected)
    inward_fraction = float(np.mean(reflected @ np.asarray(normal) <= 0))
    assert inward_fraction == 1.0
    reflection_results.append(dict(hit_sensor=hit_sensor, mean_rate=float(np.asarray(rate).mean()), inward_fraction=inward_fraction))

out = dict(job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename, n=n,
    rayleigh_rotational_invariants=results,
    polarization_fidelity_limit=dict(lucid_unpolarized_second=unpolarized_second.tolist(), physical_linearly_polarized_second=physical_polarized_second),
    reflection_water_hemisphere=reflection_results)
print(json.dumps(out, indent=2))
Path(__file__).with_name('results.json').write_text(json.dumps(out, indent=2) + '\n')
