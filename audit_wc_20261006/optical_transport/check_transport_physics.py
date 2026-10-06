"""Independent DATA photon-step checks; run only inside milano/roma Slurm jobs.

Checks analytic transport probabilities, phase-function moments and rotation,
plus the SK calibration paper's 400 nm water coefficients. No production edits.
"""
import json
import os
import socket

assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}, (
    "Run this CPU numerical audit on a milano/roma allocation only")
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.optics import (
    compute_scatter_direction, sample_cosine_hemisphere,
    solve_rayleigh_inverse_cdf,
)
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection
from lucid.wavelength.medium import make_medium
from lucid.wavelength.scattering import compute_mie_scatter_direction, hg_sample_cos_theta


def summarize(name, observed, expected, tolerance):
    passed = abs(observed - expected) < tolerance
    item = dict(name=name, observed=float(observed), expected=float(expected),
                tolerance=float(tolerance), passed=bool(passed))
    results.append(item)
    print(json.dumps(item), flush=True)
    assert passed, item


N = 262144
results = []
metadata = dict(host=socket.gethostname(), partition=os.environ["SLURM_JOB_PARTITION"],
                job_id=os.environ.get("SLURM_JOB_ID"), jax=jax.__version__,
                backend=jax.default_backend(), samples=N)
print(json.dumps(metadata), flush=True)
keys = jax.random.split(jax.random.PRNGKey(72041), N)
direction = jnp.asarray([0.3, 0.4, np.sqrt(0.75)], dtype=jnp.float32)
normal = jnp.asarray([0., 0., 1.])
D, LR, LM, LA, R, G = 20., 30., 50., 40., .25, .95
params = ScalarMixReflection(.05, R, .55, .9)


@jax.jit
def step(keys):
    return jax.vmap(lambda key: photon_iteration_sample(
        jnp.zeros(3), direction, 0., D, normal, LR, LM, G, params, LA,
        True, 400., key, .299792 / 1.33,
        reflection_fn=scalar_mix_reflection))(keys)


pos, dirs, times, deposited, survived, continuing, _, indirect = map(np.asarray, step(keys))
# Scatter points remain on the initial ray. Surface points receive the 1e-4 m
# normal nudge, with 5e-5 m transverse component for this oblique direction.
# A distance-to-boundary threshold would mislabel legitimate near-wall scatters.
scattered = np.linalg.norm(np.cross(pos, np.asarray(direction)), axis=1) < 1e-5
reflected = (deposited == 0) & ~scattered
mu = 1 / LR + 1 / LM
alpha = 1 / LA
ps = np.exp(-mu * D)
summarize("surface_probability", np.mean(~scattered), ps, .004)
summarize("deposit_and_survive", np.mean(deposited * survived),
          (1 - R) * np.exp(-(mu + alpha) * D), .004)
summarize("reflection_and_survive", np.mean(reflected * survived),
          R * np.exp(-(mu + alpha) * D), .004)
summarize("scatter_and_survive", np.mean(scattered * survived),
          mu / (mu + alpha) * (1 - np.exp(-(mu + alpha) * D)), .004)
summarize("continuation_probability", np.mean(continuing),
          R * np.exp(-(mu + alpha) * D)
          + mu / (mu + alpha) * (1 - np.exp(-(mu + alpha) * D)), .004)
summarize("conditional_scatter_distance", np.mean(np.linalg.norm(pos[scattered], axis=1)),
          1 / mu - D / np.expm1(mu * D), .06)
p_mie = (1 / LM) / mu
cosine = dirs @ np.asarray(direction)
summarize("mixed_scatter_mean_cos", np.mean(cosine[scattered]), p_mie * G, .007)
summarize("mixed_scatter_second_moment", np.mean(cosine[scattered] ** 2),
          (1 - p_mie) * .4 + p_mie * (1 + 2 * G ** 2) / 3, .007)
summarize("surviving_scatter_mean_cos", np.mean(cosine[scattered & (survived > 0)]),
          p_mie * G, .008)
summarize("absorbed_scatter_mean_cos", np.mean(cosine[scattered & (survived == 0)]),
          p_mie * G, .012)
summarize("flight_time_distance", np.max(np.abs(times * (.299792 / 1.33)
          - np.where(scattered, np.linalg.norm(pos, axis=1), D))), 0., 2e-5)

u = jnp.linspace(0., 1., 100001)
mu_ray = np.asarray(solve_rayleigh_inverse_cdf(u))
cdf = .5 + .375 * mu_ray + .125 * mu_ray ** 3
summarize("rayleigh_inverse_cdf_max_error", np.max(np.abs(cdf - np.asarray(u))), 0., 1e-6)

for name, axis in [("z", jnp.asarray([0., 0., 1.])), ("oblique", direction)]:
    ray = np.asarray(jax.jit(jax.vmap(lambda k: compute_scatter_direction(axis, k)))(keys))
    mie = np.asarray(jax.jit(jax.vmap(lambda k: compute_mie_scatter_direction(axis, k, G)))(keys))
    hemi = np.asarray(jax.jit(jax.vmap(lambda k: sample_cosine_hemisphere(axis, k)))(keys))
    summarize(f"rayleigh_{name}_mean_cos", np.mean(ray @ axis), 0., .004)
    summarize(f"rayleigh_{name}_second_moment", np.mean((ray @ axis) ** 2), .4, .004)
    summarize(f"mie_{name}_mean_cos", np.mean(mie @ axis), G, .004)
    summarize(f"lambertian_{name}_mean_cos", np.mean(hemi @ axis), 2 / 3, .004)
    summarize(f"lambertian_{name}_second_moment", np.mean((hemi @ axis) ** 2), .5, .004)
    summarize(f"lambertian_{name}_wrong_hemisphere", np.mean(hemi @ axis < 0), 0., 1e-6)

medium = make_medium(wavelength_grid=jnp.asarray([400.]))
# Evaluate the published equations with Table 3's rounded constants; the prose
# reports 402 m while the rounded table implies about 400.4 m. That sub-percent
# rounding mismatch is not a LUCiD physics defect.
for name, observed, expected, tolerance in [
        ("SK_400nm_absorption_length_m", 1 / medium.absorption_coeff[0],
         1 / (.624 * 2.96e7 / 400.**4 + .624 * .0324 * (.8)**10.9), .001),
        ("SK_400nm_symmetric_length_m", 1 / medium.scatter_coeff[0],
         1 / (8.51e7 / 400.**4 * (1 + 1.14e5 / 400.**2)), .001),
        ("SK_400nm_asymmetric_length_m", 1 / medium.mie_scatter_coeff[0],
         1 / (1e-4 * (1 + 4.62e6 / 400.**4 * (400. - 392.)**2)), .01)]:
    summarize(name, float(observed), expected, tolerance)

# Document a physics-model difference, not a numerical regression: the SK reference
# uses p(mu)=2mu, 0<=mu<=1, for asymmetric scattering (mean 2/3); HG(g=.95) differs.
hg = np.asarray(hg_sample_cos_theta((jnp.arange(N) + .5) / N, .95))
model_difference = dict(SK_asymmetric_mean_cos=2 / 3, LUCiD_HG_mean_cos=float(hg.mean()),
                        SK_backward_probability=0., LUCiD_HG_backward_probability=float(np.mean(hg < 0)),
                        source="https://arxiv.org/pdf/1307.0162 p.31")
print(json.dumps({"model_difference": model_difference}), flush=True)
output_directory = os.environ.get("AUDIT_OUTPUT_DIR", os.path.dirname(__file__))
with open(os.path.join(output_directory, "transport_checks.json"), "w") as out:
    json.dump(dict(metadata=metadata, checks=results, model_difference=model_difference), out, indent=2)
