"""Pinned JAX 0.4.38 minimal regression; --fixed tests only an in-memory fix."""
import inspect, os, sys
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import jax
import jax.numpy as jnp
from lucid.geometry import generate_detector
import lucid.propagation.base as base
import lucid.propagation.shared as shared
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection

if "--fixed" in sys.argv:
    src = inspect.getsource(base.compute_sensor_intersections_base).replace(
        "discriminant = b ** 2 - 4 * a * c",
        "perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d\n"
        "    discriminant = 4 * a * (sensor_radius ** 2 - jnp.sum(perp ** 2, axis=1))")
    ns = dict(base.__dict__)
    exec(src, ns)
    shared.compute_sensor_intersections_base = ns["compute_sensor_intersections_base"]

det = generate_detector("config/SK_WAND_geom_config.json")
params = ScalarMixReflection(.05, .25, .55, .9)
prop = shared.create_propagator(det, jnp.array(det.all_points), det.S_radius,
                               temperature=0., deposit_leg_bound=True)

@jax.jit
def check(o, d, key):
    first = prop(o[None, :], d[None, :])
    has = jnp.any(first["inside_sensor"])
    sensor = first["sensor_indices"][jnp.argmax(first["sensor_weights"]), 0]
    distance = jnp.sqrt(jnp.sum((first["positions"][0] - o)**2) + 1e-12) - 1e-6
    s = photon_iteration_sample(o, d, 0., distance, first["normals"][0],
        1e20, 1e20, .9, params, 1e20, has, 400., key, .299792458/1.33,
        reflection_fn=scalar_mix_reflection)
    second = prop(s[0][None, :], s[1][None, :])
    same = jnp.any((second["sensor_indices"] == sensor) & second["inside_sensor"]) & has & (s[5] > 0)
    return same, sensor, s[0], s[1], second["positions"][0]

o = jnp.array([-11.555176734924316, -8.132349967956543, -15.8067045211792])
d = jnp.array([.32635268568992615, -.2600875198841095, .9087620377540588])
same, sensor, pos, direction, nextpos = check(o, d, jax.random.PRNGKey(1))
print(dict(jax=jax.__version__, fixed="--fixed" in sys.argv, sensor=int(sensor),
           false_self_hit=bool(same), next_leg_m=float(jnp.linalg.norm(nextpos-pos))), flush=True)
assert bool(same) == ("--fixed" not in sys.argv)
