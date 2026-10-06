"""Minimal deterministic regression for a reflected SK PMT ray hitting itself.

Slurm only. Optionally pass --fixed to substitute the suggested stable
discriminant in memory (production source remains untouched).
"""
import inspect, os, sys
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import jax
import jax.numpy as jnp
import numpy as np
from lucid.geometry import generate_detector
import lucid.propagation.base as base
import lucid.propagation.shared as shared
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection

if "--fixed" in sys.argv:
    source=inspect.getsource(base.compute_sensor_intersections_base)
    source=source.replace("discriminant = b ** 2 - 4 * a * c", "perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d\n    discriminant = 4 * a * (sensor_radius ** 2 - jnp.sum(perp ** 2, axis=1))")
    namespace=dict(base.__dict__)
    exec(source,namespace)
    shared.compute_sensor_intersections_base=namespace["compute_sensor_intersections_base"]

det=generate_detector("config/SK_WAND_geom_config.json")
prop=shared.create_propagator(det,jnp.array(det.all_points),det.S_radius,temperature=0.,deposit_leg_bound=True)
# This actual SK ray was found by the independent geometry audit.
o=jnp.array([[6.3169121742248535,10.762299537658691,7.549770355224609]])
d=jnp.array([[-0.1375226527452469,-0.43994948267936707,-0.88742995262146]])
params=ScalarMixReflection(.05,.25,.55,.9)

@jax.jit
def reproduce(key):
    first=prop(o,d)
    distance=jnp.sqrt(jnp.sum((first["positions"][0]-o[0])**2)+1e-12)-1e-6
    step=photon_iteration_sample(o[0],d[0],0.,distance,first["normals"][0],
        1e20,1e20,.9,params,1e20,True,400.,key,.299792458/1.33,
        reflection_fn=scalar_mix_reflection)
    second=prop(step[0][None,:],step[1][None,:])
    same=(second["sensor_indices"]==10101)&second["inside_sensor"]
    return step[5],same.any(),step[0],step[1],second["positions"][0],second["sensor_weights"].sum()

for seed in range(32):
    continuing,selfhit,newpos,newdir,secondpos,secondcharge=reproduce(jax.random.PRNGKey(seed))
    if continuing and selfhit:
        print({"fixed":"--fixed" in sys.argv,"seed":seed,"false_self_hit":bool(selfhit),"newpos":np.asarray(newpos).tolist(),"direction":np.asarray(newdir).tolist(),"self_hit_distance_m":float(jnp.linalg.norm(secondpos-newpos)),"deposition":float(secondcharge)},flush=True)
        if "--fixed" in sys.argv:
            raise AssertionError("Stable-discriminant fix still self-intersects")
        break
else:
    if "--fixed" not in sys.argv:
        raise AssertionError("Expected current implementation to self-intersect")
    print("Stable-discriminant fix: no self-hit in 32 seeds",flush=True)
