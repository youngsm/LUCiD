"""Find and validate a scalar-JIT counterexample in the pinned JAX runtime."""
import inspect,json,os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
import jax
import jax.numpy as jnp
import numpy as np
from lucid.geometry import generate_detector
import lucid.propagation.base as base
import lucid.propagation.shared as shared
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection,scalar_mix_reflection

out=Path(__file__).resolve().parent
det=generate_detector("config/SK_WAND_geom_config.json")
params=ScalarMixReflection(.05,.25,.55,.9)

def build():
    prop=shared.create_propagator(det,jnp.array(det.all_points),det.S_radius,temperature=0.,deposit_leg_bound=True)
    @jax.jit
    def check(o,d,key):
        first=prop(o[None,:],d[None,:]);has=jnp.any(first["inside_sensor"])
        sensor=first["sensor_indices"][jnp.argmax(first["sensor_weights"]),0]
        distance=jnp.sqrt(jnp.sum((first["positions"][0]-o)**2)+1e-12)-1e-6
        s=photon_iteration_sample(o,d,0.,distance,first["normals"][0],1e20,1e20,.9,params,1e20,has,400.,key,.299792458/1.33,reflection_fn=scalar_mix_reflection)
        second=prop(s[0][None,:],s[1][None,:])
        same=jnp.any((second["sensor_indices"]==sensor)&second["inside_sensor"])&has&(s[5]>0)
        return same,sensor,s[0],s[1],second["positions"][0]
    return check

check=build();data=np.load(out/"reflection_sample.npz")
indices=np.argsort(data["depth"])[::-1][:500]
found=None
for index in indices:
    o=jnp.asarray(data["origins"][index]);d=jnp.asarray(data["directions"][index])
    for seed in range(16):
        same,sensor,pos,direction,nextpos=check(o,d,jax.random.PRNGKey(seed))
        if same:
            found={"job":os.environ["SLURM_JOB_ID"],"jax":jax.__version__,"index":int(index),"seed":seed,"sensor":int(sensor),"origin":np.asarray(o).tolist(),"direction":np.asarray(d).tolist(),"new_origin":np.asarray(pos).tolist(),"new_direction":np.asarray(direction).tolist(),"self_hit_distance_m":float(jnp.linalg.norm(nextpos-pos))}
            break
    if found:break
assert found,"No scalar-JIT counterexample found in search"
print("FOUND",json.dumps(found),flush=True)
src=inspect.getsource(base.compute_sensor_intersections_base).replace("discriminant = b ** 2 - 4 * a * c","perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d\n    discriminant = 4 * a * (sensor_radius ** 2 - jnp.sum(perp ** 2, axis=1))")
ns=dict(base.__dict__);exec(src,ns);shared.compute_sensor_intersections_base=ns["compute_sensor_intersections_base"]
fixed=build();same,*_=fixed(jnp.array(found["origin"]),jnp.array(found["direction"]),jax.random.PRNGKey(found["seed"]))
found["fixed_selfhit"]=bool(same);assert not same
(out/f"minimal_example_jax{jax.__version__}.json").write_text(json.dumps(found,indent=2))
print("FIXED: same ray/key has no selfhit",flush=True)
