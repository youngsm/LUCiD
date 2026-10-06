"""Actual sampled reflection update fused with both full SK propagation calls."""
import inspect,json,os,sys
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
sample=np.load(out/"reflection_sample.npz")
orig=jnp.asarray(sample["origins"]);dirs=jnp.asarray(sample["directions"])
params=ScalarMixReflection(.05,.25,.55,.9)
report={"job":os.environ["SLURM_JOB_ID"],"jax":jax.__version__,"n":len(orig)}
for mode in ("baseline","fixed"):
    if mode=="fixed":
        src=inspect.getsource(base.compute_sensor_intersections_base).replace("discriminant = b ** 2 - 4 * a * c","perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d\n    discriminant = 4 * a * (sensor_radius ** 2 - jnp.sum(perp ** 2, axis=1))")
        ns=dict(base.__dict__);exec(src,ns);shared.compute_sensor_intersections_base=ns["compute_sensor_intersections_base"]
    prop=shared.create_propagator(det,jnp.array(det.all_points),det.S_radius,temperature=0.,deposit_leg_bound=True)
    @jax.jit
    def check(o,d):
        first=prop(o,d);has_first=jnp.any(first["inside_sensor"],axis=0)
        slot=jnp.argmax(first["sensor_weights"],axis=0)
        sensor=jnp.take_along_axis(first["sensor_indices"],slot[None,:],axis=0)[0]
        dist=jnp.sqrt(jnp.sum((first["positions"]-o)**2,axis=1)+1e-12)-1e-6
        keys=jax.random.split(jax.random.PRNGKey(1232),len(o))
        def step(p,d,dist,normal,hit,key):
            return photon_iteration_sample(p,d,0.,dist,normal,1e20,1e20,.9,params,1e20,hit,400.,key,.299792458/1.33,reflection_fn=scalar_mix_reflection)
        s=jax.vmap(step)(o,d,dist,first["normals"],has_first,keys)
        second=prop(s[0],s[1]);reflection=has_first&(s[5]>0)
        same=jnp.any((second["sensor_indices"]==sensor[None,:])&second["inside_sensor"],axis=0)&reflection
        return has_first,reflection,same,sensor,s[0],s[1],keys
    first,reflection,same,sensor,newpos,newdir,keys=jax.tree.map(np.asarray,check(orig,dirs))
    report[mode]={"initial_pmt_hits":int(first.sum()),"sampled_pmt_reflections":int(reflection.sum()),"false_same_pmt_hits":int(same.sum()),"fraction":float(same.sum()/reflection.sum())}
    if same.any():
        i=int(np.flatnonzero(same)[0]);report[mode]["example"]={"idx":i,"sensor":int(sensor[i]),"origin":np.asarray(orig[i]).tolist(),"direction":np.asarray(dirs[i]).tolist(),"key":keys[i].tolist(),"new_origin":newpos[i].tolist(),"new_direction":newdir[i].tolist()}
    print(mode,report[mode],flush=True)
report_file=out/f"fused_results_jax{jax.__version__}.json"
report_file.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
