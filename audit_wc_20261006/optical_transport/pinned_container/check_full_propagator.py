"""Check candidate float32 reflection issue through the actual SK_WAND map."""
import json, os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import jax
import jax.numpy as jnp
import numpy as np
from lucid.geometry import generate_detector
from lucid.propagation.shared import create_propagator
from lucid.simulation.optics import compute_reflection_direction, sample_cosine_hemisphere
from lucid.utils import normalize

out=Path(__file__).resolve().parent
det=generate_detector("config/SK_WAND_geom_config.json")
print("Building production geometry/map",flush=True)
prop=create_propagator(det,jnp.array(det.all_points),det.S_radius,temperature=0.,deposit_leg_bound=True)
print("Map built",flush=True)
samples=np.load(out/"reflection_sample.npz")
orig=jnp.array(samples["origins"]); dirs=jnp.array(samples["directions"])
first=prop(orig,dirs)
hit=first["positions"];normals=first["normals"]
initial=jnp.any(first["inside_sensor"],axis=0)
dist=jnp.sqrt(jnp.sum((hit-orig)**2,axis=1)+1e-12)-1e-6
newpos=orig+dist[:,None]*normalize(dirs)-1e-4*normalize(normals)
# Identify the physically first PMT, independently of tiny surviving weights.
first_slot=jnp.argmax(first["sensor_weights"],axis=0)
first_sensor=jnp.take_along_axis(first["sensor_indices"],first_slot[None,:],axis=0)[0]
centers=jnp.array(det.all_points)[first_sensor]
depth=det.S_radius-jnp.linalg.norm(newpos-centers,axis=1)
report={"job":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],"n":len(orig),"initial_hits":int(initial.sum()),"reflected_origins_inside_pmt":int((initial&(depth>0)).sum())}
print(json.dumps(report),flush=True)
for name,reflected in [("specular",compute_reflection_direction(dirs,normals)),("diffuse",jax.vmap(sample_cosine_hemisphere)(-normals,jax.random.split(jax.random.PRNGKey(714),len(orig))))]:
    second=prop(newpos,reflected)
    same_sensor=(second["sensor_indices"]==first_sensor[None,:])
    same_flag=jnp.any(same_sensor&second["inside_sensor"],axis=0)&initial
    same_map=jnp.any(same_sensor,axis=0)&initial
    second_hit=jnp.any(second["inside_sensor"],axis=0)&initial
    escaped=(~det.bounds_check(newpos))&initial
    report[name]={"old_pmt_in_map":int(same_map.sum()),"old_pmt_self_hits":int(same_flag.sum()),"second_hit":int(second_hit.sum()),"self_hit_fraction_per_pmt_reflection":float(same_flag.sum()/initial.sum()),"escaped_after_reflection":int(escaped.sum())}
    if np.asarray(same_flag).any():
        i=int(np.flatnonzero(np.asarray(same_flag))[0])
        report[name]["example"]={"sensor_id":int(first_sensor[i]),"origin":np.asarray(orig[i]).tolist(),"direction":np.asarray(dirs[i]).tolist(),"new_origin":np.asarray(newpos[i]).tolist(),"new_direction":np.asarray(reflected[i]).tolist(),"second_position":np.asarray(second["positions"][i]).tolist(),"depth_m":float(depth[i])}
    print(name,report[name],flush=True)
(out/"full_propagator_results.json").write_text(json.dumps(report,indent=2))
