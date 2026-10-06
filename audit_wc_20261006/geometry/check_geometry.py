"""Independent SK_WAND geometry checks; run only inside milano/roma Slurm."""
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID"), "Run inside a CPU Slurm allocation."
assert os.environ.get("SLURM_JOB_PARTITION", "").split(",")[0] in ("milano", "roma"), os.environ.get("SLURM_JOB_PARTITION")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from lucid.geometry import generate_detector
from lucid.overlap import create_overlap_prob
from lucid.propagation.base import compute_sensor_intersections_base
from lucid.propagation.cylinder import batch_intersect_cylinder_with_grid
from lucid.simulation.optics import compute_reflection_direction

out = Path(__file__).resolve().parent
det = generate_detector("config/SK_WAND_geom_config.json")
pts = np.asarray(det.all_points)
rad = det.S_radius
rng = np.random.default_rng(61523)
report = dict(job=os.environ["SLURM_JOB_ID"], partition=os.environ.get("SLURM_JOB_PARTITION"), devices=str(jax.devices()), radius=det.r, height=det.H,
              sensor_radius=rad, count=len(pts), surfaces={str(k): int(v) for k, v in zip(*np.unique(list(det.ID_to_case.values()), return_counts=True))},
              min_sensor_separation=float(cKDTree(pts).query(pts, k=2)[0][:, 1].min()),
              max_snap_displacement=float(np.linalg.norm(pts-det.raw_positions, axis=1).max()))

# Independent double-precision cylinder roots, on random interior rays.
n = 200000
phi = rng.uniform(0, 2*np.pi, n)
rho = np.sqrt(rng.uniform(0, 1, n)) * (det.r - 1e-3)
orig = np.column_stack([rho*np.cos(phi), rho*np.sin(phi), rng.uniform(-det.H/2+1e-3, det.H/2-1e-3, n)]).astype(np.float32)
dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:,None]; dirs = dirs.astype(np.float32)
o64=orig.astype(float); d64=dirs.astype(float)
a=np.sum(d64[:,:2]**2,axis=1); b=np.sum(o64[:,:2]*d64[:,:2],axis=1); c=np.sum(o64[:,:2]**2,axis=1)-det.r**2
t_side=(-b+np.sqrt(b*b-a*c))/a
t_cap=np.where(d64[:,2]>0,det.H/2-o64[:,2],-det.H/2-o64[:,2])/d64[:,2]
t_ref=np.minimum(t_side,t_cap)
result=batch_intersect_cylinder_with_grid(jnp.array(orig),jnp.array(dirs),det.r,det.H,64,180,80)
t_actual=np.asarray(result[1]); cap=np.asarray(result[3]); wall=np.asarray(result[2])
report["cylinder"]={"n":n,"misses":int((~np.asarray(result[0])).sum()),"max_distance_error_m":float(abs(t_actual-t_ref).max()),"distance_error_p999_m":float(np.quantile(abs(t_actual-t_ref),.999)),"wrong_wall_part":int((wall!=(t_side<t_cap)).sum())}

# Force candidate PMTs into the local geometry kernel, so any observed
# failure is independent of the accepted endpoint-search approximation.
n = 200000
idx = rng.integers(0,len(pts),n)
phi=rng.uniform(0,2*np.pi,n); rho=np.sqrt(rng.uniform(0,1,n))*(det.r-2.)
orig=np.column_stack([rho*np.cos(phi),rho*np.sin(phi),rng.uniform(-det.H/2+2,det.H/2-2,n)]).astype(np.float32)
centers=pts[idx]
axis=centers-orig; axis/=np.linalg.norm(axis,axis=1)[:,None]
v=rng.normal(size=(n,3)); v-=np.sum(v*axis,axis=1)[:,None]*axis;v/=np.linalg.norm(v,axis=1)[:,None]
impact=np.sqrt(rng.uniform(0,1,n))*rad
target=centers+v*impact[:,None]
dirs=target-orig;dirs/=np.linalg.norm(dirs,axis=1)[:,None];dirs=dirs.astype(np.float32)
cyl=batch_intersect_cylinder_with_grid(jnp.array(orig),jnp.array(dirs),det.r,det.H,64,180,80)
overlap=create_overlap_prob(None,rad)
compute=jax.jit(lambda ids,origins,directions,t:compute_sensor_intersections_base(ids,jnp.array(pts),rad,origins,directions,det.bounds_check,overlap,t_geometry=t))
weights,times,_,normals,inside,hit=compute(jnp.array(idx),jnp.array(orig),jnp.array(dirs),cyl[1])
# Exactly the simulator/step geometry arithmetic, including norm-minus-eps.
dist=jnp.sqrt(jnp.sum((hit-jnp.array(orig))**2,axis=1)+1e-12)-1e-6
normalized=jnp.array(dirs)/(jnp.linalg.norm(jnp.array(dirs),axis=1,keepdims=True)+1e-10)
newpos=jnp.array(orig)+dist[:,None]*normalized-1e-4*normals/(jnp.linalg.norm(normals,axis=1,keepdims=True)+1e-10)
reflected=compute_reflection_direction(jnp.array(dirs),normals)
cyl2=batch_intersect_cylinder_with_grid(newpos,reflected,det.r,det.H,64,180,80)
w2,t2,_,n2,in2,h2=compute(jnp.array(idx),newpos,reflected,cyl2[1])
initial=np.asarray(inside)
depth=rad-np.linalg.norm(np.asarray(newpos).astype(float)-centers,axis=1)
selfhit=initial & np.asarray(in2)
report["sphere_reflection"]={"sampled_rays":n,"initial_valid_hits":int(initial.sum()),"origins_inside_pmt_after_nudge":int((initial&(depth>0)).sum()),"self_intersections":int(selfhit.sum()),"self_intersection_fraction":float(selfhit.sum()/initial.sum()),"max_penetration_m":float(depth[initial].max()),"depth_quantiles_m":np.quantile(depth[initial],[0,.1,.5,.9,.99,1]).tolist(),"self_hit_zero_charge":int((selfhit&(np.asarray(w2)==0)).sum())}
if selfhit.any():
    i=int(np.flatnonzero(selfhit)[np.argmax(depth[selfhit])])
    report["sphere_reflection"]["example"]={"sensor_id":int(idx[i]),"sensor_center":centers[i].tolist(),"origin":orig[i].tolist(),"direction":dirs[i].tolist(),"first_hit":np.asarray(hit)[i].tolist(),"first_normal":np.asarray(normals)[i].tolist(),"new_origin":np.asarray(newpos)[i].tolist(),"reflected_dir":np.asarray(reflected)[i].tolist(),"self_hit_distance_m":float(np.asarray(t2)[i,0]),"inside_depth_m":float(depth[i])}
np.savez(out / "reflection_sample.npz",idx=idx,origins=orig,directions=dirs,first_valid=initial,selfhit=selfhit,depth=depth)
(out / "results.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
