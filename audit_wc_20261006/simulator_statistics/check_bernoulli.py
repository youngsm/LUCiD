"""Whole SK_WAND DATA path: normal-incidence spectral efficiency and count statistics."""
import json, os
import numpy as np
import jax
import jax.numpy as jnp
from lucid.simulation import setup_event_simulator
from lucid.sources.event_builder import _get_zero_track_params
from lucid.wavelength.medium import load_qe_curve, make_medium
from lucid.wavelength.optical_model import evaluate_optical_model

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
assert all(d.platform=='cpu' for d in jax.devices())
n=65536
sim=setup_event_simulator('config/SK_WAND_geom_config.json',0,K=12,is_data=True,temperature=0.,
    physics_config='config/SK_WAND_physics_config.json',default_detector_params=True,
    hit_mode='per_segment',deposit_leg_bound=True)
pts=np.asarray(sim.det_geom.sensor_points)
score=np.abs(pts[:,2])+np.abs(pts[:,1])+1000*(pts[:,0]<0)
target=int(np.argmin(score)); center=pts[target]; radial=center.copy();radial[2]=0;radial/=np.linalg.norm(radial)
origin=center-.5*radial
wavelengths=np.repeat(np.array([325,350,400,500],np.float32),n//4)
data=dict(photon_origins=jnp.broadcast_to(jnp.asarray(origin*100),(n,3)),
    photon_directions=jnp.broadcast_to(jnp.asarray(radial),(n,3)), photon_times=jnp.zeros(n),
    wavelengths=jnp.asarray(wavelengths),N=jnp.int32(n),apply_rotation=False,
    rotation_axis=jnp.array([1.,0.,0.]),rotation_angle=0.,apply_translation=False,
    translation_vector=jnp.zeros(3),photon_segment_index=jnp.arange(n,dtype=jnp.int32))
out=sim(_get_zero_track_params(),jax.random.PRNGKey(57192),data)
q,t,tr,w,wt,wtr,idx,seg=(np.asarray(v) for v in out)
valid=(w>0)&np.isfinite(wt)&np.isfinite(wtr)
gid=np.arange(len(w))%n
per_photon=np.bincount(gid[valid],minlength=n)
assert per_photon.max()<=1
assert np.allclose(w[valid],1.,atol=2e-6,rtol=0)
layout=(12,-1,n)
first=(w.reshape(layout)[0]>0)&np.isfinite(wt.reshape(layout)[0])
direct=first.sum(axis=0)
assert direct.max()<=1
prop=sim.det_geom.propagator(jnp.asarray(origin)[None],jnp.asarray(radial)[None])
D=float(np.linalg.norm(np.asarray(prop['positions'])[0]-origin))
qe_fn=load_qe_curve('config/pmt/SK_QE.json')
medium=make_medium('water',wavelength_grid=jnp.linspace(300.,648.22,200),medium_model_path='config/materials/water.json')
oa=evaluate_optical_model(sim.default_detector_params,jnp.asarray(wavelengths),medium,n,qe_fn=qe_fn)
pred=np.asarray(qe_fn(jnp.asarray(wavelengths)))*np.exp(-D*(1/np.asarray(oa.scatter_len)+1/np.asarray(oa.mie_len)+1/np.asarray(oa.abs_len)))
records=[]
for lam in np.unique(wavelengths):
    selection=wavelengths==lam; p=float(pred[selection][0]); count=int(direct[selection].sum()); N=int(selection.sum())
    z=(count-N*p)/np.sqrt(N*p*(1-p))
    binned=direct[selection].reshape(-1,256).sum(axis=1)
    records.append(dict(wavelength_nm=float(lam),photons=N,direct_detected=count,expected_probability=p,
        rate=count/N,z=z,block_variance_ratio=float(binned.var(ddof=1)/(256*p*(1-p)))))
    assert abs(z)<6
out=dict(partition=os.environ['SLURM_JOB_PARTITION'],job=os.environ.get('SLURM_JOB_ID'),source=origin.tolist(),target_sensor=target,
    surface_distance_m=D,maximum_valid_hits_per_photon=int(per_photon.max()),finite_detected_rows=int(valid.sum()),
    positive_nonfinite_ghost_rows=int(np.sum((w>0)&~valid)),records=records)
with open('audit_wc_20261006/simulator_statistics/bernoulli_results.json','w') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
