"""Paired K=12 DATA transport comparison with only sphere discriminant changed.

The source is a synthetic ensemble of isotropic photons uniformly distributed
over the SK fiducial cylinder, with a 1/lambda**2 spectrum over 300--650 nm.
This measures an ensemble-specific optical response, not a Geant4 event sample.
"""
import argparse,inspect,json,os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
import jax
import jax.numpy as jnp
import numpy as np
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
import lucid.propagation.base as base
import lucid.propagation.shared as shared

ap=argparse.ArgumentParser();ap.add_argument("--seeds",type=int,default=3);args=ap.parse_args()
n=200000
rng=np.random.default_rng(41722)
phi=rng.uniform(0,2*np.pi,n);r=np.sqrt(rng.uniform(0,1,n))*14.96228
orig=np.column_stack([r*np.cos(phi),r*np.sin(phi),rng.uniform(-16.1892,16.1892,n)])
dirs=rng.normal(size=(n,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
wl=1/(1/300-rng.uniform(0,1,n)*(1/300-1/650))
photons=dict(photon_origins=jnp.array(orig*100,dtype=jnp.float32),photon_directions=jnp.array(dirs,dtype=jnp.float32),photon_times=jnp.zeros(n),
    wavelengths=jnp.array(wl,dtype=jnp.float32),N=n,rotation_axis=jnp.array([0.,0.,1.]),rotation_angle=0.,apply_rotation=False,
    apply_translation=False,translation_vector=jnp.zeros(3),photon_segment_index=jnp.zeros(n,dtype=jnp.int32))
particle=ParticleParams.from_cartesian(1.,jnp.zeros(3),jnp.array([0.,0.,1.]))
results={"job":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],"n_per_seed":n,"seeds":list(range(701,701+args.seeds)),"source":"isotropic uniform fiducial volume, 1/lambda^2 300-650nm","runs":{}}
saved={}
for mode in ("baseline","fixed"):
    if mode=="fixed":
        src=inspect.getsource(base.compute_sensor_intersections_base)
        src=src.replace("discriminant = b ** 2 - 4 * a * c","perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d\n    discriminant = 4 * a * (sensor_radius ** 2 - jnp.sum(perp ** 2, axis=1))")
        ns=dict(base.__dict__);exec(src,ns);shared.compute_sensor_intersections_base=ns["compute_sensor_intersections_base"]
    print("Building",mode,flush=True)
    sim=setup_event_simulator("config/SK_WAND_geom_config.json",0,K=12,is_data=True,temperature=0.,
        physics_config="config/SK_WAND_physics_config.json",default_detector_params=True,hit_mode="per_segment",deposit_leg_bound=True,charge_resolution=None)
    @jax.jit
    def summarized(key):
        out=sim(particle,key,photons)
        w=out[3].reshape(12,4,n);t=out[4].reshape(12,4,n)
        # Preserve per-input-photon pairing; scalar QE acceptance is already applied.
        return jnp.stack([jnp.sum(w,axis=(0,1)),jnp.sum(jnp.where(t>150,w,0),axis=(0,1)),jnp.sum(jnp.where(t>300,w,0),axis=(0,1))])
    runs=[]
    for seed in results["seeds"]:
        q=np.asarray(summarized(jax.random.PRNGKey(seed)))
        runs.append({"seed":seed,"total_pe":float(q[0].sum()),"pe_after150ns":float(q[1].sum()),"pe_after300ns":float(q[2].sum())})
        print(mode,runs[-1],flush=True)
        if mode=="baseline":saved[seed]=q
        else:
            delta=q-saved[seed]
            runs[-1]["paired_delta"]=[float(v) for v in delta.sum(axis=1)]
            runs[-1]["paired_sigma"]=[float(v) for v in np.sqrt(np.sum(delta**2,axis=1))]
    results["runs"][mode]=runs
out=Path(__file__).resolve().parent/f"charge_bias_results_{args.seeds}seeds.json"
out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)
