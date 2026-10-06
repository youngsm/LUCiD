"""Independent SK_WAND DATA transport/orchestration checks; execute on milano/roma only."""
import json, os, time
import numpy as np
import jax
import jax.numpy as jnp
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.sources.event_builder import _trace_event_bucketed, _split_into_chunks
from lucid.propagation.shared import first_hit_survival

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
assert all(x.platform == 'cpu' for x in jax.devices())
physics_config = os.environ.get('AUDIT_PHYSICS_CONFIG', 'config/SK_WAND_physics_config.json')
output_name = os.environ.get('AUDIT_RESULT_NAME', 'results.json')
out = {'partition': os.environ['SLURM_JOB_PARTITION'], 'job': os.environ.get('SLURM_JOB_ID'), 'physics_config': physics_config, 'checks': {}}
track = ParticleParams.from_cartesian(energy=jnp.float32(0), position=jnp.zeros(3), direction=jnp.array([0.,0.,1.]), t0=jnp.float32(0))
rng = np.random.default_rng(20261006)

def mkdata(n, wavelengths, origins):
    d = rng.normal(size=(n,3)).astype('float32')
    d /= np.linalg.norm(d,axis=1,keepdims=True)
    return dict(photon_origins=jnp.asarray(origins*100), photon_directions=jnp.asarray(d),
                photon_times=jnp.zeros(n), wavelengths=jnp.asarray(wavelengths), N=jnp.int32(n),
                apply_rotation=False, rotation_axis=jnp.array([1.,0.,0.]), rotation_angle=0.,
                apply_translation=False, translation_vector=jnp.zeros(3), photon_segment_index=jnp.arange(n,dtype=jnp.int32))

# A 24-step call exposes every step's deposits without changing the transport draws.
# QE is evaluated analytically here to avoid confusing independent QE draws between K values
# with actual additional light; DATA still calls photon_iteration_sample for all transport.
n = 65536
K = int(os.environ.get('AUDIT_K', '24'))
sim = setup_event_simulator('config/SK_WAND_geom_config.json', 0, K=K, is_data=True,
    temperature=0., physics_config=physics_config,
    default_detector_params=True, hit_mode='per_photon', deposit_leg_bound=True)
sensor_count = len(sim.det_geom.sensor_points)
wavelengths = np.resize(np.array([300,325,350,375,400,450,500,600],dtype='float32'),n)
origins = np.zeros((n,3),dtype='float32')
origins[n//2:] = [13.,0.,10.]
data = mkdata(n,wavelengths,origins)
t = time.time()
lw, ts, idx, totals = (np.asarray(v) for v in sim(track,jax.random.PRNGKey(9437),data))
w = np.exp(lw).reshape(K,-1,n)
tflat = ts.reshape(K,-1,n)
major = w>1e-5
per_photon = major.sum(axis=(0,1))
records=[]
for lam in np.unique(wavelengths):
    subset=w[:,:,wavelengths==lam]
    stepcharge=subset.sum(axis=(1,2))
    total=float(stepcharge.sum())
    records.append({'wavelength_nm':float(lam),'photons':int(np.sum(wavelengths==lam)),
                    'charge_per_step':stepcharge.tolist(),'charge':total,
                    'fraction_after_12':float(stepcharge[12:].sum()/max(total,1e-20)),
                    'fraction_after_6':float(stepcharge[6:].sum()/max(total,1e-20))})
tail_check_name = f'K12_vs_K{K}'
out['checks'][tail_check_name]={'seconds':time.time()-t,'records':records,
    'max_major_deposits_per_source_photon':int(per_photon.max()),
    'source_photons_with_multiple_major_deposits':int(np.sum(per_photon>1)),
    'flat_charge':float(w.sum()),'per_sensor_charge':float(totals.sum())}
assert per_photon.max()<=1
assert np.isclose(w.sum(),totals.sum(),rtol=2e-6)
print(json.dumps(out['checks'][tail_check_name]),flush=True)

# Independent exact labels catch flattened candidate/step indexing and final chunk padding.
def fake_sim(track,key,p):
    b=len(p['photon_times']); c=4; k=3
    ids=jnp.arange(k*c*b); ph=ids%b
    valid=ph<p['N']
    seg=p['photon_segment_index'][ph]
    wt=jnp.where(valid,1.,0.)
    ti=jnp.where(valid,p['photon_times'][ph]+1.,jnp.inf)
    si=jnp.where(valid,seg%7,0)
    q=jax.ops.segment_sum(wt,si,7)
    tt=jax.ops.segment_min(ti,si,7)
    tt=jnp.where(jnp.isfinite(tt),tt,0.)
    return q,tt,tt,wt,ti,ti,si,seg
N=37
ret=_trace_event_bucketed(fake_sim,np.zeros((N,3),'float32'),np.tile(np.array([0.,0.,1.],dtype='float32'),(N,1)),
    np.arange(N,dtype='float32'),np.full(N,400,'float32'),np.arange(N,dtype='int32'),7,(8,16),jax.random.PRNGKey(11))
q,tt,tr,ww,times,treco,ss,segs,gid=ret
assert len(ww)==12*N
assert np.all(np.bincount(gid,minlength=N)==12)
assert np.array_equal(segs,gid)
assert np.array_equal(ss,gid%7)
assert np.array_equal(times,gid+1)
assert q.sum()==12*N
out['checks']['chunk_alignment']={'photons':N,'chunks':_split_into_chunks(N,(8,16)),'rows':len(ww),'each_source_photon_repeated':12,'passed':True}

# Explicit demonstration of the numerical ghost which production rejects by finite time.
out['checks']['hard_first_hit_weights']=np.asarray(first_hit_survival(jnp.ones((3,1)),jnp.array([[1.],[2.],[3.]]))).reshape(-1).tolist()
with open('audit_wc_20261006/simulator_statistics/'+output_name,'w') as f: json.dump(out,f,indent=2)
print('All simulator statistics checks passed',flush=True)
