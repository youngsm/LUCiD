import os
assert os.environ['SLURM_JOB_PARTITION'] in ('milano', 'roma')
import jax.numpy as jnp
import numpy as np
from lucid.simulation import setup_event_simulator
sim = setup_event_simulator('config/SK_WAND_geom_config.json',0,K=1,is_data=True,temperature=0.,physics_config='config/SK_WAND_physics_config.json',default_detector_params=True,hit_mode='per_segment',deposit_leg_bound=True)
print('geom', sim.det_geom.detector, 'rad',sim.det_geom.sensor_radius)
for i in (0, 100, 1000, 5000, 10000):
    pmt = sim.det_geom.sensor_points[i]
    direction = pmt / jnp.linalg.norm(pmt)
    r = sim.det_geom.propagator(jnp.zeros((1,3)),direction[None,:])
    print('pmt',i,np.asarray(pmt))
    print({k: np.asarray(v).tolist() for k,v in r.items()})
from lucid.detector_params import ParticleParams
import jax
n=30000
pmt=sim.det_geom.sensor_points[0]
direction=pmt/jnp.linalg.norm(pmt)
particle=ParticleParams.from_cartesian(1000., [0.,0.,0.],direction)
pd=dict(photon_origins=jnp.zeros((n,3)), photon_directions=jnp.tile(direction,(n,1)),photon_times=jnp.zeros(n),N=n,rotation_axis=jnp.array([0.,0.,1.]),rotation_angle=0.,apply_rotation=False,apply_translation=False,translation_vector=jnp.zeros(3),photon_segment_index=jnp.zeros(n,dtype=jnp.int32))
for wavelength in (275.,400.,674.):
    result=sim(particle,jax.random.PRNGKey(73),dict(pd,wavelengths=jnp.full(n,wavelength)))
    print('sim',wavelength,[(np.asarray(x).shape,float(np.asarray(x).sum()),float(np.asarray(x).max())) if x is not None else None for x in result])
