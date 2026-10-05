import os
assert os.environ['SLURM_JOB_PARTITION'] in ('milano', 'roma')
import jax.numpy as jnp
from lucid.detector_params import load_physics_config
from lucid.wavelength.medium import load_qe_curve, make_medium
from optical_model_fixed import evaluate_optical_model

dp, material, qe_path = load_physics_config('config/SK_WAND_physics_config.json')
medium = make_medium('water', jnp.linspace(300., 648.22, 200), material)
qe = load_qe_curve(qe_path)
wavelengths = jnp.array([275., 674.])
assert jnp.all(qe(wavelengths) == 0.)
actual = evaluate_optical_model(dp, wavelengths, medium, 2, qe_fn=qe).qe
print('QE expected [0, 0], actual:', actual)
assert jnp.all(actual == 0.), 'The medium-grid clamp changes photon wavelengths before QE lookup.'
