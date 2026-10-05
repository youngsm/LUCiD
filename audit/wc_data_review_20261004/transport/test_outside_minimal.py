"""Run only in a milano/roma allocation; expected failure on checkout82f8d24."""
import os
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
import jax
import jax.numpy as jnp
from lucid.detector_params import ParticleParams
from lucid.simulation import setup_event_simulator


def test_photons_outside_inner_detector_cannot_make_id_photoelectrons():
    sim = setup_event_simulator('config/SK_WAND_geom_config.json', 0, K=12,
        is_data=True, temperature=0., default_detector_params=True,
        physics_config='config/SK_WAND_physics_config.json',
        hit_mode='per_segment', deposit_leg_bound=True)
    n = 4096
    data = dict(photon_origins=jnp.tile(jnp.array([1645.86743, -34.457591, 69.98431]), (n, 1)),
        photon_directions=jnp.tile(jnp.array([-0.999780918, 0.020931238, 0.]), (n, 1)),
        photon_times=jnp.zeros(n), wavelengths=jnp.full(n, 400.), N=jnp.asarray(n),
        apply_rotation=False, rotation_axis=jnp.array([0., 0., 1.]), rotation_angle=0.,
        apply_translation=True, translation_vector=jnp.array([0.999780918, -0.020931238, 0.]),
        photon_segment_index=jnp.zeros(n, dtype=jnp.int32))
    particle = ParticleParams.from_cartesian(1000., [0., 0., 0.], [1., 0., 0.])
    charge = sim(particle, jax.random.PRNGKey(12341), data)[0]
    assert float(charge.sum()) == 0.0, f'External photons made {float(charge.sum())} ID PE'


if __name__ == '__main__':
    test_photons_outside_inner_detector_cannot_make_id_photoelectrons()
