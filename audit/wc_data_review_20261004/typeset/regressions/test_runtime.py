"""Seven physical contracts, expected to fail on LUCiD 82f8d249."""
import jax
import jax.numpy as jnp
import numpy as np
from lucid.detector_params import ParticleParams
from lucid.wavelength.medium import load_qe_curve, make_medium


def photon_data(origin_m, direction, n, **transforms):
    # Public DATA simulator input uses centimetres; its internal geometry uses metres.
    return dict(photon_origins=jnp.broadcast_to(jnp.asarray(origin_m) * 100., (n, 3)),
        photon_directions=jnp.broadcast_to(jnp.asarray(direction), (n, 3)),
        photon_times=jnp.ones(n), wavelengths=jnp.full(n, 400.), N=jnp.asarray(n),
        apply_rotation=False, rotation_axis=jnp.array([0., 0., 1.]), rotation_angle=0.,
        photon_segment_index=jnp.zeros(n, dtype=jnp.int32), **transforms)


def test_incident_qe_is_preserved_after_reflection(api, transparent_sim):
    target = np.asarray(api.geometry.sensor_points[4104])
    outward = target.copy(); outward[2] = 0.; outward /= np.linalg.norm(outward)
    origin = target - 5. * outward
    n = 32768
    data = photon_data(origin, outward, n)
    particle = ParticleParams.from_cartesian(1000., origin, outward)
    charge = transparent_sim(particle, jax.random.PRNGKey(412), data)[0]
    measured = float(charge.sum()) / n
    required = float(api.dp.response.qe)
    tolerance = 6. * np.sqrt(required * (1. - required) / n)
    assert abs(measured - required) < tolerance, (measured, required, tolerance)


def test_tangent_ray_hits_first_pmt(api):
    origin = jnp.array([[16.858586662707683, -.352948409856257, -.4]])
    direction = jnp.array([[0., 0., 1.]])
    result = api.geometry.propagator(origin, direction)
    weights = np.asarray(result['sensor_weights'])[:, 0]
    selected = int(np.argmax(weights))
    sensor = int(np.asarray(result['sensor_indices'])[selected, 0])
    distance = float(np.asarray(result['times'])[selected, 0, 0])
    assert weights.sum() > .99
    assert sensor == 4105, (sensor, distance)
    assert abs(distance - .16651306) < 1e-5


def test_translated_photons_outside_id_make_zero_charge(transparent_sim):
    n = 4096
    data = photon_data([16.4586743, -.34457591, .6998431],
        [-.999780918, .020931238, 0.], n, apply_translation=True,
        translation_vector=jnp.array([.999780918, -.020931238, 0.]))
    particle = ParticleParams.from_cartesian(1000., [0., 0., 0.], [1., 0., 0.])
    charge = transparent_sim(particle, jax.random.PRNGKey(12341), data)[0]
    assert float(charge.sum()) == 0., float(charge.sum())


def test_two_readout_bursts_survive_100us(api):
    d = api.digitizer
    times = np.array([100., 200100.])
    digits, hits, _ = d.digitize_and_decompose(
        sensor_idx=np.array([0, 1]), charge=np.ones(2), t_true=times, t_reco=times,
        particle_idx=np.zeros(2, int), segment_idx=np.array([0, 1]),
        emission_process=np.zeros(2, int), n_sensors=2, model=api.model,
        rng=np.random.default_rng(2), dark_rate_khz=0., apply_resolution=False)
    assert len(digits['T']) == 2, digits['T']
    assert np.array_equal(digits['T'], times)
    assert float(hits['PE'].sum()) == 2.


def test_qe_uses_unmodified_photon_wavelength(api):
    wavelengths = jnp.array([275., 674.])
    qe = load_qe_curve(api.qe_path)
    medium = make_medium('water', jnp.linspace(300., 648.22, 200), api.material)
    assert bool(jnp.all(qe(wavelengths) == 0.))
    actual = api.optical.evaluate_optical_model(
        api.dp, wavelengths, medium, 2, qe_fn=qe).qe
    assert bool(jnp.all(actual == 0.)), actual


def test_time_jitter_follows_sampled_charge(api):
    q, t = api.digitizer.apply_readout_resolution(
        np.ones(300000), np.full(300000, 100.), api.model, np.random.default_rng(142))
    for lower, upper in [(.25, .5), (1., 1.5), (2., 3.)]:
        mask = (q >= lower) & (q < upper)
        assert mask.sum() > 1000
        sigma = np.maximum(.58, .33 + np.sqrt(10. / np.maximum(q[mask], .5)))
        required_rms = float(np.sqrt(np.mean(sigma**2)))
        measured_rms = float(t[mask].std())
        assert abs(measured_rms - required_rms) < .10, (lower, upper, measured_rms, required_rms)


def test_tts_charge_is_time_translation_invariant(api):
    n = 4096
    args = (jnp.ones(n), jnp.arange(n), jnp.full(n, .1), n)
    kwargs = dict(qe=1., qe_corrections=jnp.ones(n),
                  rng_key=jax.random.PRNGKey(123), tts=3.)
    charge, _ = api.response.make_hits_data(*args, **kwargs)
    shifted, _ = api.response.make_hits_data(args[0], args[1], args[2] + 100., n, **kwargs)
    assert float(shifted.sum()) == n
    assert np.array_equal(np.asarray(charge), np.asarray(shifted))
