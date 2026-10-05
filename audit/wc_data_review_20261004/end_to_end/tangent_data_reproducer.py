"""Full SK_WAND DATA reproduction of endpoint-grid first-hit loss."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.simulation import setup_event_simulator
from lucid.sources.event_builder import _trace_event_bucketed

geom = 'config/SK_WAND_geom_config.json'
phys = 'config/SK_WAND_physics_config.json'
det = generate_detector(geom)
centers = np.asarray(det.all_points)
dp = load_detector_params(phys, num_sensors=len(centers))
dp = dp._replace(
    scattering=dp.scattering._replace(scatter_length=jnp.asarray(1e20), mie_scatter_length=jnp.asarray(1e20)),
    absorption=dp.absorption._replace(absorption_length=jnp.asarray(1e20)),
    reflection=dp.reflection._replace(sensor_reflection_rate=jnp.asarray(0.), wall_reflection_rate=jnp.asarray(0.)),
    response=dp.response._replace(qe=jnp.asarray(1.), tts=jnp.asarray(0.)))
sim = setup_event_simulator(geom, 0, K=12, is_data=True, temperature=0.,
    physics_config=phys, default_detector_params=dp, wavelength_mode=False,
    hit_mode='per_segment', deposit_leg_bound=True)
origin = np.asarray([[16.858586662707683, -0.352948409856257, -0.4]], np.float32)
direction = np.asarray([[0., 0., 1.]], np.float32)
# Independently enumerate entry to every PMT sphere with float64 arithmetic.
oc = origin.astype(np.float64) - centers
b = oc[:, 2]
discriminant = b * b - np.sum(oc * oc, axis=1) + det.S_radius ** 2
entry = -b - np.sqrt(np.maximum(discriminant, 0.))
entry = np.where((discriminant > 0.) & (entry > 0.), entry, np.inf)
true_sensor = int(np.argmin(entry))
true_time = float(entry[true_sensor] / (0.299792 / 1.33) + 7.)
out = _trace_event_bucketed(sim, origin, direction, np.asarray([7.], np.float32),
    np.asarray([400.], np.float32), np.asarray([0], np.int32),
    len(centers), (256,), jax.random.PRNGKey(0))
keep = out[3] > 1e-5
observed = {'expected_sensor': true_sensor, 'expected_time_ns': true_time,
            'expected_entry_m': float(entry[true_sensor]),
            'observed_sensor': out[6][keep].tolist(),
            'observed_time_ns': out[4][keep].tolist(),
            'observed_pe': float(out[0].sum())}
print(json.dumps(observed, indent=2), flush=True)
with open('audit/wc_data_review_20261004/end_to_end/tangent_data_results.json', 'w') as f:
    json.dump(observed, f, indent=2)
# Regression expectation for the suggested geometry fix.
assert len(out[6][keep]) == 1 and out[6][keep][0] == true_sensor, observed
