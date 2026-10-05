"""Run only within an authorized milano/roma CPU or turing GPU Slurm allocation."""
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "audit_production_sensor_response", ROOT / "lucid/simulation/sensor_response.py")
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)
make_hits_data, make_hits_per_photon = S.make_hits_data, S.make_hits_per_photon


def test_negative_tts_first_arrival_deletes_all_charge():
    result = {}
    for pe_per_sensor in (1, 10):
        n_sensors = 20_000
        n_photons = n_sensors * pe_per_sensor
        weights = jnp.ones(n_photons)
        indices = jnp.repeat(jnp.arange(n_sensors), pe_per_sensor)
        times = jnp.full(n_photons, 0.1)
        kwargs = dict(qe=1.0, qe_corrections=jnp.ones(n_sensors),
                      rng_key=jax.random.PRNGKey(123), tts=3.0)
        q, t = make_hits_data(weights, indices, times, n_sensors, **kwargs)
        pp = make_hits_per_photon(weights, indices, times, n_sensors, **kwargs)
        q_np, t_np, q_pp = np.asarray(q), np.asarray(t), np.asarray(pp[0])
        assert np.all(q_pp == pe_per_sensor)
        assert float(q_np.sum()) < 0.55 * n_photons
        result[str(pe_per_sensor)] = {
            "input_detected_pe": n_photons,
            "realistic_output_pe": float(q_np.sum()),
            "per_segment_output_pe": float(q_pp.sum()),
            "fraction_all_sensor_charge_deleted": float((q_np == 0).mean()),
            "time_shifted_repeated_realistic_output_pe": float(np.asarray(
                make_hits_data(weights, indices, times + 100.0, n_sensors, **kwargs)[0]).sum()),
            "negative_times_survive_per_segment": int((np.asarray(pp[2]) < 0).sum()),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(test_negative_tts_first_arrival_deletes_all_charge(), indent=2))
