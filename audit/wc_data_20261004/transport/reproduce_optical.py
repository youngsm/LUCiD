"""Run on Slurm milano/roma only: data-mode optical seam and PMT branch checks."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarReflection
from lucid.wavelength.medium import load_qe_curve, make_medium
from lucid.wavelength.optical_model import evaluate_optical_model


def main():
    assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
    print(json.dumps({"jax": jax.__version__, "numpy": np.__version__,
                      "job": os.environ.get("SLURM_JOB_ID"),
                      "partition": os.environ["SLURM_JOB_PARTITION"]}), flush=True)
    dp, medium_path, qe_path = load_physics_config("config/SK_like_physics_config.json")
    qe = load_qe_curve(qe_path)
    grid = jnp.linspace(300.0, 648.22, 200)
    medium = make_medium("water", wavelength_grid=grid, medium_model_path=medium_path)
    wavelengths = jnp.array([275.0, 294.0, 299.0, 300.0, 400.0, 648.22, 674.0])
    oa = evaluate_optical_model(dp, wavelengths, medium, len(wavelengths), qe_fn=qe)
    direct = np.asarray(qe(wavelengths))
    actual = np.asarray(oa.qe)
    assert direct[0] == direct[-1] == 0.0
    assert actual[0] > 0.0 and actual[-1] > 0.0
    spectral = dict(wavelengths_nm=np.asarray(wavelengths).tolist(),
                    expected_qe=direct.tolist(), actual_qe=actual.tolist())
    print(json.dumps({"spectral_clipping": spectral}), flush=True)

    x = jnp.linspace(275.0, 674.0, 100000)
    reference = np.asarray(qe(x))
    clipped = np.asarray(evaluate_optical_model(dp, x, medium, len(x), qe_fn=qe).qe)
    weights = 1.0 / np.asarray(x)**2
    integrate = getattr(np, "trapezoid", None) or np.trapz
    integrated_bias = float(integrate(clipped * weights, np.asarray(x)) /
                            integrate(reference * weights, np.asarray(x)) - 1.0)
    print(json.dumps({"full_band_qe_fractional_bias": integrated_bias}), flush=True)

    keys = jax.random.split(jax.random.PRNGKey(438), 300000)
    pmt_results = []
    for reflectance in (0.0, 0.25, 0.5):
        refl = ScalarReflection(jnp.asarray(0.0), jnp.asarray(reflectance))
        def step(key):
            return photon_iteration_sample(
                jnp.zeros(3), jnp.array([1.0, 0.0, 0.0]), 0.0, 1.0,
                jnp.array([1.0, 0.0, 0.0]), 1e20, 1e20, 0.95, refl,
                1e20, True, 400.0, key, medium.speed_of_light)[3]
        probability = float(jnp.mean(jax.jit(jax.vmap(step))(keys)))
        measured_qe = float(qe(jnp.asarray(400.0)))
        row = dict(sensor_reflectance=reflectance, fraction_not_reflected=probability,
                   incident_qe=measured_qe, predicted_direct_detection=probability*measured_qe)
        pmt_results.append(row)
        print(json.dumps(row), flush=True)
    out = dict(slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               partition=os.environ["SLURM_JOB_PARTITION"], spectral_clipping=spectral,
               full_band_qe_fractional_bias=integrated_bias,
               reflection_qe=pmt_results)
    Path("audit/wc_data_20261004/transport/optical_results.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
