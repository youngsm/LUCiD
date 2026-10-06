"""Independent SK_WAND optical calibration checks; execute only inside Slurm CPU.

Reference: Abe et al., arXiv:1307.0162, Eqs. 14--17 and Table 3.
The fitted asymmetric coefficient belongs to p(mu)=2mu, mu in [0,1].
This diagnostic quantifies the shipped alternative HG law, not a large event bug.
"""
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}, "Use approved CPU partition"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.wavelength.medium import make_medium, load_qe_curve
from lucid.wavelength.scattering import hg_sample_cos_theta

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
dp, medium_path, qe_path = load_physics_config(str(ROOT / "config/SK_WAND_physics_config.json"))
wl = np.array([300., 337., 375., 400., 405., 445., 500., 600.])
medium = make_medium("water", jnp.asarray(wl), medium_model_path=medium_path)
# Literal published fit, independently evaluated with NumPy rather than JSON.
sym = 8.51e7 / wl**4 * (1 + 1.14e5 / wl**2)
asym = 1.e-4 * (1 + 4.62e6 / wl**4 * (wl - 392.)**2)
abs_blue = .624 * 2.96e7 / wl**4 + .624 * 3.24e-2 * (wl / 500.)**10.9
np.testing.assert_allclose(medium.scatter_coeff, sym, rtol=2.e-6)
np.testing.assert_allclose(medium.mie_scatter_coeff, asym, rtol=2.e-6)
np.testing.assert_allclose(np.asarray(medium.absorption_coeff)[wl < 452], abs_blue[wl < 452], rtol=2.e-6)

n = 1_000_000
u = (jnp.arange(n) + .5) / n
mu = np.asarray(hg_sample_cos_theta(u, dp.scattering.g))
mu_sk = np.sqrt(np.asarray(u))
angles = [5., 15., 45., 90.]
rows = []
for i, w in enumerate(wl):
    rows.append({"wavelength_nm": float(w), "sym_length_m": float(1/sym[i]),
                 "asym_length_m": float(1/asym[i]),
                 "abs_length_m": float(1/np.asarray(medium.absorption_coeff)[i]),
                 "prob_any_asymmetric_scattering_20m": float(-np.expm1(-20*asym[i])),
                 "prob_any_asymmetric_scattering_50m": float(-np.expm1(-50*asym[i]))})
q400 = float(load_qe_curve(qe_path)(400.))
grid = np.linspace(300., 700., 4001)
grid_medium = make_medium("water", jnp.asarray(grid), medium_model_path=medium_path)
grid_qe = np.asarray(load_qe_curve(qe_path)(jnp.asarray(grid)))
path_estimates = []
for path in (10., 20., 35., 50.):
    # Straight-ray Cherenkov spectrum weighted by QE and absorption survival.
    # This is an analytic scale estimate, not a detector-level detected-event bound.
    weights = grid_qe / grid**2 * np.exp(-path*np.asarray(grid_medium.absorption_coeff))
    p_mie = -np.expm1(-path*np.asarray(grid_medium.mie_scatter_coeff))
    path_estimates.append({"distance_m": path,
                           "spectrum_weighted_probability_asymmetric": float(np.sum(weights*p_mie)/np.sum(weights))})
result = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "partition": os.environ["SLURM_JOB_PARTITION"],
    "node": os.uname().nodename,
    "jax_devices": [str(d) for d in jax.devices()],
    "water_reference_equations_pass": True,
    "water_rows": rows,
    "straight_path_asymmetric_scale_estimates": path_estimates,
    "phase_moments": {
        "HG_g_config": float(dp.scattering.g), "HG_mean_cos": float(mu.mean()),
        "SK_published_mean_cos": float(mu_sk.mean()),
        "HG_mean_cos_squared": float(np.square(mu).mean()),
        "SK_published_mean_cos_squared": float(np.square(mu_sk).mean()),
        "HG_probability_backscatter": float((mu < 0).mean()),
        "SK_published_probability_backscatter": float((mu_sk < 0).mean()),
        "forward_cones": [{"half_angle_degrees": a,
                           "HG_probability": float((mu > np.cos(np.deg2rad(a))).mean()),
                           "SK_published_probability": float((mu_sk > np.cos(np.deg2rad(a))).mean())}
                          for a in angles]},
    "QE_interpretation_diagnostic": {
        "config_qe_400nm": q400,
        "config_qe_corrections": np.asarray(dp.per_pmt.qe_corrections).tolist(),
        "if_table_is_cathode_qe_then_pde_at_CE_073": q400*.73,
        "if_table_is_cathode_qe_then_relative_overcount_at_CE_073": 1/.73-1,
        "if_table_is_cathode_qe_then_pde_at_CE_067": q400*.67,
        "if_table_is_cathode_qe_then_relative_overcount_at_CE_067": 1/.67-1,
        "caution": "The QE table measurement and CE convention are unproven. Conditional estimates are not measured event errors."
    }
}
(OUT / "optical_reference_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
