"""K=1 physical SK_WAND beams against independent Beer-Lambert + incident QE.

Expected coefficients use Abe et al. arXiv:1307.0162 Eqs.14--17/Table3,
with the configured Pope-Fry red-water knots and raw configured QE table.
No LUCiD optical-model evaluator is used to calculate the oracle.
Run only in milano/roma Slurm CPU allocation.
"""
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binomtest, chi2

from lucid.detector_params import load_physics_config, ParticleParams
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation import setup_event_simulator

BASE = Path(__file__).resolve().parent
N = 100000
WAVELENGTHS = (300., 350., 400., 450., 550., 600.)
DISTANCES = (5., 20.)
REFLECTIONS = (0., .25)


def independent_coefficients(wavelength, water):
    # Published fitted coefficients in m^-1 for wavelength in nm. All probe
    # wavelengths avoid the implementation's 452--476nm interpolation blend.
    assert wavelength < 452 or wavelength > 476
    p0, p1, p2, p3 = .624, 2.96e7, 3.24e-2, 10.9
    p4, p5, p6, p7, p8 = 8.51e7, 1.14e5, 1e-4, 4.62e6, 392.
    rayleigh = p4 / wavelength**4 * (1 + p5 / wavelength**2)
    mie = p6 * (1 + p7 * (wavelength-p8)**2 / wavelength**4)
    if wavelength < 464:
        c_water = p0 * p2 * (wavelength / 500.)**p3
    else:
        # The numerical red-water knots are an explicit input calibration;
        # this checks their end-to-end attenuation, not their provenance.
        ab = water["absorption"]
        c_water = np.interp(wavelength, ab["pope_fry_wavelengths_nm"], ab["pope_fry_absorption_m_inv"])
    absorption = p0 * p1 / wavelength**4 + c_water
    return float(absorption), float(rayleigh), float(mie)


def main():
    physics = "config/SK_WAND_physics_config.json"
    geometry = "config/SK_WAND_geom_config.json"
    water = json.loads(Path("config/materials/water.json").read_text())
    qe_table = json.loads(Path("config/pmt/SK_QE.json").read_text())
    g = DetectorGeometry.from_config(geometry)
    centers = np.asarray(g.sensor_points, np.float64)
    target = int(np.argmin(np.linalg.norm(centers, axis=1)))
    center = centers[target]
    direction = center / np.linalg.norm(center)
    dp, _, _ = load_physics_config(physics, num_sensors=g.num_sensors)
    sim = setup_event_simulator(geometry, 0, K=1, is_data=True,
        temperature=0., wavelength_mode=True, charge_resolution=None,
        hit_mode="per_segment", deposit_leg_bound=True,
        physics_config=physics, default_detector_params=False)
    particle = ParticleParams.from_cartesian(energy=0., position=jnp.zeros(3),
                                             direction=jnp.array([0., 0., 1.]), t0=0.)
    rows = []
    for distance in DISTANCES:
        # Specify water path to PMT sphere ENTRY rather than its center.
        origin = center - (distance + g.sensor_radius) * direction
        assert bool(np.asarray(g.detector.bounds_check(jnp.asarray(origin[None])))[0])
        # Match the cm DATA boundary exactly, then solve a float64 ray/sphere
        # intersection independently for the actual represented coordinates.
        origin_cm = np.asarray(origin * 100., np.float32)
        represented_origin = (origin_cm / np.float32(100.)).astype(np.float64)
        represented_dir = np.asarray(direction, np.float32).astype(np.float64)
        oc = represented_origin - center
        a = np.dot(represented_dir, represented_dir)
        b = np.dot(oc, represented_dir)
        c = np.dot(oc, oc) - g.sensor_radius**2
        entry_t = (-b - np.sqrt(b*b - a*c)) / a
        entry_distance = float(entry_t * np.sqrt(a))
        data = {"photon_origins": jnp.tile(jnp.asarray(origin_cm), (N, 1)),
                "photon_directions": jnp.tile(jnp.asarray(direction, jnp.float32), (N, 1)),
                "photon_times": jnp.ones(N, jnp.float32), "N": jnp.int32(N),
                "apply_rotation": False, "rotation_axis": jnp.array([1.,0.,0.]),
                "rotation_angle": 0., "apply_translation": False,
                "translation_vector": jnp.zeros(3),
                "photon_segment_index": jnp.zeros(N, jnp.int32)}
        for wavelength in WAVELENGTHS:
            absorption, rayleigh, mie = independent_coefficients(wavelength, water)
            incident_qe = float(np.interp(wavelength, qe_table["wavelengths_nm"],
                                          np.asarray(qe_table["qe_percent"]) / 100.))
            expected_probability = incident_qe * np.exp(-entry_distance * (absorption + rayleigh + mie))
            data["wavelengths"] = jnp.full(N, wavelength, jnp.float32)
            for reflection in REFLECTIONS:
                # Keep physical SK_WAND response/medium otherwise unchanged.
                params = dp._replace(reflection=dp.reflection._replace(
                    sensor_reflection_rate=jnp.array(reflection)))
                seed = 12000 + len(rows)
                out = sim(particle, params, jax.random.PRNGKey(seed), data)
                weights, times, sensor = np.asarray(out[3]), np.asarray(out[4]), np.asarray(out[6])
                hit = weights > 0
                count = int(hit.sum())
                assert np.all(sensor[hit] == target)
                assert np.all(np.isfinite(times[hit]))
                assert np.allclose(weights[hit], 1., atol=3e-5)
                expected_count = N * expected_probability
                sd = np.sqrt(N * expected_probability * (1-expected_probability))
                z = float((count-expected_count)/sd)
                p_value = float(binomtest(count, N, expected_probability).pvalue)
                row = {"lambda_nm": wavelength, "distance_requested_m": distance,
                    "analytic_entry_distance_m": entry_distance, "sensor_reflectance": reflection,
                    "n_photons": N, "incident_qe": incident_qe,
                    "alpha_abs_m_inv": absorption, "alpha_ray_m_inv": rayleigh,
                    "alpha_mie_m_inv": mie, "expected_probability": float(expected_probability),
                    "expected_count": float(expected_count), "observed_count": count,
                    "z_binomial": z, "binomial_two_sided_p": p_value}
                rows.append(row)
                print(json.dumps(row), flush=True)
    reflection_differences = []
    for i in range(0, len(rows), 2):
        r0, r25 = rows[i:i+2]
        p = r0["expected_probability"]
        z_diff = (r25["observed_count"] - r0["observed_count"]) / np.sqrt(2*N*p*(1-p))
        reflection_differences.append({"lambda_nm": r0["lambda_nm"],
            "distance_m": r0["distance_requested_m"], "z_reflectance_count_difference": float(z_diff)})
    pearson = float(sum(r["z_binomial"]**2 for r in rows))
    result = {"slurm_job": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
        "target_sensor": target, "K": 1, "total_photons": N*len(rows),
        "max_abs_z": max(abs(r["z_binomial"]) for r in rows),
        "pearson_chi2": pearson, "pearson_dof": len(rows),
        "pearson_p": float(chi2.sf(pearson, len(rows))),
        "reflection_comparisons": reflection_differences, "rows": rows}
    (BASE / "absolute_direct_light_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows", "reflection_comparisons")}, indent=2), flush=True)
    # Generous familywise threshold, with raw data retained even on failure.
    assert result["max_abs_z"] < 5, result
    assert max(abs(r["z_reflectance_count_difference"]) for r in reflection_differences) < 5


if __name__ == "__main__":
    main()
