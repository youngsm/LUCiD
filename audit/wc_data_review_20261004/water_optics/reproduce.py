"""SK_WAND optical audit; execute through run.sbatch on milano/roma only."""
import json
import os
from pathlib import Path
import re

assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import ParticleParams, load_physics_config
from lucid.simulation import setup_event_simulator
from lucid.simulation.optics import solve_rayleigh_inverse_cdf
from lucid.wavelength.medium import load_qe_curve, make_medium, qe_curve_bounds
from lucid.wavelength.optical_model import evaluate_optical_model
from lucid.wavelength.scattering import hg_sample_cos_theta

OUT = Path(__file__).parent
RESULT = {"job": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
          "jax": jax.__version__, "numpy": np.__version__}


def emit(key, value):
    RESULT[key] = value
    print(json.dumps({key: value}), flush=True)


def main():
    # End-user simulator path, exact SK_WAND geometry/physics and production hit mode.
    sim = setup_event_simulator("config/SK_WAND_geom_config.json", 0,
                               K=1, is_data=True, temperature=0.0,
                               physics_config="config/SK_WAND_physics_config.json",
                               default_detector_params=True, hit_mode="per_segment",
                               deposit_leg_bound=True)
    n = 30000
    pmt = np.asarray(sim.det_geom.sensor_points[0])
    direction = pmt / np.linalg.norm(pmt)
    particle = ParticleParams.from_cartesian(1000., [0., 0., 0.], direction)
    pd = dict(photon_origins=jnp.tile(jnp.asarray((pmt - direction) * 100.), (n, 1)),
              photon_directions=jnp.tile(jnp.asarray(direction), (n, 1)),
              photon_times=jnp.zeros(n), N=n,
              rotation_axis=jnp.asarray([0., 0., 1.]), rotation_angle=0.,
              apply_rotation=False, apply_translation=False,
              translation_vector=jnp.zeros(3), photon_segment_index=jnp.zeros(n, dtype=jnp.int32))
    rows = []
    for wavelength in (275., 400., 500., 674.):
        result = sim(particle, jax.random.PRNGKey(73),
                     dict(pd, wavelengths=jnp.full(n, wavelength)))
        charge = np.asarray(result[0])
        times = np.asarray(result[1])
        rows.append(dict(wavelength_nm=wavelength, input_photons=n,
                         detected_pe=float(charge.sum()),
                         first_hit_time_ns=float(times[charge > 0].min(initial=np.inf)),
                         hit_sensors=int((charge > 0).sum())))
    emit("e2e", dict(pmt_center_m=pmt.tolist(), center_distance_m=float(np.linalg.norm(pmt)),
                     rows=rows))
    assert rows[0]["detected_pe"] > 0 and rows[-1]["detected_pe"] > 0
    assert rows[1]["first_hit_time_ns"] == rows[2]["first_hit_time_ns"]

    dp, medium_path, qe_path = load_physics_config("config/SK_WAND_physics_config.json")
    qe = load_qe_curve(qe_path)
    qe_lo, qe_hi = qe_curve_bounds(qe_path)
    grid = jnp.linspace(max(300., qe_lo), min(700., qe_hi), 200)
    medium = make_medium("water", wavelength_grid=grid, medium_model_path=medium_path)
    wavelengths = jnp.asarray([275., 294., 299., 300., 337., 375., 398., 400., 405., 445.,
                               465., 500., 600., 648.22, 674.])
    oa = evaluate_optical_model(dp, wavelengths, medium, len(wavelengths), qe_fn=qe)
    raw_qe, clipped_qe = np.asarray(qe(wavelengths)), np.asarray(oa.qe)
    emit("spectral", dict(wavelengths_nm=np.asarray(wavelengths).tolist(),
                          raw_qe=raw_qe.tolist(), simulated_qe=clipped_qe.tolist()))
    assert raw_qe[0] == raw_qe[-1] == 0.
    assert clipped_qe[0] > 0. and clipped_qe[-1] > 0.

    lam = jnp.linspace(275., 674., 100001)
    all_oa = evaluate_optical_model(dp, lam, medium, len(lam), qe_fn=qe)
    corrected = np.asarray(qe(lam))
    actual = np.asarray(all_oa.qe)
    w = 1. / np.asarray(lam)**2
    distance_rows = []
    for distance in (0., 17., 35.):
        transmission = np.exp(-distance * (1. / np.asarray(all_oa.abs_len)
                                           + 1. / np.asarray(all_oa.scatter_len)
                                           + 1. / np.asarray(all_oa.mie_len)))
        expected = np.trapezoid(w * corrected * transmission, np.asarray(lam))
        simulated = np.trapezoid(w * actual * transmission, np.asarray(lam))
        distance_rows.append(dict(distance_m=distance, relative_charge_bias=float(simulated / expected - 1.)))
    emit("spectral_charge_bias_bare_cherenkov_275_674_nm", distance_rows)

    # Compare the implemented coefficients to the independently transcribed SK Table 3 equations.
    exact = make_medium("water", wavelength_grid=wavelengths, medium_model_path=medium_path)
    wl = np.asarray(wavelengths, dtype=np.float64)
    ref_abs = .624 * 2.96e7 / wl**4 + .624 * .0324 * (wl / 500.)**10.9
    ref_sym = 8.51e7 / wl**4 * (1. + 1.14e5 / wl**2)
    ref_asym = 1.e-4 * (1. + 4.62e6 / wl**4 * (wl - 392.)**2)
    blue = (wl >= 300.) & (wl < 452.)
    assert np.allclose(np.asarray(exact.absorption_coeff)[blue], ref_abs[blue], rtol=1e-6)
    assert np.allclose(exact.scatter_coeff, ref_sym, rtol=1e-6)
    assert np.allclose(exact.mie_scatter_coeff, ref_asym, rtol=1e-6)
    emit("coefficient_lengths_m", dict(wavelengths_nm=wl.tolist(),
                                      absorption=(1. / np.asarray(exact.absorption_coeff)).tolist(),
                                      symmetric=(1. / np.asarray(exact.scatter_coeff)).tolist(),
                                      asymmetric=(1. / np.asarray(exact.mie_scatter_coeff)).tolist()))
    emit("grid_interpolation_max_relative_error", dict(
        absorption=float(np.max(np.abs(np.asarray(oa.abs_len)[blue] / (1. / np.asarray(exact.absorption_coeff)[blue]) - 1.))),
        symmetric=float(np.max(np.abs(np.asarray(oa.scatter_len)[blue] / (1. / np.asarray(exact.scatter_coeff)[blue]) - 1.)))))

    # WCSim phase-index table and Geant4 CalculateGROUPVEL discrete derivative prescription.
    text = (OUT / "WCSimConstructMaterials.cc").read_text()
    energy_text = re.search(r"G4double ENERGY_water\[NUMENTRIES_water\] =\s*\{(.*?)\}", text, re.S)[1]
    index_text = re.search(r"G4double RINDEX1\[NUMENTRIES_water\] =\s*\{(.*?)\}", text, re.S)[1]
    energy = np.asarray([float(x) * 1.e9 for x in re.findall(r"([0-9.eE+-]+)\*GeV", energy_text)])
    index = np.asarray([float(x) for x in re.findall(r"[0-9.]+", index_text)])
    assert len(energy) == len(index) == 60
    ng_interval = .5 * (index[:-1] + index[1:]) + np.diff(index) / np.log(energy[1:] / energy[:-1])
    ng_first = index[0] + (index[1] - index[0]) / np.log(energy[1] / energy[0])
    ng_last = index[-1] + (index[-1] - index[-2]) / np.log(energy[-1] / energy[-2])
    group_energy = np.r_[energy[0], .5 * (energy[:-2] + energy[1:-1]), energy[-1]]
    group_speed = .299792458 / np.r_[ng_first, ng_interval[:-1], ng_last]
    test_wl = np.asarray([300., 337., 375., 398., 400., 405., 445., 500., 600., 648.22])
    test_energy = 1239.841984332 / test_wl
    phase_index = np.interp(test_energy, energy, index)
    speed = np.interp(test_energy, group_energy, group_speed)
    rows = []
    for distance in (17., 35.):
        rows.append(dict(distance_m=distance, lucid_time_ns=distance / medium.speed_of_light,
                         group_time_ns=(distance / speed).tolist(),
                         early_bias_ns=(distance / speed - distance / medium.speed_of_light).tolist()))
    emit("timing", dict(wavelengths_nm=test_wl.tolist(), reference_phase_index=phase_index.tolist(),
                        reference_group_index=(.299792458 / speed).tolist(), lucid_index=medium.refractive_index,
                        rows=rows))
    physical_lam = np.asarray(lam)
    physical_speed = np.interp(1239.841984332 / physical_lam, group_energy, group_speed)
    timing_weight = w * corrected * np.exp(-17. / np.asarray(all_oa.abs_len))
    norm = np.trapezoid(timing_weight, physical_lam)
    group_times = 17. / physical_speed
    mean_t = np.trapezoid(timing_weight * group_times, physical_lam) / norm
    spread_t = np.sqrt(np.trapezoid(timing_weight * (group_times - mean_t)**2, physical_lam) / norm)
    emit("detected_spectral_timing_17m", dict(mean_reference_ns=mean_t,
                                            reference_dispersion_sigma_ns=spread_t,
                                            early_mean_bias_ns=mean_t - 17. / medium.speed_of_light))

    uniform = jax.random.uniform(jax.random.PRNGKey(591), (200000,))
    rayleigh = np.asarray(jax.jit(jax.vmap(solve_rayleigh_inverse_cdf))(uniform))
    mie = np.asarray(hg_sample_cos_theta(uniform, .95))
    emit("scatter_angles", dict(rayleigh_mean_cos=float(rayleigh.mean()),
                                rayleigh_mean_cos2=float((rayleigh**2).mean()),
                                rayleigh_expected_mean_cos2=.4,
                                hg_mean_cos=float(mie.mean()), hg_backward_fraction=float((mie < 0).mean()),
                                sk_asymmetric_mean_cos=2. / 3., sk_asymmetric_backward_fraction=0.))
    assert abs(rayleigh.mean()) < .005 and abs((rayleigh**2).mean() - .4) < .005
    (OUT / "results.json").write_text(json.dumps(RESULT, indent=2) + "\n")


if __name__ == "__main__":
    main()
