"""CPU Slurm only: SK_WAND incident QE and valid negative PMT times."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import ParticleParams, load_physics_config
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarMixReflection, scalar_mix_reflection
from lucid.simulation.sensor_response import make_hits_data, make_hits_per_photon


GEOM = "config/SK_WAND_geom_config.json"
PHYS = "config/SK_WAND_physics_config.json"
OUT = Path("audit/wc_data_review_20261004/pmt_detection")


def test_sensor_boundary_incident_qe():
    dp, _, _ = load_physics_config(PHYS, num_sensors=1)
    n = 300_000
    reflectance = float(dp.reflection.sensor_reflection_rate)
    qe = float(dp.response.qe)
    refs = ScalarMixReflection(dp.reflection.wall_reflection_rate,
                              dp.reflection.sensor_reflection_rate,
                              dp.reflection.wall_fspec, dp.reflection.sensor_fspec)
    keys = jax.random.split(jax.random.PRNGKey(223), n)

    def step(key):
        return photon_iteration_sample(
            jnp.zeros(3), jnp.array([1.0, 0.0, 0.0]), 0.0, 1.0,
            jnp.array([1.0, 0.0, 0.0]), 1e20, 1e20, 0.95, refs,
            1e20, True, 400.0, key, 0.225,
            reflection_fn=scalar_mix_reflection)

    results = jax.jit(jax.vmap(step))(keys)
    boundary_weights = results[3] * results[4]
    pp = make_hits_per_photon(
        boundary_weights, jnp.zeros(n, dtype=jnp.int32), jnp.ones(n), 1,
        qe=qe, qe_corrections=jnp.ones(1), rng_key=jax.random.PRNGKey(889))
    measured = float(pp[0].sum()) / n
    expected_current = (1.0 - reflectance) * qe
    assert abs(measured - expected_current) < 0.002
    assert abs(measured / qe - 0.75) < 0.01
    return {"n_incident": n, "qe_at_400_nm": qe,
            "sensor_reflectance": reflectance,
            "fraction_nonreflected": float(boundary_weights.mean()),
            "measured_direct_detection_probability": measured,
            "current_formula": expected_current,
            "required_incident_probability": qe,
            "relative_direct_pe_loss": 1.0 - measured / qe}


def test_sk_wand_data_pencil_beam():
    detector = generate_detector(GEOM)
    points = np.asarray(detector.all_points)
    barrel_ids = np.where(np.hypot(points[:, 0], points[:, 1]) > 0.99 * detector.r)[0]
    pmt_id = int(barrel_ids[np.argmin(np.abs(points[barrel_ids, 2]))])
    target = points[pmt_id]
    direction = np.array([target[0], target[1], 0.0])
    direction /= np.linalg.norm(direction)
    origin = target - 5.0 * direction
    n = 50_000
    dp, _, _ = load_physics_config(PHYS, num_sensors=len(points))
    particle = ParticleParams.from_cartesian(1000.0, origin, direction)
    photon_data = {
        "photon_origins": jnp.broadcast_to(jnp.asarray(origin * 100.0), (n, 3)),
        "photon_directions": jnp.broadcast_to(jnp.asarray(direction), (n, 3)),
        "photon_times": jnp.ones(n), "wavelengths": jnp.full(n, 400.0),
        "N": n, "rotation_axis": jnp.array([0.0, 0.0, 1.0]),
        "rotation_angle": 0.0, "apply_rotation": False,
        "photon_segment_index": jnp.zeros(n, dtype=jnp.int32),
    }
    rows = []
    for k in (1, 12):
        for reflectance in (0.0, 0.25):
            params = dp._replace(reflection=dp.reflection._replace(
                sensor_reflection_rate=jnp.asarray(reflectance)))
            sim = setup_event_simulator(
                GEOM, 0, K=k, is_data=True, temperature=0.0,
                charge_resolution=None, physics_config=PHYS,
                default_detector_params=params, hit_mode="per_segment",
                deposit_leg_bound=True)
            output = sim(particle, jax.random.PRNGKey(412), photon_data)
            charge = np.asarray(output[3])
            times = np.asarray(output[4])
            indices = np.asarray(output[6])
            direct = (charge > 0) & (times < 30.0) & (indices == pmt_id)
            rows.append({"K": k, "sensor_reflectance": reflectance,
                         "input_photons": n, "total_detected_pe": float(charge.sum()),
                         "direct_detected_pe": float(charge[direct].sum()),
                         "direct_probability": float(charge[direct].sum()) / n,
                         "pmt_id": pmt_id, "origin_m": origin.tolist()})
            print(json.dumps({"beam": rows[-1]}), flush=True)
    for k in (1, 12):
        pair = [r for r in rows if r["K"] == k]
        ratio = pair[1]["direct_detected_pe"] / pair[0]["direct_detected_pe"]
        assert abs(ratio - 0.75) < 0.02
    return rows


def test_negative_tts_keeps_photoelectrons():
    results = []
    for occupancy in (1, 10):
        n_sensors = 20_000
        weights = jnp.ones(n_sensors * occupancy)
        indices = jnp.repeat(jnp.arange(n_sensors), occupancy)
        times = jnp.full(weights.shape, 0.1)
        kwargs = dict(qe=1.0, qe_corrections=jnp.ones(n_sensors),
                      rng_key=jax.random.PRNGKey(123), tts=3.0)
        q, t = make_hits_data(weights, indices, times, n_sensors, **kwargs)
        shifted_q, _ = make_hits_data(weights, indices, times + 100, n_sensors, **kwargs)
        per_photon = make_hits_per_photon(weights, indices, times, n_sensors, **kwargs)
        assert float(shifted_q.sum()) == n_sensors * occupancy
        assert float(per_photon[0].sum()) == n_sensors * occupancy
        assert float(q.sum()) < 0.55 * n_sensors * occupancy
        results.append({"occupancy": occupancy, "input_pe": n_sensors * occupancy,
                        "realistic_kept_pe": float(q.sum()),
                        "realistic_shifted_by_100_ns_kept_pe": float(shifted_q.sum()),
                        "per_segment_kept_pe": float(per_photon[0].sum()),
                        "fraction_sensor_charge_erased": float((q == 0).mean())})
    return results


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
    report = {"job_id": os.environ["SLURM_JOB_ID"],
              "partition": os.environ["SLURM_JOB_PARTITION"],
              "jax_version": jax.__version__, "numpy_version": np.__version__}
    report["boundary_qe"] = test_sensor_boundary_incident_qe()
    print(json.dumps(report["boundary_qe"]), flush=True)
    report["negative_tts"] = test_negative_tts_keeps_photoelectrons()
    print(json.dumps(report["negative_tts"]), flush=True)
    report["beam"] = test_sk_wand_data_pencil_beam()
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
