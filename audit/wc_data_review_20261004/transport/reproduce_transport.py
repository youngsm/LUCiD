"""CPU Slurm-only transport oracle and SK_WAND iteration-cutoff measurement."""
import json
import os
from pathlib import Path

assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import load_physics_config
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation.photon_step import photon_iteration_sample
from lucid.simulation.reflection import ScalarReflection, get_reflection_model
from lucid.wavelength.medium import load_qe_curve, make_medium, qe_curve_bounds
from lucid.wavelength.optical_model import evaluate_optical_model

OUT = Path("audit/wc_data_review_20261004/transport/results.json")


def one_step_oracle():
    """Independent exponential-competition formula, not another implementation."""
    n = 500_000
    D, LR, LM, LA, R = 17.0, 30.0, 60.0, 40.0, 0.25
    mu_s, mu_a = 1 / LR + 1 / LM, 1 / LA
    origin = jnp.zeros(3)
    incident = jnp.array([1.0, 0.0, 0.0])
    refl = ScalarReflection(jnp.asarray(0.05), jnp.asarray(R))
    keys = jax.random.split(jax.random.PRNGKey(231), n)

    @jax.jit
    def run(keys):
        return jax.vmap(lambda key: photon_iteration_sample(
            origin, incident, 0.0, D, incident, LR, LM, 0.95,
            refl, LA, True, 400.0, key, 0.299792 / 1.33))(keys)

    pos, dirs, time, detect, survival, continuing, _, indirect = run(keys)
    dist = np.asarray(time) * (0.299792 / 1.33)
    detect, survival, continuing = map(np.asarray, (detect, survival, continuing))
    scatter = dist < D - 1e-3
    reach = ~scatter
    reflect = reach & (detect == 0)
    physical = {
        "reach_fraction": np.exp(-mu_s * D),
        "scatter_fraction": -np.expm1(-mu_s * D),
        "reflected_fraction": np.exp(-mu_s * D) * R,
        "nonreflected_fraction": np.exp(-mu_s * D) * (1 - R),
        "surviving_nonreflected_fraction": np.exp(-(mu_s + mu_a) * D) * (1 - R),
        "surviving_reflected_fraction": np.exp(-(mu_s + mu_a) * D) * R,
        "surviving_scatter_fraction": mu_s / (mu_s + mu_a) * (-np.expm1(-(mu_s + mu_a) * D)),
        "scatter_mean_distance_m": 1 / mu_s - D / np.expm1(mu_s * D),
        "scatter_mean_cosine": 0.95 * (1 / LM) / mu_s,
    }
    measured = {
        "reach_fraction": np.mean(reach),
        "scatter_fraction": np.mean(scatter),
        "reflected_fraction": np.mean(reflect),
        "nonreflected_fraction": np.mean(detect),
        "surviving_nonreflected_fraction": np.mean(detect * survival),
        "surviving_reflected_fraction": np.mean(reflect * survival),
        "surviving_scatter_fraction": np.mean(scatter * survival),
        "scatter_mean_distance_m": np.mean(dist[scatter]),
        "scatter_mean_cosine": np.mean(np.asarray(dirs)[scatter, 0]),
    }
    uncertainty = {}
    for name in physical:
        if name.endswith("fraction"):
            p = physical[name]
            se = np.sqrt(p * (1 - p) / n)
        elif name == "scatter_mean_distance_m":
            se = np.std(dist[scatter], ddof=1) / np.sqrt(np.count_nonzero(scatter))
        else:
            se = np.std(np.asarray(dirs)[scatter, 0], ddof=1) / np.sqrt(np.count_nonzero(scatter))
        uncertainty[name] = {"standard_error": float(se),
                             "z_score": float((measured[name] - physical[name]) / se)}
        assert abs(measured[name] - physical[name]) < 6 * se, (name, measured[name], physical[name], se)
    assert np.allclose(np.linalg.norm(np.asarray(dirs), axis=1), 1.0, atol=1e-5)
    assert np.all(continuing <= survival)
    assert np.all(detect * continuing == 0)
    for a in (pos, dirs, time, detect, survival, continuing):
        assert np.isfinite(np.asarray(a)).all()
    return {"N": n, "parameters": {"D": D, "LR": LR, "LM": LM, "LA": LA, "R": R},
            "physical_oracle": {k: float(v) for k, v in physical.items()},
            "measured": {k: float(v) for k, v in measured.items()}, "uncertainty": uncertainty}


def sk_wand_tail():
    """Use production geometry, photon step, optical curves, and scan key schedule."""
    n, K = 250_000, 32
    geom = DetectorGeometry.from_config(
        "config/SK_WAND_geom_config.json", temperature=0.0, deposit_leg_bound=True)
    dp, medium_path, qe_path = load_physics_config("config/SK_WAND_physics_config.json")
    lo, hi = qe_curve_bounds(qe_path)
    medium = make_medium("water", wavelength_grid=jnp.linspace(max(300.0, lo), min(700.0, hi), 200),
                         medium_model_path=medium_path)
    qe_fn = load_qe_curve(qe_path)
    reflection_fn, build_refl = get_reflection_model("scalar_mix")
    refl = build_refl(dp)
    key = jax.random.PRNGKey(19452)
    key, dkey, wkey = jax.random.split(key, 3)
    direction = jax.random.normal(dkey, (n, 3))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    u = jax.random.uniform(wkey, (n,))
    wavelengths = 1 / (1 / 275.0 - u * (1 / 275.0 - 1 / 674.0))
    oa = evaluate_optical_model(dp, wavelengths, medium, n, qe_fn=qe_fn)
    # This deliberately uses LUCiD's current spectrum treatment; it is an isolation
    # test of the cutoff, not validation of QE extrapolation or physical water tables.

    @jax.jit
    def run(positions, directions, key):
        state = (positions, directions, jnp.zeros(n), jnp.ones(n), key)

        def step(state, i):
            pos, dirs, times, alive, key = state
            key, unused_prop_key = jax.random.split(key)
            prop = geom.propagator(pos, dirs)
            hit_sensor = jnp.max(prop["inside_sensor"], axis=0)
            D = jnp.sqrt(jnp.sum((prop["positions"] - pos)**2, axis=1) + 1e-12) - 1e-6
            key, subkey = jax.random.split(key)
            rkeys = jax.random.split(subkey, n)
            next_pos, next_dir, next_time, detect, atten, cont, _, indirect = jax.vmap(
                lambda p, d, t, distance, normal, lr, lm, la, hs, rk: photon_iteration_sample(
                    p, d, t, distance, normal, lr, lm, dp.scattering.g, refl,
                    la, hs, 400.0, rk, geom.speed_of_light, reflection_fn=reflection_fn)
            )(pos, dirs, times, D, prop["normals"], oa.scatter_len, oa.mie_len,
              oa.abs_len, hit_sensor, rkeys)
            physical_deposit = prop["sensor_weights"] * (alive * detect * atten)[None, :]
            expected_pe = jnp.sum(physical_deposit * oa.qe[None, :])
            cont = jnp.where(geom.detector.bounds_check(next_pos), cont, 0.0)
            next_alive = alive * cont
            stats = jnp.array([jnp.sum(next_alive), expected_pe,
                               jnp.sum(alive * (1 - atten)),
                               jnp.sum(alive * cont * ~geom.detector.bounds_check(next_pos)),
                               jnp.sum(~jnp.isfinite(next_pos).all(axis=1) & (alive > 0)),
                               jnp.sum(physical_deposit)])
            return (next_pos, next_dir, next_time, next_alive, key), stats

        return jax.lax.scan(step, state, jnp.arange(K))[1]

    results = []
    for label, origin in (("center", (0.0, 0.0, 0.0)),
                          ("near_barrel", (float(geom.detector.r) - 0.5, 0.0, 0.0)),
                          ("near_cap", (0.0, 0.0, float(geom.detector.H) / 2 - 0.5))):
        stats = np.asarray(run(jnp.tile(jnp.asarray(origin), (n, 1)), direction, key))
        row = {"source": label, "origin_m": list(origin), "N": n,
               "radius_m": float(geom.detector.r), "height_m": float(geom.detector.H),
               "alive_after_each_step": stats[:, 0].tolist(),
               "expected_pe_each_step": stats[:, 1].tolist(),
               "incident_deposit_each_step": stats[:, 5].tolist(),
               "nan_alive_positions_each_step": stats[:, 4].tolist(),
               "lost_pe_fraction_K7_vs_K32": float(stats[7:, 1].sum() / stats[:, 1].sum()),
               "lost_pe_fraction_K12_vs_K32": float(stats[12:, 1].sum() / stats[:, 1].sum())}
        assert np.isfinite(stats).all()
        print(json.dumps(row), flush=True)
        results.append(row)
    return results


def main():
    print(json.dumps({"slurm_job_id": os.environ["SLURM_JOB_ID"],
                      "partition": os.environ["SLURM_JOB_PARTITION"], "jax": jax.__version__}), flush=True)
    oracle = one_step_oracle()
    print(json.dumps({"one_step_oracle": oracle}), flush=True)
    tail = sk_wand_tail()
    result = {"slurm_job_id": os.environ["SLURM_JOB_ID"],
              "partition": os.environ["SLURM_JOB_PARTITION"],
              "one_step_oracle": oracle, "sk_wand_tail": tail}
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
