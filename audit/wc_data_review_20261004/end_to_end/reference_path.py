"""Execute with uv inside a milano/roma allocation; no production edits."""
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import ParticleParams, load_detector_params
from lucid.geometry import generate_detector
from lucid.sources.event_builder import _trace_event_bucketed
from lucid.simulation import setup_event_simulator

OUT = Path(__file__).resolve().parent
GEOM = "config/SK_WAND_geom_config.json"
PHYS = "config/SK_WAND_physics_config.json"
results = {}


def say(name, value):
    print(name, json.dumps(value), flush=True)
    results[name] = value
    (OUT / "reference_results.json").write_text(json.dumps(results, indent=2))


def oracle(origins, directions, centers, radius, cylinder_radius, height):
    """Independent float64 positive sphere-entry/cylinder-exit intersection."""
    ids = []
    ds = []
    for lo in range(0, len(origins), 128):
        o = origins[lo:lo + 128].astype(np.float64)
        d = directions[lo:lo + 128].astype(np.float64)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        oc = o[:, None, :] - centers[None, :, :]
        b = np.sum(oc * d[:, None, :], axis=-1)
        discr = b * b - np.sum(oc * oc, axis=-1) + radius * radius
        entry = -b - np.sqrt(np.maximum(discr, 0))
        exit_ = -b + np.sqrt(np.maximum(discr, 0))
        entry = np.where(entry > 0, entry, exit_)
        a_wall = np.sum(d[:, :2] ** 2, axis=1)
        b_wall = np.sum(o[:, :2] * d[:, :2], axis=1)
        c_wall = np.sum(o[:, :2] ** 2, axis=1) - cylinder_radius ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            wall = (-b_wall + np.sqrt(b_wall ** 2 - a_wall * c_wall)) / a_wall
            cap = np.where(d[:, 2] > 0, height / 2 - o[:, 2], -height / 2 - o[:, 2]) / d[:, 2]
        wall = np.where(a_wall > 0, wall, np.inf)
        cap = np.where(d[:, 2] != 0, cap, np.inf)
        leg = np.minimum(wall, cap)
        ts = np.where((discr > 0) & (entry > 0) & (entry <= leg[:, None]), entry, np.inf)
        idx = np.argmin(ts, axis=1)
        dist = np.min(ts, axis=1)
        ids.extend(np.where(np.isfinite(dist), idx, -1).tolist())
        ds.extend(dist.tolist())
    return np.asarray(ids), np.asarray(ds)


def trace(sim, origins, directions, times, wavelengths, labels, buckets, seed=41):
    return _trace_event_bucketed(
        sim, origins, directions, times, wavelengths, labels,
        len(centers), buckets, jax.random.PRNGKey(seed))


start = time.monotonic()
det = generate_detector(GEOM)
centers = np.asarray(det.all_points)
r = det.S_radius
say("geometry", {"sensors": len(centers), "radius_m": det.r, "height_m": det.H,
                 "pmt_radius_m": r})
dp = load_detector_params(PHYS, num_sensors=len(centers))
dp_unit = dp._replace(
    scattering=dp.scattering._replace(scatter_length=jnp.asarray(1e20), mie_scatter_length=jnp.asarray(1e20)),
    absorption=dp.absorption._replace(absorption_length=jnp.asarray(1e20)),
    reflection=dp.reflection._replace(sensor_reflection_rate=jnp.asarray(0.), wall_reflection_rate=jnp.asarray(0.)),
    response=dp.response._replace(qe=jnp.asarray(1.), tts=jnp.asarray(0.)))
sim = setup_event_simulator(GEOM, 0, K=12, is_data=True, temperature=0.0,
                            physics_config=PHYS, default_detector_params=dp_unit,
                            hit_mode="per_segment", deposit_leg_bound=True,
                            wavelength_mode=False)
pick = np.linspace(0, len(centers) - 1, 65, dtype=int)
origins = np.repeat(np.asarray([[5., -2., 3.]], np.float32), len(pick), axis=0)
directions = (centers[pick] - origins).astype(np.float32)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
times = np.full(len(pick), 7., np.float32)
wavelengths = np.full(len(pick), 400., np.float32)
labels = np.arange(len(pick), dtype=np.int32)
oi, od = oracle(origins, directions, centers, r, det.r, det.H)
a = trace(sim, origins, directions, times, wavelengths, labels, (65,))
positive = a[3] > 1e-5
gid = a[8][positive]
expected_times = times[gid] + od[gid] / (0.299792 / 1.33)
say("deterministic_meter_boundary", {
    "photons": len(pick), "sum_pe": float(a[0].sum()), "positive_records": int(positive.sum()),
    "wrong_sensor_records": int(np.sum(a[6][positive] != oi[gid])),
    "max_time_error_ns": float(np.max(np.abs(a[4][positive] - expected_times))),
    "sample_pe_weight": float(a[3][positive][0]),
    "input_origin_m": origins[0].tolist()})
assert positive.sum() == len(pick)
assert np.all(a[6][positive] == oi[gid])
assert np.max(np.abs(a[4][positive] - expected_times)) < 0.002
controls = {}
for name, lab, bucket in [("padding", labels, (256,)), ("chunking", labels, (16, 32)),
                          ("irrelevant_segment_labels", labels + 999, (65,))]:
    b = trace(sim, origins, directions, times, wavelengths, lab, bucket)
    controls[name] = {"charge_equal": bool(np.array_equal(a[0], b[0])),
                      "time_equal": bool(np.array_equal(a[1], b[1])),
                      "sum_pe": float(b[0].sum()),
                      "positive_records": int(np.sum(b[3] > 1e-5))}
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
say("deterministic_controls", controls)
# Independent ray census exposes geometry losses without a stochastic optics confound.
rng = np.random.default_rng(85)
n = 8192
origins = np.repeat(np.asarray([[0., 0., 0.]], np.float32), n, axis=0)
directions = rng.normal(size=(n, 3)).astype(np.float32)
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
times = np.full(n, 7., np.float32)
wavelengths = np.full(n, 400., np.float32)
labels = np.zeros(n, np.int32)
oi, od = oracle(origins, directions, centers, r, det.r, det.H)
a = trace(sim, origins, directions, times, wavelengths, labels, (8192,))
hit = a[3] > 1e-5
observed_ids = np.full(n, -1, np.int32)
observed_ids[a[8][hit]] = a[6][hit]
say("isotropic_geometry_oracle", {"photons": n, "oracle_hits": int(np.sum(oi >= 0)),
                                  "observed_hits": int(np.sum(observed_ids >= 0)),
                                  "false_negatives": int(np.sum((oi >= 0) & (observed_ids < 0))),
                                  "false_positives": int(np.sum((oi < 0) & (observed_ids >= 0))),
                                  "wrong_sensor": int(np.sum((oi >= 0) & (observed_ids >= 0) & (oi != observed_ids)))})
np.savez(OUT / "oracle_mismatches.npz", origins=origins, directions=directions,
         oracle_sensor=oi, observed_sensor=observed_ids, oracle_distances=od)

sim_phys = setup_event_simulator(GEOM, 0, K=1, is_data=True, temperature=0.0,
                                 physics_config=PHYS, default_detector_params=True,
                                 hit_mode="per_segment", deposit_leg_bound=True)
n = 32768
target = int(pick[30])
v = centers[target] / np.linalg.norm(centers[target])
origins = np.repeat((centers[target] - v).astype(np.float32)[None, :], n, axis=0)
directions = np.repeat(v.astype(np.float32)[None, :], n, axis=0)
times = np.full(n, 7., np.float32)
wavelengths = np.full(n, 400., np.float32)
labels = np.zeros(n, np.int32)
a = trace(sim_phys, origins, directions, times, wavelengths, labels, (8192,))
# Direct expectation: no-scatter/no-absorption survival x nonreflection x exactly one QE.
with open("config/materials/water.json") as f:
    medium = json.load(f)
with open("config/pmt/SK_QE.json") as f:
    qe = json.load(f)
lam = 400.
sc, mi, ab = medium["scattering"]["symmetric"], medium["scattering"]["asymmetric"], medium["absorption"]
alpha_s = sc["P4"] / lam ** 4 * (1 + sc["P5"] / lam ** 2)
alpha_m = mi["P6"] * (1 + mi["P7"] / lam ** 4 * (lam - mi["P8"]) ** 2)
alpha_a = ab["P0"] * ab["P1"] / lam ** 4 + ab["P0"] * ab["P2"] * (lam / 500.) ** ab["P3"]
q = np.interp(lam, qe["wavelengths_nm"], np.asarray(qe["qe_percent"]) / 100.)
d = 1. - r
p = np.exp(-d * (alpha_s + alpha_m + alpha_a)) * 0.75 * q
observed = float(a[0].sum())
sigma = np.sqrt(n * p * (1 - p))
say("actual_direct_optics", {"photons": n, "target_sensor": target, "travel_m": d,
                             "qe_at_400": float(q), "expected_single_qe_pe": float(n * p),
                             "expected_double_qe_pe": float(n * p * q), "observed_pe": observed,
                             "z_score_single_qe": float((observed - n * p) / sigma)})
assert abs(observed - n * p) < 5 * sigma
say("elapsed_s", time.monotonic() - start)
