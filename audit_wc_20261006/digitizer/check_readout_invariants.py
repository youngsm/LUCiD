"""Independent SK_WAND host-readout audit; run ONLY on milano/roma.

Loads production modules without importing unrelated simulation machinery.
These are invariant checks, not a claim of complete detector calibration.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('This audit must run inside a milano/roma Slurm allocation')
if os.environ.get('SLURM_JOB_PARTITION') not in ('milano', 'roma'):
    raise RuntimeError(f"Unapproved partition: {os.environ.get('SLURM_JOB_PARTITION')}")

ROOT = Path(__file__).resolve().parents[2]
def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

dig = load('audit_actual_digitizer', 'lucid/simulation/digitizer.py')
trig = load('audit_actual_trigger', 'lucid/simulation/trigger.py')
physics = json.loads((ROOT / 'config/SK_WAND_physics_config.json').read_text())
model = dig.resolve_model_config(physics['digitizer'])
cfg = trig.TriggerConfig.from_block(physics['trigger'])
rng = np.random.default_rng(61801)
print('partition', os.environ['SLURM_JOB_PARTITION'], 'job', os.environ['SLURM_JOB_ID'])
print('physics', physics['digitizer'], physics['trigger'])

# Independent scalar oracle: open a charge gate at each unconsumed photoelectron.
# Every positive detected input must belong to exactly one gate for SK_WAND's
# zero-deadtime preset; no splitting or creation of charge is allowed.
for trial in range(200):
    n = int(rng.integers(1, 1400))
    sensor = rng.integers(0, 47, n)
    times = rng.uniform(-100, 8000, n)
    q = np.ones(n)
    res = dig.digitize_event(sensor, times, q, 47, model)
    oracle = []
    for s in sorted(set(sensor)):
        ids = np.flatnonzero(sensor == s)
        ids = ids[np.argsort(times[ids])]
        while len(ids):
            first = times[ids[0]]
            members = ids[times[ids] <= first + 200.0]
            oracle.append((s, first, len(members)))
            ids = ids[times[ids] > first + 200.0]
    actual = sorted(zip(res.digit_sensor_idx, res.digit_time, res.digit_pe_true))
    np.testing.assert_allclose(actual, sorted(oracle))
    assert np.all(res.photon_digit_idx >= 0)
    assert res.digit_pe_true.sum() == n
    np.testing.assert_array_equal(res.digit_sensor_idx[res.photon_digit_idx], sensor)
print('PASS 200 independent scalar-oracle integration/conservation cases')

# Ghost residual weights with non-finite candidate times must not become one
# PE through the minimum-one SPE sampler, even at a separate PMT.
sd, hits, seg = dig.digitize_and_decompose(
    sensor_idx=np.array([0, 1]), charge=np.array([.999999, .000001013]),
    t_true=np.array([70., np.inf]), t_reco=np.array([70., np.inf]),
    particle_idx=np.array([0, 0]), segment_idx=np.array([0, 0]),
    emission_process=np.array([0, 0]), n_sensors=2, model=model,
    rng=np.random.default_rng(3), apply_resolution=False)
assert sd['sensor_idx'].tolist() == [0]
print('PASS nonfinite fractional ghost is dropped before SPE rounding')

# Independent trigger oracle: union every interval on which a trailing 200 ns
# window contains >=25 hits, then expand connected intervals by +/-300 ns.
# Continuous times avoid ambiguous equality at the TDC/window boundary.
for trial in range(100):
    t = np.sort(rng.uniform(-500, 5000, 450))
    edges = np.unique(np.r_[t, t + cfg.window_ns])
    intervals = []
    for left, right in zip(edges[:-1], edges[1:]):
        midpoint = (left + right) / 2
        count = np.sum((t > midpoint - cfg.window_ns) & (t <= midpoint))
        if count >= cfg.n_thr:
            if intervals and intervals[-1][1] == left:
                intervals[-1][1] = right
            else:
                intervals.append([left, right])
    expanded = []
    for left, right in intervals:
        left -= cfg.pad_before_ns
        right += cfg.pad_after_ns
        if expanded and left <= expanded[-1][1]:
            expanded[-1][1] = max(expanded[-1][1], right)
        else:
            expanded.append([left, right])
    expected = np.array(expanded).reshape(-1, 2)
    actual = trig.find_trigger_gates(t, cfg)
    np.testing.assert_allclose(actual, expected)
print('PASS 100 independent interval-oracle trigger cases')

# Real readout sequence: discriminator + dark + trigger must preserve every
# decomposition foreign key, correctly classify darkness, and keep canonical
# window/digit slices. Cover interleaved sources, multiple digits per sensor,
# and negative absolute times (ordinary WAND +/-250 ns t0 augmentation).
for trial in range(100):
    n = 1600
    s = rng.integers(0, 100, n)
    t = rng.choice([50., 3100., 6500.], n) + rng.normal(0., 8., n)
    sd, hs, ss = dig.digitize_and_decompose(
        sensor_idx=s, charge=np.ones(n), t_true=t, t_reco=t,
        particle_idx=rng.integers(0, 8, n), segment_idx=rng.integers(0, 45, n),
        emission_process=np.zeros(n, int), n_sensors=100, model=model, rng=rng,
        dark_rate_khz=model['dark_rate_khz'], readout_pad_ns=500.)
    for tab, keys in [(sd, ['T']), (hs, ['T','T_reco']), (ss, ['T','T_reco'])]:
        for key in keys:
            tab[key] -= 250.
    result = trig.apply_trigger(sd, hs, ss, cfg)
    assert result is not None
    ds, hs, ss, pw = result
    assert np.all(ds['PE'] >= model['threshold_pe'])
    assert np.any(ds['T'] < 0.)
    assert len(pw['window_start']) == 3
    for tab in (hs, ss):
        idx = tab['digit_idx']
        assert np.all((idx >= 0) & (idx < len(ds['PE'])))
        np.testing.assert_array_equal(tab['sensor_idx'], ds['sensor_idx'][idx])
    assert np.all(hs['particle_idx'][hs['emission_process'] == 2] == -1)
    assert np.all(ss['emission_process'] == 0)
    for a, b, lo, hi in zip(pw['window_start'], pw['window_end'],
                             pw['digit_offsets'][:-1], pw['digit_offsets'][1:]):
        assert np.all((ds['T'][lo:hi] >= a) & (ds['T'][lo:hi] <= b))
        assert np.all(np.diff(ds['sensor_idx'][lo:hi].astype(int)) >= 0)
print('PASS 100 production-order noise/discriminator/trigger/remapping cases')

# Check sampled timing against the configured charge law using independent
# charge bins, and print the actual accepted one-PE dark rate. Shape itself is
# a user-accepted approximation, so no assertion against a different detector.
q, t = dig.apply_readout_resolution(np.ones(1_000_000), np.zeros(1_000_000), model, rng)
for low, high in [(.25, .5), (1., 1.5), (2., 3.)]:
    keep = (q >= low) & (q < high)
    sigma = np.maximum(.58, .33 + np.sqrt(10 / np.maximum(q[keep], .5)))
    normalized_rms = np.std(t[keep] / sigma)
    assert abs(normalized_rms - 1) < .025
    print('charge_bin', low, high, 'time_rms_ns', t[keep].std(), 'normalized_rms', normalized_rms)
print('single_PE_acceptance', np.mean(q >= .25))
print('isolated_dark_output_khz', model['dark_rate_khz'] * np.mean(q >= .25))
print('PASS charge-conditioned timing, no extra upstream TTS in SK_WAND config')
print('DONE: all independent host readout invariants passed')
