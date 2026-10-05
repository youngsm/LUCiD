"""Standalone production DATA electronics repro; execute only under Slurm CPU."""
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def production_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

D = production_module('audit_digitizer', 'lucid/simulation/digitizer.py')
T = production_module('audit_trigger', 'lucid/simulation/trigger.py')
CFG = json.loads((ROOT / 'config/SK_WAND_physics_config.json').read_text())
MODEL = D.resolve_model_config(CFG['digitizer'])
TRIGGER = T.TriggerConfig.from_block(CFG['trigger'])


def make_digits(times, n_sensors=None, resolution=False, dark=0.0, seed=2):
    times = np.asarray(times)
    n = len(times)
    return D.digitize_and_decompose(
        sensor_idx=np.arange(n), charge=np.ones(n), t_true=times, t_reco=times,
        particle_idx=np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)],
        segment_idx=np.arange(n), emission_process=np.zeros(n, int),
        n_sensors=n if n_sensors is None else n_sensors,
        model=MODEL, rng=np.random.default_rng(seed), dark_rate_khz=dark,
        readout_pad_ns=TRIGGER.window_ns + max(TRIGGER.pad_before_ns, TRIGGER.pad_after_ns),
        apply_resolution=resolution)


def test_delayed_burst_deleted_before_trigger():
    times = np.r_[np.full(80, 100.0), np.full(80, 200100.0)]
    sd, hs, sh = make_digits(times)
    result = T.apply_trigger(sd, hs, sh, TRIGGER)
    before_cap = D.digitize_event(np.arange(160), times, np.ones(160), 160, MODEL)
    raw_gates = T.find_trigger_gates(before_cap.digit_time, TRIGGER)
    assert before_cap.n_digits == 160 and len(raw_gates) == 2
    assert len(sd['T']) == 80 and len(result[3]['window_start']) == 1
    assert np.all(result[0]['T'] < 1000.0)
    return dict(input_photoelectrons=160, uncapped_independent_trigger_windows=raw_gates.tolist(),
                production_photoelectrons=80, production_trigger_windows=np.column_stack(
                    [result[3]['window_start'], result[3]['window_end']]).tolist(),
                capture_mean_lifetime_ns=205000.0,
                fraction_captures_after_cap=float(np.exp(-1e5 / 205000.0)))


def test_readout_timing_uses_wrong_charge():
    n = 1200000
    q, t = D.apply_readout_resolution(np.ones(n), np.full(n, 100.0), MODEL,
                                     np.random.default_rng(142))
    bins = [(0.25, 0.50), (1.00, 1.50), (2.00, 3.00)]
    out = []
    for lower, upper in bins:
        selected = (q >= lower) & (q < upper)
        expected_sigma = np.maximum(0.58, 0.33 + np.sqrt(10.0 / np.maximum(q[selected], 0.5)))
        actual_sigma = float(t[selected].std())
        out.append(dict(charge_range_pe=[lower, upper], samples=int(selected.sum()),
                        measured_sigma_ns=actual_sigma,
                        primary_wcsim_sigma_ns=float(np.sqrt(np.mean(expected_sigma ** 2)))))
    assert abs(out[0]['measured_sigma_ns'] - out[2]['measured_sigma_ns']) < 0.05
    assert out[0]['primary_wcsim_sigma_ns'] > 4.7 and out[2]['primary_wcsim_sigma_ns'] < 2.5
    return out


def lut_single_pe(n, seed):
    source = (HERE / 'reference/WCSimPMTObject.cc').read_text()
    body = source.split('G4double* PMT20inch::Getqpe()', 1)[1].split('{', 2)[2].split('};', 1)[0]
    body = re.sub(r'//[^\n]*', '', body)
    cdf = np.array([float(v) for v in re.findall(r'\d+\.\d+', body)])
    assert len(cdf) == 501 and cdf[-2] == 1.0 and cdf[-1] == 0.0
    rng = np.random.default_rng(seed)
    indices = np.searchsorted(cdf[:-1], rng.random(n), side='left')
    return (indices - 50.0 + rng.random(n)) / 22.83


def wcsim_probability(charge):
    x = charge + 0.1
    polynomial = -0.06374 + x * (3.748 + x * (-63.23 + x * (452.0 + x * (-1449.0 + x *
        (2513.0 + x * (-2529.0 + x * (1472.0 + x * (-452.2 + x * (51.34 + x * 2.370)))))))))
    return np.where(x < 1.1, np.clip(polynomial, 0, 1), 1.0)


def measure_deliberate_response_approximation():
    n = 1200000
    raw = lut_single_pe(n, 442)
    reco = D._sample_spe_charge(np.ones(n), MODEL['spe'], np.random.default_rng(443))
    keep = D.apply_discriminator(reco, MODEL)
    probability = wcsim_probability(raw)
    return dict(
        statement='These are documented deliberate model approximations, not automatically new bugs.',
        wcsim_mean_raw_single_pe=float(raw.mean()), lucid_mean_raw_single_pe=float(reco.mean()),
        wcsim_sd_raw_single_pe=float(raw.std()), lucid_sd_raw_single_pe=float(reco.std()),
        wcsim_prob_charge_gt4=float((raw > 4.0).mean()), lucid_prob_charge_gt4=float((reco > 4.0).mean()),
        wcsim_prob_charge_gt3=float((raw > 3.0).mean()), lucid_prob_charge_gt3=float((reco > 3.0).mean()),
        wcsim_single_pe_acceptance=float(probability.mean()), lucid_single_pe_acceptance=float(keep.mean()),
        wcsim_mean_readout_per_pe=float(np.mean(raw * probability) * 0.985),
        lucid_mean_readout_per_pe=float(np.mean(reco * keep)),
        lucid_dark_rate_post_cut_khz=float(MODEL['dark_rate_khz'] * keep.mean()),
        wcsim_dark_rate_post_cut_khz=float(4.2 * 1.367 * probability.mean()))


def test_empty_deposits_no_dark():
    sd, hs, sh = make_digits([], n_sensors=11096, resolution=True, dark=4.2)
    assert len(sd['T']) == 0
    return dict(physics_deposits=0, production_dark_digits=0,
                status='Window is undefined for an empty event; missing free-running DAQ is a modeling limitation.')


if __name__ == '__main__':
    out = dict(configuration=str(ROOT / 'config/SK_WAND_physics_config.json'),
               delayed_burst_cap=test_delayed_burst_deleted_before_trigger(),
               count_instead_of_charge_timing=test_readout_timing_uses_wrong_charge(),
               deliberate_response_approximation=measure_deliberate_response_approximation(),
               empty_event_dark=test_empty_deposits_no_dark())
    print(json.dumps(out, indent=2))
