"""Only audit-local monkeypatches; production source files are never edited."""
import json
import numpy as np
from repro_electronics import D, T, MODEL, TRIGGER, make_digits


def test_uncapping_restores_delayed_trigger_in_reference_readout():
    times = np.r_[np.full(80, 100.0), np.full(80, 200100.0)]
    current_sd, current_h, current_s = make_digits(
        times, n_sensors=11096, resolution=True, dark=4.2, seed=344)
    current = T.apply_trigger(current_sd, current_h, current_s, TRIGGER)
    cap = D._MAX_DIGIT_TIME_NS
    D._MAX_DIGIT_TIME_NS = np.inf
    restored_sd, restored_h, restored_s = make_digits(
        times, n_sensors=11096, resolution=True, dark=4.2, seed=344)
    D._MAX_DIGIT_TIME_NS = cap
    restored = T.apply_trigger(restored_sd, restored_h, restored_s, TRIGGER)
    assert not np.any(current[1]['particle_idx'] == 1)
    assert np.any(restored[1]['particle_idx'] == 1)
    return {
        'before_capture_pe': float(current[1]['PE'][current[1]['particle_idx'] == 1].sum()),
        'after_capture_pe': float(restored[1]['PE'][restored[1]['particle_idx'] == 1].sum()),
        'before_prompt_pe': float(current[1]['PE'][current[1]['particle_idx'] == 0].sum()),
        'after_prompt_pe': float(restored[1]['PE'][restored[1]['particle_idx'] == 0].sum()),
        'before_trigger_windows': np.column_stack([current[3]['window_start'], current[3]['window_end']]).tolist(),
        'after_trigger_windows': np.column_stack([restored[3]['window_start'], restored[3]['window_end']]).tolist(),
    }


def test_using_sampled_charge_restores_the_intended_timing_curve():
    n = 1200000
    rng = np.random.default_rng(461)
    q = D._sample_spe_charge(np.ones(n), MODEL['spe'], rng)
    corrected_t = D._sample_time_jitter(np.full(n, 100.0), np.maximum(q, 0.5), MODEL, rng)
    out = []
    for lower, upper in [(0.25, 0.5), (1.0, 1.5), (2.0, 3.0)]:
        m = (q >= lower) & (q < upper)
        oracle_rms = float(np.sqrt(np.mean((0.33 + np.sqrt(10 / np.maximum(q[m], 0.5))) ** 2)))
        actual_rms = float(corrected_t[m].std())
        assert abs(actual_rms - oracle_rms) < 0.04
        out.append({'charge_range_pe': [lower, upper], 'corrected_sigma_ns': actual_rms,
                    'primary_wcsim_sigma_ns': oracle_rms})
    return out


print(json.dumps({'uncapping_default_ski_dark_and_trigger': test_uncapping_restores_delayed_trigger_in_reference_readout(),
                  'sampled_charge_fix': test_using_sampled_charge_restores_the_intended_timing_curve()}, indent=2))
