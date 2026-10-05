"""Run only within an authorized milano/roma Slurm CPU allocation."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "audit_production_digitizer", ROOT / "lucid/simulation/digitizer.py")
D = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D
SPEC.loader.exec_module(D)
TSPEC = importlib.util.spec_from_file_location(
    "audit_production_trigger", ROOT / "lucid/simulation/trigger.py")
TR = importlib.util.module_from_spec(TSPEC)
sys.modules[TSPEC.name] = TR
TSPEC.loader.exec_module(TR)


def test_capture_light_is_removed_before_trigger():
    # A prompt PE and a detected PE from a neutron capture 200 us later.
    times = np.array([100.0, 200_100.0])
    sd, hits, seg = D.digitize_and_decompose(
        sensor_idx=np.array([0, 1]), charge=np.ones(2),
        t_true=times, t_reco=times, particle_idx=np.array([0, 1]),
        segment_idx=np.array([0, 1]), emission_process=np.zeros(2, int),
        n_sensors=2, model=D.resolve_model_config("ski"),
        rng=np.random.default_rng(0), apply_resolution=False)
    pure_windowing = D.digitize_event(np.array([0, 1]), times, np.ones(2),
                                    2, D.resolve_model_config("ski"))
    assert pure_windowing.n_digits == 2
    assert sd["sensor_idx"].tolist() == [0]
    assert hits["particle_idx"].tolist() == [0]
    assert seg["segment_idx"].tolist() == [0]
    tau_ns = 205_000.0
    return {
        "input_pe": 2, "windowing_only_pe": float(pure_windowing.digit_pe_true.sum()),
        "production_digits": sd["sensor_idx"].tolist(),
        "production_pe": float(sd["PE"].sum()),
        "capture_retention_100us_exp_tau205us": float(1 - np.exp(-100_000 / tau_ns)),
        "capture_retention_535us_exp_tau205us": float(1 - np.exp(-535_000 / tau_ns)),
    }


def test_default_sk_configuration_response():
    samples = 300_000
    out = {}
    for name in ("SK_like", "SK", "SK_WAND"):
        cfg = json.loads((ROOT / f"config/{name}_physics_config.json").read_text())
        model = D.resolve_model_config(cfg.get("digitizer"))
        q, t = D.apply_readout_resolution(
            np.ones(samples), np.full(samples, 100.0), model,
            np.random.default_rng(42))
        out[name] = {
            "model": model["model"], "charge_mean_pe": float(q.mean()),
            "charge_sigma_pe": float(q.std()), "time_sigma_ns": float(t.std()),
            "fraction_below_0p25pe": float((q < 0.25).mean()),
            "dark_rate_khz": float(model["dark_rate_khz"]),
            "integration_window_ns": model["integration_window_ns"],
        }
    assert out["SK_like"]["time_sigma_ns"] == 0
    assert 0.0119 < out["SK_like"]["charge_sigma_pe"] < 0.0121
    assert out["SK_WAND"]["charge_sigma_pe"] > 0.6
    return out


def test_neutron_capture_has_no_after_trigger_readout():
    cfg = json.loads((ROOT / "config/SK_WAND_physics_config.json").read_text())
    times = np.r_[np.full(80, 100.0), np.full(7, 200_100.0)]
    sensors = np.arange(times.size)
    r = D.digitize_event(sensors, times, np.ones(times.size), times.size,
                         D.resolve_model_config(cfg["digitizer"]))
    gates = TR.find_trigger_gates(r.digit_time, TR.TriggerConfig.from_block(cfg["trigger"]))
    kept = TR.hits_in_gates(r.digit_time, gates)
    assert kept[:80].all()
    assert not kept[80:].any()
    return {
        "input_prompt_detected_pe": 80, "input_capture_detected_pe": 7,
        "gates_ns": gates.tolist(), "capture_detected_pe_kept": int(kept[80:].sum()),
        "source_skiv_she_threshold": 58, "source_skiv_aft_end_ns": 535_000,
    }


def test_sk_timing_depends_on_count_instead_of_measured_charge():
    q, t = D.apply_readout_resolution(
        np.ones(600_000), np.full(600_000, 100.0),
        D.resolve_model_config("ski"), np.random.default_rng(123))
    low = (q >= 0.25) & (q < 0.5)
    high = (q >= 2.0) & (q < 3.0)
    expected_sigma = np.maximum(0.58, 0.33 + np.sqrt(10.0 / np.maximum(q, 0.5)))
    out = {}
    for label, mask in (("low_charge_0p25_to_0p5", low), ("high_charge_2_to_3", high)):
        out[label] = {
            "n": int(mask.sum()), "measured_sigma_ns": float(t[mask].std()),
            "wcsim_charge_dependent_rms_ns": float(np.sqrt(np.mean(expected_sigma[mask] ** 2))),
        }
    assert abs(out["low_charge_0p25_to_0p5"]["measured_sigma_ns"] -
               out["high_charge_2_to_3"]["measured_sigma_ns"]) < 0.1
    return out


if __name__ == "__main__":
    print(json.dumps({
        "capture_cap": test_capture_light_is_removed_before_trigger(),
        "sk_defaults": test_default_sk_configuration_response(),
        "missing_after_trigger": test_neutron_capture_has_no_after_trigger_readout(),
        "timing_charge_correlation": test_sk_timing_depends_on_count_instead_of_measured_charge(),
    }, indent=2))
