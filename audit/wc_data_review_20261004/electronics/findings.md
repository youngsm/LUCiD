# SK_WAND DATA electronics review

Reference LUCiD checkout: `82f8d24`.
Reference configs: `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json`.
Production call chain: `sources/event_generation.py:587` pools detected deposits, calls `digitize_and_decompose`, then applies `apply_trigger` at line 630; pooled pile-up/supernova calls the same digitizer at line 1296.
SK_WAND selects `ski` with a 200 ns integration window, zero configured deadtime, a 0.25 pe discriminator, 4.2 kHz input dark rate, and a 200 ns / 25-hit trigger with 300 ns pre/post padding.
The production digitizer routines were loaded from the actual source file, rather than copied into a separate reference implementation.
No production source files were edited.

## Executed evidence and reproduction

`repro_electronics.py` ran as Slurm job `39894018` on the `milano` partition, using 4 CPUs and zero GPUs.
The complete result is in `job_39894018.log`.
The job script is `run_cpu.sbatch`, submitted with `sbatch audit/wc_data_review_20261004/electronics/run_cpu.sbatch`.
`check_minimal_fixes.py` ran with `uv run` inside milano allocation `39894030` on `sdfmilan269`, using 2 CPUs and zero GPUs.
The result is in `minimal_fix_39894030.log`.
After the user tightened the execution rule, all further file inspection and edits also happened inside allocation `39894030`.

The exact Python command, run within either approved allocation, was:

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/electronics/repro_electronics.py
```

Replace the final filename with `check_minimal_fixes.py` for the proposed-fix check.

## Confirmed defect E1: every detected deposit after 100 us is deleted before electronics and trigger

Affected code: `lucid/simulation/digitizer.py:109` fixes `_MAX_DIGIT_TIME_NS = 1e5`; line 518 filters deposits by `(t_reco - ref) < _MAX_DIGIT_TIME_NS` before digitization at line 539.
This affects the selected SK_WAND production config regardless of its trigger settings.
For ordinary single-vertex events the reference is the earliest detected deposit; pooled pile-up and supernova pass a per-interaction `t_ref`.

The executed end-user readout-stage reproducer supplied 80 detected prompt photoelectrons at 100 ns and 80 detected delayed photoelectrons at 200100 ns, on different PMTs.
The configured production trigger, applied directly to the otherwise identical uncapped digit list, finds two gates: `[-200,600]` ns and `[199800,200600]` ns.
The production `digitize_and_decompose` call deletes all 80 delayed photoelectrons, leaves only 80 prompt photoelectrons, and therefore yields one gate.
This defect precedes any independently chosen DAQ acceptance window, so even a delayed burst satisfying the configured 25-hit trigger is unavailable to that trigger.

The second executed check used all 11096 PMTs, the selected ski SPE response, the 4.2 kHz dark noise, and the actual selected trigger.
The unmodified digitizer retains zero delayed photoelectrons and one trigger gate.
Changing only the audit-imported `_MAX_DIGIT_TIME_NS` constant to infinity restores 58 delayed physics photoelectrons after the discriminator and a second trigger gate.
The changed RNG consumption means the prompt discriminator draws are different, so the 59 versus 62 prompt photoelectrons in this comparison do not measure a physical effect.
The missing delayed burst and its restoration are the relevant result.

Physical relevance: delayed neutron-capture light is a measured detector signal, not an unphysical late numerical tail.
SK neutron data have a measured hydrogen-capture lifetime of `203.7 +/- 2.8 us`, and SK-IV deliberately records delayed coincidences through 535 us after a prompt trigger in [the primary SK paper, sections 2-4](https://arxiv.org/abs/1311.3738).
Using a representative 205 us exponential lifetime, a 100 us cutoff excludes `exp(-100/205) = 61.4%` of capture times in events with prompt light, even before DAQ selection.
This percentage refers to the timing fraction, not a measured final neutron detection efficiency.
An isolated late capture with no earlier detected prompt light uses its own earliest deposit as the single-vertex reference and can avoid this particular cutoff, so the behavior also depends on whether prompt light was detected.

Smallest useful reproduction, using the selected model and actual production function:

```python
import numpy as np
from lucid.simulation.digitizer import digitize_and_decompose, resolve_model_config

t = np.array([100., 200100.])
sd, hits, seg = digitize_and_decompose(
    sensor_idx=np.array([0, 1]), charge=np.ones(2), t_true=t, t_reco=t,
    particle_idx=np.array([0, 1]), segment_idx=np.array([0, 1]),
    emission_process=np.zeros(2, int), n_sensors=2,
    model=resolve_model_config('ski'), rng=np.random.default_rng(2),
    apply_resolution=False)
assert sd['sensor_idx'].tolist() == [0]  # Detected delayed photon is deleted.
```

Minimal fix suggestion: make the late-light cutoff an explicit detector/DAQ configuration field and avoid deleting finite detected deposits before trigger selection.
For a physics-complete readout, bound dark generation using requested DAQ gates or separated bounded chunks rather than deleting signal photons to bound the dark span.
Simply setting this constant to infinity reproduces the signal restoration but can allocate excessive dark noise for long-lived radioactive tails, so that one-line change alone should not be shipped without bounded dark windows.
SK-IV neutron tagging additionally needs the deliberate forced after-trigger gate described below; removing the cutoff alone does not provide that feature.

## Confirmed defect E2: charge-dependent PMT timing uses true PE count instead of sampled analogue charge

Affected code: `lucid/simulation/digitizer.py:398` samples the analogue charge into `pe_reco`, but line 402 calls `_sample_time_jitter(digit_time, pe_true, model, rng)`.
The ski formula at lines 345-349 therefore has the same width for every one-photoelectron digit, irrespective of its measured pulse charge.
This is active in the SK_WAND production path when `apply_smearing=True`.
The selected detector response has `tts=0`, so this is not an additional upstream per-photon TTS smear in the selected configuration.

Independent implementation expectation: [pinned WCSim `WCSimWCDigitizer.cc:343-357`](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimWCDigitizer.cc#L343) computes `Q = max(peSmeared,0.5)` after SPE charge sampling and threshold acceptance, and passes that Q into `PMT->HitTimeSmearing`.
[The PMT20inch response formula](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimPMTObject.cc#L91) then uses `max(0.58,0.33+sqrt(10/Q))` ns.
The charge dependence has independent physical support: [SK calibration, section 3.1.8 and figures 23-24](https://arxiv.org/abs/1307.0162) fits timing distributions separately in measured charge bins.
The real SK timing response is asymmetric, so the WCSim Gaussian formula is an implementation oracle for the intended LUCiD model and cannot by itself certify an exact real-detector timing model.

Executed result for 1200000 one-photoelectron digit samples:

| Measured charge bin | LUCiD measured timing RMS | Intended WCSim charge-conditioned RMS |
| --- | ---: | ---: |
| 0.25-0.50 pe | 3.483 ns | 4.802 ns |
| 1.00-1.50 pe | 3.495 ns | 3.174 ns |
| 2.00-3.00 pe | 3.503 ns | 2.421 ns |

The low-charge timing width is underestimated by about 27.5%, while the high-charge single-photoelectron width is overestimated by about 44.7%.
This distorts charge-time correlations used by timing reconstruction and calibration; it does not directly remove photons.
This is a moderate response bug rather than a detector-wide factor-of-four photon-yield bug.

Smallest useful statistical reproduction:

```python
import numpy as np
from lucid.simulation.digitizer import apply_readout_resolution, resolve_model_config
q, t = apply_readout_resolution(np.ones(300000), np.zeros(300000),
    resolve_model_config('ski'), np.random.default_rng(142))
low, high = (q >= .25) & (q < .5), (q >= 2) & (q < 3)
assert abs(t[low].std() - t[high].std()) < .15  # Both use count=1.
# Intended widths differ: 4.802 ns for low versus about 2.42 ns for high.
```

Minimal fix suggestion: pass sampled analogue charge to `_sample_time_jitter`, using a 0.5 pe floor for ski/hk, instead of passing the true PE count.
Keep the legacy model behavior separate if backward compatibility requires it.
The executed audit-only proposed-fix check produced 4.803 ns, 3.174 ns, and 2.426 ns in the same three bins, consistent with the independent expectation.
Matching the complete WCSim electronics pipeline would additionally apply its discriminator noise and 0.985 charge efficiency in the documented order, but those are separate explicit model choices here.

## Deliberate approximations and validation gaps, not new confirmed implementation bugs

The code explicitly chooses a fitted Gaussian-plus-exponential SPE response and a sharp discriminator rather than WCSim's tabulated SPE spectrum and probabilistic discriminator.
The executed comparison independently parsed the PMT20inch CDF from upstream source, excluded the final dummy element, and used the actual [`rn1pe` sampling rule](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimWCPMT.cc#L55).
It also evaluated the actual threshold polynomial in [WCSimWCDigitizer.hh:102](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/include/WCSimWCDigitizer.hh#L102), rather than relying on comments about a 50% point.
The current fitted SK SPE RMS is `0.748 pe` versus `1.030 pe` for the referenced LUT, about 27% narrower.
The probability of a one-photoelectron charge above 4 pe is `9.2e-6` versus `0.01345`, about 1467 times smaller in this sample.
The one-photoelectron discriminator acceptance is `0.7682` versus `0.7391`; the charge expectation per generated PE after acceptance is `0.9805 pe` versus `0.9432 pe` for WCSim including its 0.985 efficiency.
These differences undermine a claim of exact physical charge-response fidelity, even though they are declared approximations rather than accidental arithmetic defects.
If that approximation is unacceptable, the smallest reliable change is to sample and sum the existing calibrated CDF directly, rather than fit its central shape while losing the broad charge tail.

SK_WAND has only an independently firing multiplicity trigger and lacks the SK-IV forced after-trigger readout used to retain 2.2 MeV neutron-capture light.
A low-multiplicity delayed capture can therefore be excluded even before 100 us, and removing the late-light cap does not restore it.
This is a detector-era/DAQ-model limitation, because the selected electronics preset is explicitly named `ski`; it should not be silently treated as a claimed implementation of SK-IV neutron DAQ.
If modern SK neutron tagging is part of the intended physical target, add an explicit SHE/AFT readout model with its calibrated prompt threshold and delayed gate rather than lower the ordinary multiplicity trigger.

Dark noise is sampled only across an event's detected-deposit span plus padding; an empty physics deposit list has zero dark digits.
The selected pipeline therefore represents event-conditioned readouts rather than a full free-running detector stream including independent dark-only triggers.
The reference SK configuration has no measured per-PMT gain/QE/timing tables and no asymmetric calibrated timing distribution, so these pieces cannot be certified as exactly matching real SK.
The sharp-discriminator model's simulated output dark rate was `3.226 kHz/PMT`; the selected WCSim implementation, including its actual 1.367 conversion factor, gives `4.243 kHz/PMT` in the same SPE calculation.
That WCSim comparison does not by itself establish which physical SK period/calibrated dark rate the intended detector should reproduce, so it is not presented as an independently proven physical dark-rate bug.

## Code-level checks without evidence of a large default defect

The sliding per-PMT integration algorithm sums detected charges within an inclusive 200 ns window and vetoes the configured subsequent deadtime; the selected deadtime is zero.
The fractional transport weight `0.999999` near full overlap is rounded with `np.rint` before SPE multiplicity sampling at line 320, so it does not become zero through integer truncation.
The default SK_WAND response has no upstream per-photon TTS, so its digitizer jitter is applied once.
Digit and trigger times remain float64 through the inspected host readout stages.
The `sensor_digits` writer path at `lucid/sources/writer.py:510-518` writes negative digit times without a positive-time filter.
The decomposition discriminator and trigger remaps retain foreign keys into the surviving digit list in the executed production-stage examples.
The WCSim TDC uses truncation toward zero, while LUCiD uses rounding at digitizer.py:360; this is a bounded sub-0.4 ns convention discrepancy and not a large photon-yield defect.
These checks establish the inspected behavior, not a claim that every possible electronics edge case or real-detector calibration has been exhaustively validated.
