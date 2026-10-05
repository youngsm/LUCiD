# SK_WAND PMT detection review

Reference checkout: `82f8d24` with `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json`.
Production code was not changed.
All numerical execution used milano CPU allocations with `uv run` and zero GPUs.
After the execution restriction was tightened, subsequent inspection and editing used milano shell allocation `39894029` on `sdfmilan272`.

## Confirmed production defect: incident QE is reduced again by sensor reflection

**Impact:** direct photoelectrons are approximately 25% too low for the selected `sensor_reflection_rate=0.25`.
This is distinct from the duplicated QE factor fixed by `82f8d24`.

Motta and Schonert define `QE(lambda,theta)=A(lambda,theta)*Pconv(lambda)`, where A is the incident photocathode absorption probability, alongside separate reflection and transmission probabilities.
Incident QE already includes reflection losses, so applying it conditional on nonreflection without renormalizing introduces another loss.
See [Eq. (1), pages 1-2](https://arxiv.org/pdf/physics/0408075).
Independently, WCSim's sensitive-detector-only QE mode actually applies `1/(1-0.25)` to tabulated QE after reflection has removed photons.
See [WCSimWCSD.cc:192-194 and 348-350](https://github.com/WCSim/WCSim/blob/develop/src/WCSimWCSD.cc#L192).
The exact experiment/digitization provenance of `SK_QE.json` has not been independently recovered; this is a normalization defect relative to configured incident QE, not certification of the curve against SK measurements.

Affected actual call path:

- `lucid/production/run_job.py:309` builds DATA with `K=12`, `temperature=0.0`, and `hit_mode='per_segment'`.
- `lucid/simulation/photon_step.py:97` samples reflection with probability R, leaving deposits only on the complement 1-R.
- `lucid/simulation/simulator.py:676` applies that nonreflection decision to sensor deposits.
- `lucid/simulation/simulator.py:824` supplies the per-wavelength incident QE.
- `lucid/simulation/sensor_response.py:278` Bernoulli-samples that QE without dividing by 1-R.

The result is `(1-R)*QE_incident`, rather than `QE_incident`.

### Executed evidence

Successful job: `39894066`, milano, 4 CPUs, 12 GB, zero GPUs.
Artifacts: [reproduce.py](reproduce.py), [results.json](results.json), [cpu-39894066.log](cpu-39894066.log).
JAX was `0.11.0`, NumPy was `2.4.3`, and `JAX_PLATFORMS=cpu` was set.
The installed CUDA plugin warned about version mismatch but was not used.

The actual boundary plus response test sends 300,000 photons into a sensor with negligible absorption/scattering.
QE at 400 nm is `0.2261907458`, while measured detection is `0.16977`, agreeing with the erroneous `(1-0.25)*QE=0.1696430594`.
The measured direct PE deficit is **24.94%**.

The end-user DATA interface test uses the actual 11,096-sensor SK_WAND geometry and medium/QE config.
A 50,000-photon 400 nm pencil beam starts 5 m before barrel PMT 4105, with emission times of 1 ns and segment ID 0.
It uses `setup_event_simulator(is_data=True)`, `per_segment`, `deposit_leg_bound=True`, and the same random seed at reflectance 0 and 0.25.
Prompt target-sensor hits arrive before 30 ns.

| Maximum steps | Direct PE, R=0 | Direct PE, R=0.25 | Direct PE loss |
| --- | ---: | ---: | ---: |
| 1 | 10,993.99 | 8,220.99 | 25.22% |
| 12 | 10,997.99 | 8,225.99 | 25.20% |

K=12 totals are 11,144.99 PE at R=0 and 10,296.99 PE at R=0.25 because reflection adds late hits.
Those totals are not a measurement of the correction's complete-event effect; direct hits isolate the defect.
The earlier job `39893957` incorrectly placed the source inside a PMT sphere and failed its E2E assertion.
Its log is retained, but it is not used as incident-photon evidence.

### Minimal test and suggested fix

`test_sensor_boundary_incident_qe()` in [reproduce.py](reproduce.py) is the executed isolated test.
The physical regression assertion below fails on HEAD with measured `0.16977` versus expected `0.2261907458`:

```python
assert abs(measured_direct_detection_probability - incident_qe) < 0.002
```

For the selected scalar reflection model, pass conditional QE to response sampling:

```python
qe_conditional = qe_incident / (1.0 - sensor_reflection_rate)
```

Keep reflection at probability R and validate `QE_incident + R <= 1`.
Include per-PMT efficiency in the incident probability before conditional normalization.
For angle-dependent reflection, carry each encounter's actual R through propagation rather than dividing by the configured scalar rate.
Use the same incident/conditional convention in expected-value transport.
An alternative categorical boundary draw uses `[R, QE_incident, 1-R-QE_incident]` for reflection, detection and loss.

[verify_conditional_fix.py](verify_conditional_fix.py) executes the proposed scalar correction through the production boundary/response functions without production edits.
Job `39894029` observed probability `0.22331` for 100,000 photons versus expected `0.2261907458`, about 2.18 binomial standard errors apart.
The conditional probability was `0.3015876710`.
See [fix-39894029.log](fix-39894029.log).

Exact main reproduction command, from the repository root:

```bash
sbatch audit/wc_data_review_20261004/pmt_detection/run_cpu.sbatch
```

Correction verification inside an approved CPU allocation, with the environment exports in `run_cpu.sbatch`:

```bash
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/pmt_detection/verify_conditional_fix.py
```

## Confirmed nondefault defect: negative TTS first arrival erases all charge

**Applicability:** `hit_mode='realistic'`, nonzero `response.tts`, and geometric arrivals within several TTS widths of time zero.
Selected SK_WAND production uses `per_segment` and loads `tts=0`, so this defect is dormant in the reference configuration.
The ski digitizer's later charge-dependent jitter is separate and does not pass through this mask.

`make_hits_data` smears photon times at `sensor_response.py:232`, finds their minimum at line 236, then requires that smeared minimum to exceed zero at line 239.
If any detected photon fluctuates across time zero, it erases all sensor charge, including other photoelectrons with positive measured times.
Timing fluctuations must not destroy accumulated charge without a configured acquisition gate.
The defect also breaks invariance under a harmless change of time origin.
WCSim's [PMT20inch::HitTimeSmearing](https://github.com/WCSim/WCSim/blob/develop/src/WCSimPMTObject.cc#L88) draws Gaussian timing fluctuations rather than an optical-photon rejection probability.

Job `39894066` independently reexecuted this previous lead with QE=1, geometric times 0.1 ns, TTS sigma 3 ns, and 20,000 sensors.

| PE per sensor | Input PE | Realistic output PE | Output after +100 ns | Per-segment output PE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20,000 | 10,244 | 20,000 | 20,000 |
| 10 | 200,000 | 240 | 200,000 | 200,000 |

At 10 PE per sensor, 99.88% of sensors lose all their charge.
The minimal suggested fix removes the positivity requirement on the smeared first-arrival time:

```python
nonzero_mask = (total_charge > 1e-10) & jnp.isfinite(detector_mins)
```

`test_negative_tts_keeps_photoelectrons()` in [reproduce.py](reproduce.py) is the executed isolated reproduction.
A corrected regression assertion is `assert q.sum() == n_sensors * occupancy` for both occupancies and either time origin.
Input-record validity can remain tied to valid unsmeared arrivals; a measured negative residual is a valid timestamp.

## Checked behavior and remaining physical assumptions

The previously duplicated wavelength QE factor is corrected at HEAD: the per-photon curve is normalized to the 400 nm reference before `response.qe` is applied.
Bernoulli detection per actual PhotonSim photon is appropriate conditional on the emitted photon list; per-photon Poisson draws would permit multiple idealized photoelectrons from one photon.
The selected `response.tts=0` and ski digitizer timing model do not smear timing twice by default.
`qe_corrections=1.0` expands to per-sensor ones and adds no efficiency factor in SK_WAND.
Transport has no photocathode-position or incidence-angle QE/collection-efficiency calibration.
Spherical PMT geometry and scalar reflectance are explicit simplifications, and they have not been classified here as implementation defects.
The exact SK_QE provenance, angular and immersion response remain calibration obligations; correcting normalization does not establish perfect agreement with real SK.
