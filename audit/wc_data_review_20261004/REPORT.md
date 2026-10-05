# SK WAND water Cherenkov data audit

The audit reproduced substantial errors in PMT efficiency, first-sensor selection, detector boundaries, delayed light, pion trajectories, and GENIE angular distributions.
Minimal tests, measured results, and suggested fixes are linked below.
Production code has not been changed.

Reference: LUCiD `82f8d249fd3fcbff1a26dd8a94465dc07e5f3839`, with `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json`.
The actual geometry has 11,096 PMTs, radius 16.962280 m, height 36.378422 m, and spherical PMT radius 0.254 m.
Standard production uses hard overlap, `per_segment`, `K=12`, SK-I digitization, and the configured multiplicity trigger.
Library DATA defaults differ, so scope matters.

The source review used PhotonSim `v1.0.0`, commit `ff73224a669fdc3598dc2993a8763223b11a3e5b`, pinned by the container workflow, with Geant4 11.3.0.
The audit baseline changed only its GUI-dependent entry point for batch execution; physics, materials, stepping, generation, and output code were unchanged.
An unseen custom host binary needs its own version check.
Every numerical test used `uv run` on milano CPU allocations; zero GPUs were used.
After the user's clarification, inspection and editing also ran in allocations.
Twelve component reviewers participated, including GPT 6 Astra for independent photocathode and material checks.

## Substantial physical defects

| Defect | Measured consequence | Scope | Minimal fix |
| --- | --- | --- | --- |
| Incident QE reduced again by reflection | About 25% fewer direct PE at R=0.25 | Configured PMT response | Conditional QE=incident QE/(1-R), or categorical boundary outcomes |
| G4 secondary creation cutoff at 10 us | Identical 2.2 MeV gamma at 200 us: 6,716 photons become 454, a 93.24% loss | Delayed G4 light and captures | Remove source cut; apply explicit readout acceptance downstream |
| Thermal-neutron elastic model is unsuitable | Capture tail beyond 1 ms is 5.92% versus about 0.75% expected; mean 289.9 versus 204.4 us | Neutron physics in water, visible after removing source cut | Replace elastic model with validated thermal HPT scattering; retain the tested capture model |
| Incomplete first-PMT lookup | Wrong PMT and 77.62 ns extra delay in minimal ray | Especially near detector boundaries | Complete travelled-segment intersection |
| Exterior photons create ID hits | 100,000 exterior photons produce 22,214 false PE | Exiting translated tracks/showers | Bounds mask after source transforms |
| Pion replacement rewinds trajectory | Creation vertex up to 3.352 m behind completed endpoint | Charged-pion deflections | Current post-step position and time |
| GENIE isotropization is biased | Forward hemisphere 70.395% instead of 50% | Isotropized GENIE samples | Uniform SO(3) rotation |
| Detected light deleted after 100 us | Two triggerable bursts become one | Selected electronics with delayed light | Explicit DAQ acceptance and bounded dark windows |

Percentages describe the named experiments, not aggregate bias in every WAND event.
The QE result concerns direct light; reflection can add late hits elsewhere.
Exterior-beam response is not a cohort contamination percentage.
The two delayed cuts are distinct stages and their losses must not be added.

## Reproduction setup

Run the examples inside a milano or roma allocation, from the repository root:

```bash
export JAX_PLATFORMS=cpu
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
uv run --offline --no-project \
  --python audit/wc_data_20261004/env_system/bin/python python REPRODUCER.py
```

Minimal regressions intentionally fail on the audited checkout.
Larger measurement scripts may assert the observed defect and save its result.
Each lane report supplies exact commands, allocation IDs, logs, and correction checks.

## PMT incident efficiency

[Evidence](pmt_detection/findings.md), [boundary and actual DATA tests](pmt_detection/reproduce.py), [isolated correction](pmt_detection/verify_conditional_fix.py).
Affected code: `photon_step.py:97`, `simulator.py:676`, `sensor_response.py:278`.
Boundary detection is 0.16977 rather than incident QE 0.22619075.
An actual SK_WAND beam records 10,997.99 prompt PE at R=0 versus 8,225.99 at R=0.25.

```python
from audit.wc_data_review_20261004.pmt_detection.reproduce import test_sensor_boundary_incident_qe
r = test_sensor_boundary_incident_qe()
assert abs(r['measured_direct_detection_probability'] - r['required_incident_probability']) < .002
```

Incident QE already includes reflection: QE=A*Pconv, with A the incident absorption probability.
See [Motta and Schonert Eq. 1](https://arxiv.org/pdf/physics/0408075) and [independent review](astra_physics/findings.md).
For scalar reflection, use `qe_conditional = qe_incident / (1-R)` and validate `qe_incident + R <= 1`.
Angular models must carry the actual encounter reflectance.
The correction recovers incident QE within sampling uncertainty.
Absolute calibration of the supplied QE curve remains unresolved.

## Delayed photon production

[Source evidence and fixes](geant4_physics/findings.md), [minimal tests](geant4_physics/test_source_physics.py), [comparison results](geant4_physics/comparison_results.json).
PhotonSim `src/SteppingAction.cc:102` kills tracks created after 10 us, after their first step.
Some first-step light survives while most of a late charged track is lost.
An identical event shifted in time must retain its pre-readout photon yield.

```python
from audit.wc_data_review_20261004.geant4_physics.test_source_physics import counts
assert counts('gamma_delayed', '').sum() >= .95 * counts('gamma_prompt', '').sum()
```

This fails at 454 versus 6,716 photons.
Removing the cut in an isolated copy restores exactly 6,716 delayed photons with the matched seed.
For 100 actual 1 MeV neutron events, baseline versus uncut output has 1,962 versus 21,142 photons, a 90.72% reduction; registered capture gammas are 3 versus 100.
These are source yields, not final neutron detection efficiencies.

## First PMT selection

[Geometry evidence](geometry/findings.md), [minimal production DATA regression](end_to_end/tangent_data_reproducer.py), [complete-selection prototype](geometry/reproduce_selection_fix.py).
The candidate lookup uses the eventual envelope endpoint rather than the whole photon segment.
This ray begins inside water and outside every PMT sphere:

```python
import jax.numpy as jnp
from lucid.geometry.detector_geometry import DetectorGeometry
g = DetectorGeometry.from_config('config/SK_WAND_geom_config.json', temperature=0., deposit_leg_bound=True)
r = g.propagator(jnp.array([[16.8585866627, -.3529484099, -.4]]), jnp.array([[0., 0., 1.]]))
assert 4105 in r['sensor_indices'][:, 0]
```

It should hit PMT 4105 at 0.166513 m but selects PMT 458 at 17.662575 m.
A separate production DATA test measures 7.738719 ns expected versus 85.358406 ns observed.
The independent float64 oracle checks all PMTs and excludes hits after leaving the finite cylinder.
Among true hits, wrong or lost first-PMT assignments total 1.07% for uniformly sampled standard production vertices, 7.22% at an allowed top vertex, and 29.09% at 10 cm inside the barrel.
These are isotropic incidence probes, not complete-event charge-bias estimates.
Central probes agree with the oracle.

The minimal reliable replacement takes the smallest positive sphere entry before the cylinder exit across all sensors, using chunks to limit memory.
A grid traversal or BVH can accelerate this complete test.
Increasing fixed endpoint neighbors cannot guarantee correct tangential intersections.
The isolated prototype restores the correct first sensor on all 22 saved problematic rays.
The accepted sphere approximation does not explain missing hits with those same spheres.

## Detector birth boundary

[Evidence and fixed control](transport/findings.md), [28-line public DATA regression](transport/test_outside_minimal.py), [isolated fix](transport/verify_outside_fix.py).
The minimal test translates an interior emitter outside the ID and asserts zero ID charge.
Instead it produces 909 PE from 4,096 photons.
The larger beam deposits light despite all valid sensor-intersection flags being false.
Apply the birth bounds after all transforms:

```python
mask = (jnp.arange(n_rays) < photon_data['N']) & get_inside_detector_flag(final_origins)
```

This reduces the exterior beam's 22,214 PE to zero while leaving the interior control identical.
The independent physical reference is an opaque ID [blacksheet](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimConstructMaterials.cc#L734).
A physical exterior-light model needs explicit external geometry and transmission boundaries.

The source lane also reproduced the effect with an actual 1 GeV muon event at an allowed production vertex.
Of 203,162 photons, 110,320 were emitted outside the ID after translation.
The unchanged simulator yields 5,027.998 PE, versus 4,930.998 with the bounds veto: 97 false PE, or +1.97%, and 37 extra lit PMTs.
Interior deposit records remain bitwise identical.
This is one affected event, not a cohort average.
See [source evidence](photon_sources/findings.md) and [actual_source_mask_results.json](photon_sources/actual_source_mask_results.json).

## Pion deflection handling

[Source evidence](geant4_physics/findings.md) and [position-continuity regression](geant4_physics/test_source_physics.py).
The replacement uses a previously stored vertex and time with post-step momentum.
In 20 actual 1 GeV pion events, 61 replacements have a maximum position discontinuity of 3.352 m.
The invariant compares the parent's final segment endpoint with the child's creation vertex:

```python
from audit.wc_data_review_20261004.geant4_physics.test_source_physics import test_pion_deflection_preserves_position
test_pion_deflection_preserves_position('')
```

Replace the stored position/time with `track->GetPosition()` and `track->GetGlobalTime()`.
The isolated correction gives zero gap for all 59 observed replacements in its sample.
Changed histories mean the replacement count need not match baseline.
This proves continuity restoration, not a quantified whole-event yield correction.

## GENIE angular distribution

[Source evidence](geant4_physics/findings.md) and [isotropy regression](geant4_physics/test_source_physics.py).
Uniform axes with the existing rotation-angle distribution do not give uniform orientations.
The 20,000-event forward-neutrino control has mean cos(theta)=0.33213 and forward fraction 0.70395.

```python
from audit.wc_data_review_20261004.geant4_physics.test_source_physics import test_genie_isotropic_directions_are_uniform_on_sphere
test_genie_isotropic_directions_are_uniform_on_sphere('')
```

Use a uniform SO(3) rotation, coherently applied to all event momenta.
For a uniform axis, accept a uniform angle on [0,pi] with probability sin(angle/2)^2.
The corrected variant gives mean cos(theta)=-0.00340, forward fraction 0.4971, and second moment 0.33338.
Ordinary particle-gun isotropic directions and Python Shoemake rotations passed their checks.

## Late electronics acceptance

[Evidence and corrections](electronics/findings.md), [executed reproducer](electronics/repro_electronics.py).
`lucid/simulation/digitizer.py:518` deletes arrivals beyond 100 us before the configured trigger evaluates them.

```python
import numpy as np
from lucid.simulation.digitizer import digitize_and_decompose, resolve_model_config
t = np.array([100., 200100.])
sd, _, _ = digitize_and_decompose(sensor_idx=np.array([0,1]), charge=np.ones(2),
    t_true=t, t_reco=t, particle_idx=np.array([0,1]), segment_idx=np.array([0,1]),
    emission_process=np.zeros(2, int), n_sensors=2, model=resolve_model_config('ski'),
    rng=np.random.default_rng(2), apply_resolution=False)
assert len(sd['PE']) == 2
```

Only one digit survives.
The stronger actual SK_WAND response, dark, and trigger check restores 58 delayed physics PE and a second gate when the cutoff is removed in an isolated copy.
Use explicit DAQ acceptance and bounded dark intervals.
Setting this constant to infinity alone can allocate excessive dark for radioactive tails.
The SK-I preset's lack of modern SK-IV forced after-trigger gates is a distinct detector-era limitation.

## Thermal neutron capture physics

[Source evidence](geant4_physics/findings.md), [original capture results](geant4_physics/thermal_results.json), and [corrected distribution](geant4_physics/thermal_thermalelastic_results.json).
This is a physics configuration defect, separate from the late-track and late-digit cuts.
The diagnostic removes the already-proven source time cut so capture physics can be measured independently.
For 1,000 initially thermal neutrons, 997 capture in the original elastic model; capture times have mean 289.94 +/- 15.86 us, standard deviation 500.63 us, and median 129.55 us.
The fraction beyond 1 ms is 5.918%, compared with about 0.750% for the independent 204.42 us water lifetime.
An unphysical surviving-neutron tail even produces a beta decay after about 1,159 seconds.

Stationary-target elastic scattering overcools neutrons, while the generic capture cross section has a low-energy clamp, allowing the capture rate to fall at very low velocity.
Disabling elastic is a causal diagnostic, not a valid fix.
The minimal tested correction replaces `G4HadronElasticPhysics` with `G4HadronElasticPhysicsHPT`, while retaining the original capture model and correct water thermal-scattering mapping.
All 1,000 then capture, with mean 202.71 +/- 6.35 us, standard deviation 200.82 us, and median 137.36 us.
The corrected fraction beyond 1 ms is 0.700%, and its 95th percentile is 603.71 us, versus about 612 us physically.
The [minimal capture regression](geant4_physics/test_neutron_capture.py) fails on the original uncut sample and passes after this change.
The precise constructor change and execution logs are in the source report.
Full HP-capture alternatives remained biased in the executed checks and are not recommended as the tested correction.
Agreement in this thermal benchmark does not certify every neutron energy, material, capture fraction, or displacement distribution.
The independent specialist interpretation is in [astra_physics/findings.md](astra_physics/findings.md).

## Additional confirmed defects and calibration

| Defect | Measured effect and scope | Minimal fix |
| --- | --- | --- |
| Water phase-index table differs from measured water | At 54 MeV, 100 muon events yield 149 versus 947 primary Cherenkov photons after changing only RINDEX, an 84.27% primary-light deficit | Calibrated phase-index knots; regenerate sources |
| Timing response uses true PE rather than sampled charge | Low-charge RMS 3.483 instead of 4.802 ns; high-charge single-PE RMS 3.503 instead of 2.421 ns | Sampled charge with intended floor |
| Medium clamping alters PMT QE | 275 and 674 nm give 128 and 49 PE from 30,000 photons despite configured QE zero; broadband +0.38 to +0.68% | Original wavelengths for QE; clamp only medium interpolation |
| SN chunks overlap in detector time | Two simultaneous 1-PE digits instead of one 2-PE digit; dark PE 2.02 times expected | Merge actual overlapping arrival/readout spans before electronics |
| Direct ROOT CLI ignores readout config | Same real empty event retained with basic/no trigger and rejected with ski/trigger through run_job | Forward digitizer and trigger blocks |
| Entry index mistaken for EventID | Retained event 7 at entry 0 returns 0 of 9 photons; ordinary serial production unaffected | Stored EventID chunk join; handle repeated IDs explicitly |
| Input vertex overwritten in truth | Rigidly displaced real shower at (5,-2,3) m saved at zero, a 6.164 m error | Transform and retain the input interaction vertex |
| Negative TTS erases sensor charge | Realistic-mode 10-PE occupancy loses 99.88%; selected per_segment/tts=0 avoids it | Retain finite negative timestamps |

[Material evidence and minimal threshold code](photon_sources/findings.md), [real A/B measurements](photon_sources/index_ab_results.json).
The water-index discrepancy is a physical input calibration error, separate from LUCiD's explicit constant-speed approximation.
Near-threshold primary-light loss is not a production-wide muon or whole-event deficit; secondary electron light remains.
At high fixed speed, the QE-weighted material discrepancy is about 1.1%.

[Timing tests and fix](electronics/findings.md), [spectral test and fix](water_optics/findings.md), [SN, CLI, and input-vertex tests](production_io/findings.md), [ROOT EventID test](g4_inputs/findings.md), [TTS test](pmt_detection/findings.md).
Each includes minimal code, exact source locations, commands, observations, and applicability.
The SN, direct-CLI, displaced-input, and realistic-TTS cases do not describe ordinary origin-zero single-vertex run_job production.

## Pipeline coverage and rejected concerns

[End-to-end evidence](end_to_end/findings.md) includes two actual 100 MeV electron PhotonSim events processed through `_run_lucid`, selected optics, translation, smearing, dark noise, digitization, trigger, and four HDF5 files.
Independent checks cover digit references, trigger-window inclusion, particle/segment truth-charge conservation, and translated metadata.
Successful execution verifies integration while retaining the physics defects above.

| Stage | Independent checks | Limitations |
| --- | --- | --- |
| ROOT and ancestry | Genuine vector branches; units; shuffled chunks; daughters; electron/pi0 genealogy; empty events | External EventID defect; finite topology coverage |
| Source handoff | 513 photons over padded chunks; one record each; times through 200 us; 20,000 coherent rotations | Source cuts, material and bounds defects |
| Geometry | Bounded all-PMT float64 oracle; central DATA comparison; displaced-source timing | Near-boundary candidate omissions; small grazing float32 errors |
| Transport | 500,000 exponential-hazard trials; scattering mixture; normalized directions; reflection hemisphere | Physical material and phase-function calibration separate |
| Iteration limit | 750,000 photons through 32 iterations; no added detected PE after K=12 | One undetected photon survived 12; not exactly unbiased |
| Sensor response | QE-once control; actual incident boundary/beam; rounding of 0.999999 weights | QE/reflection bug; angular and per-PMT calibration absent |
| Electronics | Selected settings; charge/time/dark tests; delayed gates; foreign-key remaps | Late cut and timing bug; simplified SK response |
| Output | Real ROOT-to-four-files chain; exact float64 second-scale time; charge/window joins | Conditional input-origin metadata; complete truth contract |

The previously fixed double QE is excluded independently: 5,485 observed PE versus 5,525 single-QE expectation and 1,250 double-QE expectation.
The input metre-to-centimetre boundary is correct in displaced-source controls.
Older central-loss estimates from an unbounded-sphere oracle were physically invalid; the corrected water-bounded oracle finds none.
The shared scatter-distance/reflection random key operates on mutually exclusive branches and was rejected as a correlation defect.
The GENIE runner returns entry count despite a misleading variable name, so suspected beamOn overcount was rejected.

## Physical fidelity still unestablished

Passing these checks does not establish exact equivalence to real SK or exhaust all possible events.
The accepted spherical PMT representation is not classified as a bug.
Wall snapping, scalar reflection, finite iterations, and simplified optical/electronic response are recorded as approximations.

LUCiD uses constant n=1.33 rather than dispersive group velocity.
Against WCSim/Geant4, spectrum-weighted arrival over 17 m is about 3.14 ns early and lacks about 0.91 ns chromatic spread.
PhotonSim records polarization, but LUCiD drops it and uses unpolarized Rayleigh scattering.
Executed transverse moments are about (0.300,0.301), whereas linearly polarized photons require (0.200,0.400) in the polarization basis.
These require detector-level validation and are not proved total-charge defects.

The fitted SPE response is explicit approximation: its RMS is 27% narrower than its referenced table and its above-4-PE tail is about 1,467 times smaller in the sample.
The sharp discriminator and Gaussian timing do not reproduce full measured SK response.
Absolute QE provenance, red-water absorption data, per-PMT calibration, angular/immersion response, and water variation remain unresolved.
The chosen SK-I readout is not a calibrated modern SK neutron-tagging DAQ.
These assumptions must be specified or calibrated before claiming exact physical detector agreement.

[Independent physical review](physical_review/findings.md).
[Allocation and ROOT repair workflow](WORKFLOW.md).
