# Independent PMT probability review

Scope: LUCiD `82f8d24`, `config/SK_WAND_geom_config.json`, and `config/SK_WAND_physics_config.json`.
This lane assessed QE and reflectance semantics; the PMT lane owns the numerical reproducer.
No production files were edited and this lane ran no numerical jobs.
After the stricter allocation instruction, all filesystem inspection and report writing ran in CPU allocation `39894048` on `milano`, host `sdfiana007`.

## Finding: incident QE is used as conditional conversion efficiency

The code implements `P(detect on this encounter) = (1 - R_sensor) * q`, where q is the configured QE curve including sensor corrections.
This is correct only if q means detection probability conditional on a photon not being reflected.
It is incompatible with the measured incident-photon QE semantics advertised by `docs/concepts/wavelength.md:59`.

[Motta and Schonert, arXiv:physics/0408075, equation 1](https://arxiv.org/pdf/physics/0408075) defines `QE(lambda, theta) = A(lambda, theta) * Pconv(lambda)`, with incident-photon probabilities `A + R + T = 1`.
Thus external QE already includes absorption probability, and a second nonreflection factor repeats a loss contained in that measurement.
The paper also notes that manufacturer QE spectra normally use air at near-normal incidence; immersion and angular corrections are separate calibration questions.

Elementary conditional probability gives effective surface outcomes: detection q, reflection R, and terminal loss `1-q-R`, requiring `q+R<=1`.
If reflection is sampled first, the remaining branch must use `P(detect | not reflected)=q/(1-R)`.
The current branch instead uses q, yielding `(1-R)q`.
At production R=0.25 this is a 25% deficit relative to supplied incident QE at an otherwise equivalent sensor encounter.
Restoring the convention increases affected yield by one third relative to current yield.
This does not establish exactly 25% event-charge disagreement with real Super-Kamiokande, because other optical approximations and subsequent encounters also matter.

## Source call path

`lucid/wavelength/medium.py:67` reads JSON `qe_percent` and divides by 100 without reflection calibration.
`lucid/detector_params.py` projects scalar response QE from this curve at 400 nm.
`lucid/wavelength/optical_model.py:131` multiplies the same curve by its wavelength deviation.
`lucid/simulation/simulator.py:523` divides by reference QE, then `:965` and `:993` restore `response.qe`; nominally this returns the original curve.
`lucid/simulation/photon_step.py:98` and `:99` first classify the arriving photon as reflected or nonreflected.
`lucid/simulation/sensor_response.py:226` and `:280` apply the supplied QE as Bernoulli acceptance to the nonreflected candidate.
The expectation path has the same extra factor in `lucid/simulation/photon_step.py:286`.
There is no hidden `1/(1-R)` normalization.

## WCSim and Geant4 distinguish these probabilities

[Geant4 G4OpBoundaryProcess::DoAbsorption](https://github.com/Geant4/geant4/blob/master/source/processes/optical/include/G4OpBoundaryProcess.hh) applies its EFFICIENCY Bernoulli after absorption is selected.
That boundary property is conditional; its name does not make it equal to external QE.
For an opaque effective surface it needs `EFFICIENCY=q/(1-R)` to reproduce external QE.
For a transmissive photocathode the microscopic conversion efficiency is `q/A`.

[WCSimWCSD.cc](https://github.com/WCSim/WCSim/blob/develop/src/WCSimWCSD.cc) explicitly sets `ratio=1./(1.-0.25)` before requesting QE in method 3, SensitiveDetector_Only.
[WCSimStackingAction.cc](https://github.com/WCSim/WCSim/blob/develop/src/WCSimStackingAction.cc) applies the same ratio in methods 1 and 2.
This corroborates the distinction between input QE and conditional acceptance.
The fixed historical WCSim correction should not be copied blindly for arbitrary LUCiD reflectances.

## Reproduction and provenance limits

The PMT lane's boundary experiment in job `39893957` used 300,000 incident photons and actual SK_WAND QE at 400 nm.
Input QE was `0.2261907458`, nonreflected fraction `0.7503799796`, and measured direct detection probability `0.16977`.
The implemented formula predicts `0.1696430594`, while incident-QE semantics predict `0.2261907458`.
The measured relative deficit is `0.2494387895`.
Log: `audit/wc_data_review_20261004/pmt_detection/cpu-39893957.log`.
That job's separate end-to-end assertion failed because its initial beam source was inside the sensor sphere; its boundary result remains valid, but the failed beam is not production-impact evidence.
The PMT lane corrected the source and submitted `39894066`; consult that lane's final report for the corrected result and exact reproduction command.

Exact provenance of `config/pmt/SK_QE.json` digits was not established.
Git commit `dbf5c87` introduced the table without measurement metadata; the deleted predecessor CSV identified itself as development data and cannot authenticate this replacement.
The verified defect is therefore mismatch of probability composition with documented incident-QE semantics.
Whether these exact entries were measured under specified illumination or medium conditions remains unverified.
If independent provenance proves the table was intentionally converted to conditional efficiencies, the physical-loss claim would not apply to that table, but documentation and the external-QE import contract would still be wrong.

## Minimal correction and regression

Specify incident-photon QE as the configuration contract and compose mutually exclusive detection, reflection, and loss probabilities consistently.
The smallest correction preserving reflection-first sampling uses `q/(1-R_sensor)` on nonreflected sensor encounters, with actual encounter reflectance, validation of `0<=q<=1-R_sensor`, and handling of `R_sensor=1`.
Do not scale detected PE weights by `4/3`; that produces fractional photoelectrons and wrong counting fluctuations.
In the expectation branch, deposit incident QE times reach and absorption factors while continuing reflected weight with R.
Keep both implementations consistent.
A targeted regression should illuminate a real sensor from outside its sphere at fixed wavelength, disable bulk attenuation and scattering, vary compatible R values, and verify first-encounter PE fraction remains q while reflected fraction becomes R.
Separately establish QE table provenance, measurement medium, and angular convention before claiming absolute real-detector agreement.

## Corrected end-to-end confirmation

PMT lane job `39894066` completed successfully with the beam origin 5 m before the sensor center and 50,000 input photons through the SK_WAND data-mode simulator.
For K=1, R=0 yielded `10993.98828125` direct PE and R=0.25 yielded `8220.9912109375`.
For K=12, R=0 yielded `10997.98828125` prompt PE and R=0.25 yielded `8225.9912109375`, approximately 0.748 times the zero-reflectance yield.
This independently ties the boundary-law deficit to the actual production geometry and data path.
It also shows reflected light partially restores total charge, so the prompt deficit and total-charge deficit must be reported separately.
Log: `audit/wc_data_review_20261004/pmt_detection/cpu-39894066.log`.
Reproduction entry point: `sbatch audit/wc_data_review_20261004/pmt_detection/run_cpu.sbatch`, which executes that lane's `reproduce.py` through uv in a milano CPU allocation.


## Thermal-neutron specialist assessment

GPT 6 Astra separately reviewed the source lane's excessive capture-time tail and surviving neutron.
It identified a thermal-neutron physics configuration defect rather than treating a fitted mean as sufficient evidence.
Stationary-target elastic scattering can overcool neutrons; with a low-energy capture cross-section clamp, the capture rate n*sigma*v then decreases.
Disabling elastic scattering is a causal diagnostic, not a physical correction.
The specialist recommended validating bound-water thermal scattering, material temperature and mapping, the evaluated data, capture-time shape, late tail, survivor outcomes and displacement.
At that advisory stage, a thermal-scattering correction was a candidate, and HP capture alone had already failed.
[Geant4 HPT construction](https://github.com/Geant4/geant4/blob/master/source/physics_lists/lists/src/G4PhysListFactory.cc) and [the SKG4 configuration](https://arxiv.org/html/2412.04186v2) support checking bound-water thermal scattering.
The SKG4 paper's separate thermal-boost modification concerns competing hydrogen/gadolinium capture fractions and does not repair pure-water capture timing.

The source lane subsequently validated a narrower correction: replace only the elastic constructor with G4HadronElasticPhysicsHPT and retain the original capture model.
Its actual ROOT regression restores both the pure-water thermal capture mean and the late tail.
Expanded HP-capture variants remain unsuitable in the executed controls.
See [the final source report](../geant4_physics/findings.md) for the completed tests and scope.
This supplement records the specialist's advisory interpretation and the source lane's later evidence; it does not claim independent specialist execution of those experiments.
