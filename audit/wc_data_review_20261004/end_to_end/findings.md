# SK_WAND end-to-end DATA reference audit

This lane uses `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json` at checkout `82f8d24`.
Production code is unchanged.
Every executed test below ran on CPU in the Slurm `milano` partition with `uv run`.
Batch job `39894061` ran the reference controls on `sdfmilan269`.
Interactive job `39894072` on the same node ran the tangent first-hit regression and subsequent inspection.

## Confirmed production defect: first-hit PMT missing from endpoint candidates

This finding is owned by the geometry lane, and this lane provides an independent end-to-end reproduction through the actual DATA production bucket wrapper.
The physical model here intentionally retains LUCiD's allowed spherical-PMT simplification.
A ray must stop at its first intersected PMT sphere, even when that sphere is well before the cylinder-wall endpoint used to find grid candidates.
The geometrical expectation follows directly from solving `|o + t d - p|^2 = r^2` for all 11,096 configured PMTs and selecting the earliest positive entry.
The oracle uses independent NumPy float64 arithmetic, without LUCiD's intersection or candidate code.

The input photon starts at `(16.8585866627, -0.3529484099, -0.4)` meters and travels in direction `(0, 0, 1)` at emission time 7 ns.
The true first entry is PMT 4105 after 0.166512855 m, with expected arrival at 7.738719170 ns.
The real DATA `per_segment` simulator, `K=12`, `deposit_leg_bound=True`, hard overlap, and production bucket size 256 instead reports PMT 458 at 85.358406067 ns.
The observed arrival is 77.619686896 ns too late, and the only reported PMT is wrong.
The deterministic control makes QE unity and scattering, absorption, and reflection negligible to expose the transport defect without a random detection decision.
These controls preserve the exact selected SK geometry and DATA meter boundary.
The same incomplete candidate lookup is used under the production optical parameters, where the additional erroneous flight also changes scattering and absorption.

The affected path is `_trace_event_bucketed` to the DATA `per_segment` simulator to `_common_propagation` to `lucid.propagation.shared.create_propagator`.
The incomplete endpoint lookup is `lucid/propagation/shared.py:264-272`, where the cylinder-boundary grid cell supplies all sensor candidates.
Adding first-hit ordering at `shared.py:288` cannot recover an earlier PMT excluded from those candidates.
The meter conversion itself is correct: `event_builder.py:252` multiplies by 100 and `simulator.py:769` divides by 100 before tracing.

The executed minimal useful regression is `tangent_data_reproducer.py` in this directory.
The assertion deliberately fails on the current production implementation.
Results are in `tangent_data_results.json` and log `tangent_39894072.log`.
A minimal correct fix is to gather candidates intersected along the entire inward ray segment, then select the earliest positive sphere entry.
A spatially indexed nearest-sphere fallback for grazing or wall-adjacent rays is a focused implementation option.
A full all-PMT nearest-entry search is a correctness oracle for validating such an acceleration.
Changing `K`, endpoint ordering, or overlap temperature cannot fix missing earlier candidates.

## Correct domains directly exercised

`reference_path.py` executes the actual production `_trace_event_bucketed` wrapper and hard DATA `per_segment` simulator with the selected SK_WAND geometry.
The geometry contains 11,096 sensors, cylinder radius 16.962280245 m, height 36.378422222 m, and PMT radius 0.254 m.
Sixty-five targeted PMT rays begin at the displaced origin `(5, -2, 3)` meters.
Every photon reaches the independently predicted PMT, and the maximum independent flight-time error is 0.000942719 ns.
This is direct evidence that the previously fixed factor-100 origin error is absent from this production meter boundary.

At unity QE with no reflection, absorption, scattering, or TTS, padding 65 photons to 256, dividing them into buckets of 16 and 32, and changing irrelevant segment labels all preserve charge and first-hit time exactly.
Every control yields exactly 65 positive photon records.
These controls validate masking and aggregation without demanding identical random realizations across differently shaped stochastic draws.

For 8,192 isotropic rays at the detector center, a float64 all-PMT sphere-entry oracle independently predicts 3,472 sensor hits.
The production DATA wrapper reports the same 3,472 sensors, with zero extra, missing, or incorrectly identified sensors.
This is central-ray coverage and does not validate the wall-adjacent domain, which fails the separate regression above.

For 32,768 monochromatic 400 nm photons launched 1 m in front of PMT 5200, the direct sphere flight is 0.746 m.
Using the configured scalar reflection convention, the independent direct detection probability is `exp(-d * (alpha_R + alpha_M + alpha_A)) * (1 - R_sensor) * QE(400)`.
The independent formula evaluates coefficients directly from the selected water JSON equations and interpolates the selected QE JSON without calling LUCiD's optics implementation.
It predicts 5,524.586741 direct PE for one QE application and only 1,249.610379 PE if QE is applied twice.
The observed DATA count is 5,485 PE, or -0.584 standard deviations relative to the single-QE binomial expectation.
The test decisively excludes the previously fixed double-QE defect in this selected call path.
This agreement checks implementation against the configured reflection/QE convention; the physical validity of measured QE conditional after reflection is audited separately by the PMT lane.

The hard first-hit survival cap returns 0.999998987 weight for one photon instead of exactly one.
This is a one-part-per-million numerical bias and is not classified as a major defect here.
The electronics lane reports that production rounds these weights before SPE sampling.

The reference test completed in 27.84 seconds and passed its assertions.
Results are in `reference_results.json` and log `reference_39894061.log`.

## Exact reproduction

The approved batch script is `run_reference.sbatch`.
Submit it with `sbatch audit/wc_data_review_20261004/end_to_end/run_reference.sbatch`.
The underlying invocation, run inside an approved CPU allocation, is:

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/end_to_end/reference_path.py
```

Replace the final filename with `tangent_data_reproducer.py` for the confirmed failing regression.
The CUDA plugin warning does not affect the CPU run; `JAX_PLATFORMS=cpu` was explicit.

## Limitations

The positive controls do not establish full physical correctness of SK_WAND DATA mode.
The confirmed grazing first-hit defect already disproves that assertion.
They also do not validate upstream Geant4, measured SK optical constants, PMT material response, digitizer, or trigger, which other lanes examine independently.
The complete available ROOT-to-production-writer path was executed with two true PhotonSim electron events, as detailed below.

## Executed ROOT-to-four-file production integration

`root_writer_control.py` writes a real uproot TTree with 8,192 injected 400 nm photons emitted from `(5, -2, 3)` meters toward 64 configured PMTs, reads it through `_read_event_raw`, and executes the production `_run_lucid` wrapper with SK_WAND.
It verifies the reader's millimeter-to-meter origin boundary before simulation.
This controlled fixture uses a scalar primary creator-process string for uproot writer compatibility; it does not validate the exact Geant4 `vector<string>` schema.
The actual SKI digitizer, 4.2 kHz dark noise, and configured 25-hit trigger execute and write all four HDF5 files.
With resolution disabled for this conservation control, the written sensor charge equals the summed per-digit truth decomposition exactly.
The event contains 291 digits, 1,395.998657 physics PE, and 53 dark PE.
Every saved digit time is inside its recorded trigger window, and hit sensor identities agree with the referenced digit.

The controlled nonzero primary position is preserved in the segment positions but the interaction vertex label is `(0, 0, 0)`.
This conditional truth-metadata defect is owned by the production_io lane; normal WAND-generated Geant4 vertices at the origin are unaffected by that particular defect.
The result is recorded in `writer_validation_results.json` under `writer_output`.

`true_root_writer.py` executes the unchanged production `_run_lucid` wrapper on `geant4_physics/e100tiny.root`, generated by the actual PhotonSim binary.
This input has two real 100 MeV electron events, 21,225 and 22,143 photons, 3,402 and 3,237 raw segments, and valid Geant4 creator-process vector branches.
It uses the selected reference geometry and physics configs, default production translation and resolution, the actual SKI digitizer and trigger, and an 8,192-ray bucket to limit startup work.
Both events pass the trigger and are saved in `true_writer_output/{sensor,hits,step,labl}/wc_*_0000.h5`.
Event 0 contains 662 digit records and 915.767456 reconstructed PE, with 726.999268 truth physics PE and 40 truth dark PE.
Event 1 contains 641 digit records and 973.259766 reconstructed PE, with 799.999207 truth physics PE and 38 truth dark PE.
Reconstructed sensor charge and truth photoelectron counts differ because resolution and the charge discriminator are enabled; they are not required to be equal.

The independent `validate_output.py` check joins digit references and verifies sensor identities, trigger-window inclusion, and exact physics-charge conservation between particle and segment truth decompositions.
It also verifies primary energy 100 MeV, PDG 11, and agreement of the written interaction vertex with the translated first primary segment start for these origin-centered input events.
The existing structural verifier additionally accepts both events, including digit foreign keys and trigger-window offsets.
This is successful pipeline execution and consistency coverage; it does not supersede the physical defects identified in this audit.
All writer runs and independent checks execute in Slurm milano job `39894072`.
The execution logs are `writer_control_39894072.log`, `true_writer_39894072.log`, and `writer_validation_39894072.log`.
