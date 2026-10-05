# Independent physical review

Reference: `82f8d24`, `config/SK_WAND_geom_config.json`, and `config/SK_WAND_physics_config.json`.
The lane made no production changes and used zero GPUs.
Numerical execution used shared Slurm job **39894013**, partition `milano`, node `sdfmilan269`, with the allocation owner's permission.
The unused pending private job `39894036` was canceled.
After the user's tightened allocation requirement, inspection and file edits also ran in that shared allocation.

## Outcome and limits

This review independently read the data-mode source adapter, shared propagation scan, sampled photon iteration, reflection models, sensor response, geometry normal construction, and production photon plumbing.
Its executed diagnostic supports the current unpolarized Rayleigh rotation and reflected-direction implementation.
It identified a concrete physical limitation involving polarization, but does not classify a modeled unpolarized approximation as a new major implementation bug without an agreed physical acceptance requirement.
Other lanes own and reproduce the QE/reflection semantics, out-of-band QE, time-of-flight, external photon births, and Geant4 physics-list findings.
Literal equivalence to a real detector cannot be certified from this software review because the implemented approximation set and authoritative detector calibration are not equivalent to a complete detector physical specification.

## Executed independent optical invariants

Artifact: `repro_independent_optical_moments.py`.
Results: `results.json` and `optical_moments.log`.
Command inside the allocation:

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/physical_review/repro_independent_optical_moments.py
```

The script executed **250,000 rays for each of five incident axes**, including axes unrelated to the world coordinate directions.
For the unpolarized dipole distribution, the independently expected local second moments are `(0.3, 0.3, 0.4)` and the mean vector is zero.
The observed local moments were approximately `(0.300820, 0.299664, 0.399467)` for every axis.
The largest absolute local mean was `0.001880` and the maximum unit-vector norm error was `1.19e-7`.
This directly checks the world/local frame mapping rather than trusting its comments or existing test suite.
The scalar-mix reflection sample sent every one of **250,000 wall rays** and **250,000 sensor rays** into the water hemisphere for the chosen oblique incidence.
The observed reflection probability values were `0.0500000` and `0.25`, respectively.
These checks verify specific distributions and coordinate conventions, not total detector physics.
The environment emitted a CUDA-plugin compatibility warning, but `JAX_PLATFORMS=cpu` forced CPU execution and the calculations completed successfully.

## Physical limitation: input polarization is discarded

The actual PhotonSim producer records `PhotonPolX/Y/Z` in `DataManager.cc:154`, and obtains the polarization vector in `SteppingAction.cc:209`.
The LUCiD reader requests only positions, directions, emission times, and wavelengths in `lucid/sources/root_reader.py:151`.
`PhotonState` in `lucid/simulation/types.py:56` has no polarization field.
`compute_scatter_direction` in `lucid/simulation/optics.py:100` samples a uniform azimuth and a polar law proportional to `1 + cos(theta)^2`.
Thus the pipeline implements an unpolarized scattering approximation regardless of the supplied Cherenkov photon polarization.
This identifies the explicit implemented distribution, not evidence that its physical accuracy has been validated for SK.

Geant4's [Cherenkov implementation](https://github.com/Geant4/geant4/blob/master/source/processes/electromagnetic/xrays/src/G4Cerenkov.cc#L308) creates a linearly polarized photon.
Geant4's [Rayleigh implementation](https://github.com/Geant4/geant4/blob/master/source/processes/optical/src/G4OpRayleigh.cc#L135) projects the incoming polarization onto the outgoing transverse plane and accepts a direction according to the squared polarization projection.
For incoming direction `z` and polarization `x`, this gives a normalized angular density `3/(8*pi) * (1 - d_out.x**2)` and exact second moments `(0.2, 0.4, 0.4)`.
The existing sampler instead measured global second moments `(0.299664, 0.300820, 0.399467)`.
The transverse variance along the polarization axis is therefore about **50% higher**, and along its perpendicular about **25% lower**, than the linearly polarized single-scattering expectation.
These percentages concern the conditional singly scattered distribution and must not be presented as an overall SK photon-count error.
The effect on real event charge, late-light maps, and timing requires a detector-level comparison because photons have many emission polarization directions, source wavelengths, and geometrical paths.

A small explicit reproduction of the physical discrepancy is:

```python
from lucid.simulation.optics import compute_scatter_direction
import jax
import jax.numpy as jnp
import numpy as np

keys = jax.random.split(jax.random.PRNGKey(62409), 250_000)
dirs = np.asarray(jax.jit(jax.vmap(compute_scatter_direction,
                  in_axes=(None, 0)))(jnp.array([0., 0., 1.]), keys))
print(np.mean(dirs * dirs, axis=0))  # observed ~(.300, .301, .399)
# Polarization x requires (.2, .4, .4), which the API cannot encode.
```

A physical enhancement requires preserving the already available ROOT polarization vectors, rotating them with the event, carrying them through the photon state, and using a polarized Rayleigh sampler with polarization updates.
Reflection must preserve or transform polarization consistently if the implementation claims polarization-sensitive surface physics.
There is no honest one-line fix because the current source-to-state interface discards a physically relevant state variable.
This is a fidelity limitation and audit gap under the user's permitted-approximation rule, not a new major yield bug claimed by this lane.

## Candidate defects handed to the responsible lanes

- **External photon births:** the production photon arrays are translated and traced without an initial emission-volume mask.
  `event_generation.py:443` and `event_builder.py:244` send the entire translated photon list, and `simulator.py:573` starts every input ray alive.
  The detector-volume check in `simulator.py:666` occurs after computing that step's deposit.
  The photon-sources lane independently confirmed the missing input mask, and geometry/transport own the executed detector-boundary reproduction and minimal fix.
  In the chosen closed-cylinder optical model, an external photon crossing an opaque boundary or approaching an ID sensor from behind cannot be treated as a front-face photon born in the instrumented water.
  This input-mask issue is distinct from the fidelity of charged secondary production outside the detector.
- **QE and reflection:** `photon_iteration_sample` makes nonreflection and detection the same boundary branch; Bernoulli QE is then applied in `sensor_response`.
  This yields an incident detection probability `(1 - R) * QE`.
  Whether the configured QE is absolute external quantum efficiency or conditional on nonreflection determines whether that factor is valid.
  The PMT lane owns the photocathode physical reference, calibration convention audit, and executed yield reproduction.
- **Clipped out-of-band QE:** `evaluate_optical_model` clips the wavelengths to the medium grid before evaluating `qe_fn`, preventing the original input wavelengths from exercising its true out-of-domain behavior.
  The water-optics lane owns the source-appropriate executed reproduction and physical effect.

## Rejected false positives

- **The sampled path reuses `k2` for a truncated scattering distance and reflection Bernoulli draw.**
  The source code uses an independent `k1` for the reach-surface decision.
  A reached-surface branch never uses the sampled scattering distance and a scattered branch never uses the reflection draw.
  This key reuse does not correlate two jointly realized physical observables in one step.
- **Rayleigh and Mie candidate directions use the same `k3`.**
  An independent `k5` selects which scattering mechanism occurred, and only that mechanism's direction is realized.
  The unused candidate does not create an additional physical draw or a correlation in the realized path.
- **Absorption is correlated with direction because scattering and absorption share a key.**
  The current source uses the dedicated `k6` for the absorption survival draw.
  The historical shared-direction key is absent from this checkout's sampled path.
- **Sensor reflection restarts inside the sensor because all normals are negated.**
  `compute_sensor_intersections_base` negates the spherical outward normal before returning it, and the photon step negates that returned normal when moving the restart point into water.
  The paired signs are consistent with the two physical water boundaries.
  The executed reflected-hemisphere oracle confirms the resulting direction convention for oblique incidence.
- **Wavelength QE is mismatched after flattening scan outputs.**
  The actual output shape is `(K, candidates, rays)` and C-order flattening makes the original ray index `flat_index % rays`.
  The code uses exactly this mapping.
- **Default data mode uses expected-value photon transport because `use_expected_value=True`.**
  The `is_data` dispatch chooses `photon_iteration_sample` before considering the expected-value flag.
- **Raw detected photoelectron counts require an extra Poisson draw.**
  For a fixed emitted photon list, physical quantum conversion is Bernoulli thinning, and the detected total is conditionally Binomial or Poisson-binomial.
  Photon-count fluctuations originate upstream in the actual emission process.
  Adding an independent Poisson draw after physical Bernoulli conversion would add variance without a corresponding physical process.
  The comments calling a fixed-input Bernoulli total "raw Poisson counts" are not sufficient evidence of a defect.

## What remains physically unproved

The reference configuration does not establish authoritative measured per-PMT QE, collection efficiency, angular response, timing offsets, charge response, or their evolution with detector period.
The scalar surface-reflection coefficients do not specify the real angle, wavelength, and polarization dependence of photocathode or black-liner reflection.
A bulk-water precomputed charged shower translated into a bounded detector does not by itself validate charged transport and secondary production across detector-material boundaries.
Fixing any initial optical-emission mask is separate from validating the charged shower outside the modeled detector.
Uniform optical curves do not validate real water stratification or changing attenuation.
Spherical PMTs and those other modeled approximations require explicit accuracy tolerances and detector-level calibration comparisons before claiming equivalence to a real SK-like instrument.
Existing unit tests, agreement between two paths using the same simplified physics, or the independent invariants above cannot supply that missing physical validation.

## Shared-step execution detail

The initial shared interactive step inherited `SLURM_STEP_NUM_TASKS=2` from its parent allocation.
It was closed and the entire diagnostic was reexecuted successfully in an explicit `--ntasks=1` shared step of the same job.
The final `results.json` and `optical_moments.log` are from that single-task reexecution.
