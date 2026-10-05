# WC DATA transport review

Reference: checkout `82f8d24` and the requested SK_WAND geometry and physics files.
All numerical execution used Slurm CPU partition `milano`; no GPU was used.
Production files were not edited.

## Confirmed boundary bug: exterior photons produce ID photoelectrons

The DATA simulator does not reject photons emitted outside the inner-detector water volume after rotation and translation.
Initial survival is one at `lucid/simulation/simulator.py:573`, and the DATA intensity mask at `simulator.py:818-820` checks only the padded photon count.
At a wall entry from outside, geometry reports no valid PMT intersection, but its independent line-proximity weight can still be approximately one for a PMT whose back lies on that line.
The weight is formed before the volume condition at `lucid/propagation/base.py:297`, while the valid sensor condition appears separately at `base.py:309-318`.
The scan deposits this weight at `simulator.py:676-677` without requiring `hit_sensor` or an initially interior origin.
Checking the new position at `simulator.py:666-669` restricts continuation and does not prevent the current false deposit.

This violates the boundary of the SK-like ID model, whose blacksheet separates inner-detector light from exterior light.
An independent detector implementation gives blacksheet absorption length `1e-9 cm`, zero photoelectron efficiency, and a reflecting/absorbing surface: [pinned WCSim material construction](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimConstructMaterials.cc#L734-L746), [surface setup](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimConstructMaterials.cc#L1944-L1954).
The exterior ray hits the rear boundary and has no geometrically valid sensor candidate, so accepting sphere-shaped PMTs does not explain this deposit.

The reproducer uses production DATA settings: `K=12`, `temperature=0`, `hit_mode='per_segment'`, `deposit_leg_bound=True`, wavelength-dependent water/QE, and both requested files.
For 100,000 photons at 400 nm emitted 0.5 m outside barrel PMT4104 and directed inward, LUCiD returns 22,214 PE, including 22,007 on that PMT.
All four `inside_sensor` flags are false, but the target candidate weight is `0.99999899`.
The independent opaque-boundary expectation is zero ID PE.
An interior control 0.5 m inward of the same PMT returns 21,206 total PE, including 17,391 on the target and additional reflected light elsewhere.

The smallest executed failing test is `test_outside_minimal.py`.
It starts with an interior emitter, then applies the production translation pathway to move it 1 m outward, so the missing check must follow the transform.
On 4,096 photons it fails with `AssertionError: External photons made 908.9999389648438 ID PE`.
Its full code is 28 lines and uses the public DATA simulator without mocks.

Minimal suggested fix at `simulator.py:819` for the current ID-only surface model:

```python
mask = (jnp.arange(n_rays) < photon_data['N']) & get_inside_detector_flag(final_origins)
```

`verify_outside_fix.py` executes that one-line change in an isolated production-module copy.
The 100,000-photon exterior case falls from 22,214 PE to exactly zero, while every interior control charge remains unchanged.
A model intended to include exterior light needs explicit opaque surfaces and external detector geometry instead of treating rear incidence as an ID PMT hit.

The bug applies to production because source translation is enabled by default in `lucid/production/run_job.py:296`, and translated exiting tracks can emit outside the ID.
The source and event-builder paths do not crop these photons before calling the simulator.
The false response per affected exterior photon is large; campaign-wide bias depends on photon positions and directions, so the 22.2% targeted-beam response must not be presented as a whole-event bias.
The source lane is quantifying exterior fractions from actual Geant4 samples separately.

Executed job: interactive CPU allocation `39894013`, partition `milano`.
Artifacts: `reproduce_outside.py`, `outside_results.json`, `outside-39894013.log`, `test_outside_minimal.py`, `minimal-outside-39894013.log`, `verify_outside_fix.py`, `outside_fixed_results.json`, and `outside-fixed-39894013.log`.
Run inside a `milano` or `roma` allocation from the repository root:

```bash
export JAX_PLATFORMS=cpu
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/transport/test_outside_minimal.py
```

Replace the final filename with `reproduce_outside.py` for the larger measurement or `verify_outside_fix.py` for the fix check.

## Transport mechanisms checked without a large new defect

`reproduce_transport.py` independently checks one-step probabilities against homogeneous exponential scattering and absorption hazards.
For boundary distance D, total scattering rate s, absorption rate a, and reflectance R, independent expectations are `P(reach)=exp(-sD)`, `P(surviving reflected)=R*exp(-(s+a)D)`, and `P(surviving scatter)=s/(s+a)*(1-exp(-(s+a)D))`.
For scattering conditioned to precede the boundary, the mean flight distance is `1/s-D/(exp(sD)-1)`.
For the specified unpolarized Rayleigh/Henyey-Greenstein model, mean scattering cosine is `g*(1/L_M)/(1/L_R+1/L_M)`.

On 500,000 trials, all nine measured fractions or moments agree with these analytic expectations within two standard errors, and directions remain normalized and finite.
These checks cover combined Rayleigh/Mie rate, mixture selection, conditional flight distance, absorption-event competition, and reflection/scattering termination.
The shared key for truncated scatter distance and reflect/no-reflect choice is consumed on mutually exclusive branches with an independent surface-reach decision; this is not a confirmed correlation defect.
The dedicated absorption key is independent of the scattering-angle key.

A 32-iteration run through actual SK_WAND geometry and current production optical curves used 250,000 photons each from the center, 0.5 m inside the barrel, and 0.5 m inside an end cap.
Relative expected-PE losses at API default `K=7` were 0.01479%, 0.00718%, and 0.00714% respectively.
At production `K=12`, no additional detected PE appeared between iterations 13 and 32 in these samples.
One undetected photon in the end-cap sample was still alive after iteration 12, so finite K is not proven exactly unbiased.
No living photon developed a non-finite position.
This supports treating production K truncation as a small numerical approximation in the sampled configuration rather than claiming a large charge-loss bug.

Executed batch job: `39893952`, partition `milano`, completed with exit code zero.
Artifacts: `reproduce_transport.py`, `run_cpu.sbatch`, `cpu-39893952.log`, and `results.json`.
Submission command: `sbatch audit/wc_data_review_20261004/transport/run_cpu.sbatch`.

## Limits and scope handoffs

These results validate specific stochastic transport laws and a confirmed exterior boundary defect, not all detector physics.
The statistical tests use current water tables and QE interpolation to isolate transport, so they do not independently certify those inputs.
The water-optics lane covers phase/group velocity, wavelength curves, and the physical scattering phase function.
The PMT lane covers incident QE versus reflectance normalization.
The geometry lane independently confirms near-wall first-sensor omissions and misrouting, which can alter absorption, scattering, timing, and sensor assignment more than the K cutoff.
