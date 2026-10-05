# SK_WAND geometry audit

Reference checkout: `82f8d24`.
Reference geometry: `config/SK_WAND_geom_config.json` -> `config/sk_geometry.npz`.
Production propagation uses hard overlap (`temperature=0.0`) and `deposit_leg_bound=True` at `lucid/production/run_job.py:308-318`.
No production files were edited.

## Confirmed defect: endpoint-only sensor lookup misses the first PMT

**Applicability: current SK_WAND production data mode.**
The sparse lookup is keyed only by the ray's eventual intersection with the cylindrical envelope at `lucid/propagation/shared.py:264-272`.
Only four sensors near that wall or cap cell are tested at `shared.py:274-286`.
The sensor positions are on the envelope, but their accepted spherical detection surfaces protrude into the water.
A ray can intersect one of these spheres well before reaching the cell used for candidate selection.
The missing sensor never reaches the otherwise valid first-hit ordering at `shared.py:290` and `lucid/propagation/base.py:56-64`.
This causes missing photons, assignment to a later PMT, incorrect arrival times, incorrect absorption/scattering flight lengths, and reflection at the wrong sensor.
This remains a bug while accepting the sphere approximation: the implementation does not correctly trace its own declared spheres.

### Minimal geometry reproduction

The following uses the actual reference geometry and the exact production transport options.
Run it only within a `milano` or `roma` allocation through `uv run`.

```python
import jax.numpy as jnp
from lucid.geometry.detector_geometry import DetectorGeometry

g = DetectorGeometry.from_config("config/SK_WAND_geom_config.json",
                                temperature=0.0, deposit_leg_bound=True)
o = jnp.array([[16.858586662707683, -0.352948409856257, -0.4]])
d = jnp.array([[0.0, 0.0, 1.0]])
r = g.propagator(o, d)
assert 4105 in r["sensor_indices"][:, 0]  # Fails: [7757, 458, 461, 7758].
```

PMT 4105 is the first sphere hit, at `0.4 - sqrt(0.254**2 - 0.1**2) = 0.16651338 m`.
The origin lies inside the water and outside every PMT sphere.
The current propagator chooses PMT 458 at `17.66257515 m` instead.
The independent finite-cylinder exit is `18.58921111 m`, so the expected first hit lies inside the actual travelled water leg.

A separate full data-path reproducer, `../end_to_end/tangent_data_reproducer.py`, independently confirmed the same failure through `_trace_event_bucketed`, the production meter-to-centimeter-to-meter conversion, bucket 256, `K=12`, and `per_segment` hit mode.
With ideal optical response and emission time 7 ns, PMT 4105 should receive the photon at `7.738719 ns`.
The produced hit is on PMT 458 at `85.358406 ns`, which is `77.619687 ns` late.
That test was executed in `milano` job `39894072`; its log is `../end_to_end/tangent_39894072.log`.
The ideal response isolates the geometry defect while retaining the real SK_WAND geometry and production data plumbing.

### Independent oracle and measured incidence

`reproduce_geometry.py` computes the first positive ray-sphere entry over every one of the 11,096 PMTs using NumPy float64.
It independently computes the finite-cylinder exit, then excludes sphere intersections after that exit.
The discriminant is computed from the closest-approach vector rather than subtracting two large squared path lengths.
Every sampled origin is outside all PMT spheres.
`results.json` and `cpu-39894012.log` contain the resulting counts.
Each row has 20,000 isotropic direction probes.
Percentages below are relative to true PMT hits, with lost and wrong-first categories disjoint.

| Origin | True hits | Lost hit | Wrong first PMT |
| --- | ---: | ---: | ---: |
| Center | 8,660 | 0.000% | 0.000% |
| r = 8 m, z = 0 | 8,970 | 0.011% | 0.000% |
| r = R - 2 m, z = 0 | 10,960 | 0.940% | 1.651% |
| r = 15 m, z = 0 | 10,824 | 0.878% | 1.524% |
| r = 16.5 m, z = 0 | 11,946 | 3.742% | 10.079% |
| r = R - 0.1 m, z = 0 | 10,158 | 7.049% | 22.042% |

At r = R - 0.1 m, the first true sensor is absent from the four candidates for 2,954 of 10,158 true hits (29.080%).
The remaining single failure is a separate small floating-point boundary issue, described below.
There were no extra geometric PMT hits relative to the bounded oracle.

`reproduce_production_population.py` additionally samples the actual standard production vertex volume, `r <= 0.9 R` and `|z| <= 0.9 H/2`, using the same distribution as `lucid/sources/writer.py:347-353`.
Its directions are independent isotropic probes rather than a particular particle shower.
In 20,000 such probes, 9,825 true hits included 36 losses (0.366%) and 69 assignments to the wrong first PMT (0.702%).
At the allowed top vertex `(0, 0, 0.9 H/2)`, 10,472 true hits included 227 losses (2.168%) and 529 wrong first PMTs (5.052%).
These probe rates demonstrate position-dependent geometric corruption within the standard vertex volume.
They are not measurements of the final PE or energy bias for a complete WAND event population.
Charged-track emission origins and reflected/scattered photons can approach the envelope more closely than the interaction vertex.

### Minimal suggested fix

Replace endpoint-cell candidate selection for hard data mode with a complete ray-segment PMT broad phase, then choose the smallest valid positive sphere-entry distance before the cylinder exit.
A small correctness-first implementation evaluates all PMT spheres and takes `argmin` over entries satisfying `0 < t_entry <= t_exit`.
A chunked reduction avoids constructing a full photons-by-PMT tensor for production-sized inputs.
The subsequent sensor normal, position, flight time, optical attenuation, and reflection must all use that same first intersection.
For production performance, a volume grid traversal or BVH may accelerate the same complete intersection test.
Increasing a fixed number of sensors near the envelope endpoint cannot guarantee correctness for tangential rays that strike a sphere many meters before that endpoint.

`reproduce_selection_fix.py:24-36` is an executed local prototype of the complete bounded candidate selection.
It picks the independent oracle's first sensor on all 22 saved problematic rays.
The existing sensor-intersection function, supplied with the corrected candidate for the minimal case, returns PMT 4105 at `0.16651316 m` with a valid inside-sensor flag.
The prototype and results are in `fix_and_boundary_results.json` and `fix-cpu-39894012.log`.
It does not edit the production implementation.

## Supporting boundary evidence for the transport lane

The same selection-fix script tests a photon born 2 m outside the cylinder and aimed radially inward at PMT 4105.
Its origin is outside the detector, every returned `inside_sensor` flag is false, yet PMT 4105 receives overlap weight `0.99999899`.
The sphere's first entry is outside the water and is correctly rejected by the geometry flag, but the deposit weight at `lucid/propagation/base.py:297` is not gated by that flag or by the original emission bounds.
`fix_and_boundary_results.json` preserves the exact origin, direction, candidates, flags, weights, times, and stop.
The source lane independently verified that production does not reject translated photon origins outside the inner detector before propagation.
The transport lane owns the full data-mode reproduction, physical applicability, and minimal boundary fix for this defect.

## Checks and limitations

The `.npz` file contains 11,096 unique PMT IDs and the same ID set as `config/geofile_SuperK.txt`.
Its stored positions agree with the geofile after cm-to-m and mm-to-m conversions to within `5.11e-15 m`.
There are 7,752 barrel, 1,672 top, and 1,672 bottom sensors.
All stored viewing directions have unit norm within `5e-10`.
The configured grid is `(n_cap, n_angular, n_height) = (55, 170, 58)`.
The default wall snapping changes positions by at most `0.02872546 m`; this is recorded as a geometry approximation, not counted as a new bug.
The accepted PMT sphere approximation, unmodelled full outer-detector geometry, and absolute calibration against real SK are not established by these geometric checks.

Among correctly selected spheres, float32 entry distances differ from the float64 oracle by up to roughly 1 cm for nearly grazing long flights.
A few rays retain positive overlap weight while the hard sphere-intersection flag is false because the discriminant is formed by large-term subtraction at `lucid/propagation/base.py:224-261`.
There were approximately one or two such rays per 20,000 rays in several samples.
This is small compared with the endpoint-lookup defect and is not presented as a major simulation bug.
A stable closest-approach sphere discriminant, as used in the independent oracle and prototype, is the direct numerical remedy.

The older `audit/wc_data_20261004/transport/geometry_results.json` compared against unbounded spheres and therefore labelled some intersections beyond the water boundary as missing hits.
Those old center-loss estimates are invalid as physical loss estimates.
The new bounded oracle finds zero center hit losses or wrong PMTs in 20,000 directions, agreeing with the independently executed central full DATA oracle in the end-to-end lane.
The old artifacts have been preserved.

## Executed commands and jobs

Batch job `39893930` ran on `milano` and established the minimal wrong-sensor case, then stopped because the reproducer originally asserted no returned hit rather than the observed later-sensor hit.
After correcting that assertion, all completed experiments ran inside persistent `milano` allocation `39894012` on `sdfmilan272` with 2 CPUs and 8 GB memory.
No GPU was used.
After the user's tightened execution instruction, all further file reads, edits, and shell commands were performed inside that allocation.

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
uv run --offline --no-project \
  --python audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/geometry/reproduce_geometry.py
uv run --offline --no-project \
  --python audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/geometry/reproduce_selection_fix.py
uv run --offline --no-project \
  --python audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/geometry/reproduce_production_population.py
```

The environment reports JAX `0.11.0` and NumPy `2.4.3`.
The installed CUDA plugin is incompatible with that JAX version, but it is disabled and every experiment explicitly uses the CPU platform.
