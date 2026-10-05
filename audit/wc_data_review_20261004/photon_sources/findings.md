# Photon source boundary and emission physics

Reference detector: `config/SK_WAND_geom_config.json` plus `config/SK_WAND_physics_config.json` at LUCiD `82f8d24`.
PhotonSim source: `ff73224`, preserved under `audit/wc_data_20261004/sources/PhotonSim-v1.0.0`.
All execution occurred in milano CPU allocation `39894010` on `sdfmilan269`, with zero GPUs.
All Python execution used `uv run` and the pre-existing CPU environment.
No production code or old audit artifacts were edited.

## Source material discrepancy: large Cherenkov light suppression near threshold

PhotonSim `src/DetectorConstruction.cc:88-107` registers water's phase index with ten values from `1.333` to `1.345` over 1.84 to 4.51 eV.
At 300 nm this actual table gives `n=1.34301477`, while the published Daimon and Masumura measured-water Sellmeier fit at 20 C gives `n=1.35920604` and active SK WCSim gives `n=1.35942091`.
At 400 nm the values are `1.33799884`, `1.34355668`, and `1.34418552`, respectively.
The two independent physical/reference descriptions agree closely.

The consequence follows directly from the Frank-Tamm yield `dN/(dx dE) = alpha/(hbar c) * max(1 - 1/(beta*n(E))**2, 0)`.
The emission band, QE curve, particle speed, and all other factors are held fixed.
For a 52 MeV kinetic-energy muon, the source predicts zero direct muon Cherenkov photons in the emitted band, versus 544 photons/m from measured water and 563 photons/m from SK WCSim.
For a 54 MeV muon, the source gives 710 photons/m versus 1777 and 1841 photons/m, respectively.
Weighting the same spectra with the actual SK QE curve gives 70.8 versus 164.9 and 174.3 potential PE/m, a 57 to 59 percent deficit.
These are emission yields per meter at fixed speed, not the whole detector hit count for a stopping muon event.
Delta electrons and Michel electrons must be counted separately in an end-to-end comparison.
At 1 GeV kinetic energy the QE-weighted deficit is about 1.1 percent, so this is localized near threshold and does not explain a campaign-wide 75 percent charge loss.

The 300 nm muon threshold rises from 50.35 to 52.63 MeV.
The relativistic cone angle at 300 nm shifts from 42.63 to 41.88 degrees.
At 400 nm it shifts from 41.90 to 41.64 degrees.

Sources:

- [Measured-water coefficients and paper attribution](https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/H2O/nk/Daimon-20.0C.yml).
- [Pinned active SK WCSim table](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimConstructMaterials.cc#L403-L445), installed as `RINDEX` at line 1853.
- [Geant4 Cherenkov implementation](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/electromagnetic/xrays/src/G4Cerenkov.cc).

Executed mathematical reproducer: `check_cherenkov_index.py`.
Results: `cherenkov_index_results.json` and `index-39894010.log`.
Exact command inside the allocation:

```bash
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/photon_sources/check_cherenkov_index.py
```

Smallest useful threshold test:

```python
import numpy as np
m_mu = 105.6583755
beta = np.sqrt(1 - (m_mu / (m_mu + 52))**2)
n_source, n_measured = 1.3430147716549474, 1.3592060384061295
assert beta * n_source < 1
assert beta * n_measured > 1
```

Minimal fix: replace PhotonSim's `ConstructWater()` phase-index values with calibrated water dispersion at SK water temperature across the emitted band, then regenerate the ROOT sources.
The active SK WCSim table or a measured-water dispersion formula can provide that table.
Changing only LUCiD's transport speed cannot restore absent photons or repair their emission angles.
An audit-only alternate binary and `DetectorConstruction_measured.cc` have been built, replacing only the ten water phase-index values with the measured formula at the same knots.
The actual Geant4 A/B comparison is complete with 100 fixed +z muon events at each kinetic energy and the same seeds, with muon decay and capture disabled in both variants.
The only changed physics input is the ten water phase-index values at the existing emission-band knots.
At 54 MeV the original source emits 149 primary-muon Cherenkov photons versus 947 in the measured-index variant, an 84.27 percent direct-primary deficit.
At 52 MeV it emits zero versus 148 primary-muon photons.
The deficit increases on a slowing track because the wrong threshold terminates direct emission too early.
Other photons from secondary electrons remain in both samples.
`compare_index_ab.py`, `index_ab_results.json`, and `index-ab-39894010.log` contain the executed comparison.

## External-origin photons have no source bounds mask

The production driver translates the entire raw photon list at `lucid/sources/event_generation.py:443-497`.
The complete list reaches `_trace_event_bucketed()` at `lucid/sources/event_builder.py:244-296`.
The only initial intensity mask at `lucid/simulation/simulator.py:819-820` is the active-versus-padding count, and propagation survival starts at unity at line 570.
Photons emitted beyond the modeled ID blacksheet by a translated charged track or shower therefore reach the transport kernel.
The first bounds check follows step deposition at lines 652-655.
The transport lane owns a full DATA-mode outside-origin reproducer, its detector-level consequence, and the minimal fix.
`check_outside_population.py` measures the source populations from actual Geant4 files using 20 events and 25 independent default-fiducial placements and coherent isotropic rotations per event.
For 1 GeV muons, 1.61 percent of emitted photon placements are outside the ID, and 8.6 percent of placements contain outside sources.
The largest fraction in this sample is 54.30 percent.
For 3 GeV muons the corresponding fractions are 17.71 percent and 46.2 percent, with a maximum of 85.65 percent.
The 3 GeV result illustrates escaping tracks and is above the default campaign's 2 GeV upper energy bound.
The 100 MeV electron sample has no outside sources in these 500 placements.
These are source fractions, not whole-cohort detected-charge biases.
The evidence is `outside_population_results.json` and `outside-population-39894010.log`.

`check_actual_source_mask.py` then runs the real 1 GeV muon event 16 at a default-fiducial vertex `(-5.3866086, 0.8363550, 16.2970123)` meters through actual SK_WAND DATA transport.
It uses per-segment hit mode, K=12, configured wavelength optics, absorption, scattering, reflections and QE, identical photon ordering, keys and 32768-photon padding.
The event has 203162 source photons, including 110320 beyond the ID.
The all-source simulation gives 5027.9976 PE on 543 sensors.
Applying only an initial exterior-source intensity veto gives 4930.9976 PE on 506 sensors.
Exterior sources add 97.0 PE, a 1.967 percent increase relative to the vetoed event, and light 37 extra PMTs.
Inside photon deposit records are bitwise identical between runs.
First times on all 506 shared lit sensors are unchanged.
No whole-cohort PE-bias estimate is claimed from this deliberately selected escaping event.
`actual_source_mask_results.json`, `actual_source_mask_sensor_arrays.npz`, and `actual-source-mask-39894010.log` retain the result.
The test broadcasts the existing `N` intensity comparison onto the photon axis and sets it to zero only for exterior origins, without editing production code.
The minimal production fix is to intersect the initial active photon mask with `detector.bounds_check(final_origins)` before setting `photon_intensities`.

## Executed boundary checks that passed

The actual SK_WAND geometry and production chunk wrapper were exercised with 513 source photons across three 256-photon chunks.
For this conservation check only, transport was transparent and sensors perfectly efficient to isolate the handoff.
All 513 photons produced exactly one detected record with the correct source ID, segment ID, and target PMT ID.
The total was 512.99988 PE from float32 reduction, a relative error of 2.4e-7.
Padding contributed no detected records, and no photon was lost or duplicated at chunk boundaries.
Input emission times ranged from 0 to 200000 ns.
The maximum residual against independent `emission_time + ray-sphere flight_distance / configured_light_speed` was 4.29e-6 ns.
The Python handoff itself does not impose the Geant4 10 us cutoff.

Twenty thousand `_random_rotation_matrix()` draws had rotated +z-axis means `(0.000789, -0.005336, -0.002729)` and second moments `(0.332280, 0.331380, 0.336339)`, consistent with zero means and one-third second moments.
A full `_rotate_event_raw()` check verified common rotation of photon origins/directions, segment endpoints/directions, and track origins/directions.
Reproducer: `check_source_boundary.py`.
Results: `source_boundary_results.json` and `cpu-39894010.log`.
The command above applies with the filename changed to `check_source_boundary.py`.
These checks establish tested conservation and transformation properties, not exact physical equivalence to SK.
Parametric `TrackSource`, SIREN yield, and fitted track `t0` models are not on the selected production `is_data=True` call path.
DATA mode uses individual Geant4 photon origins, directions, times, and wavelengths with unit initial photon intensity.

Minimal WC initial-source veto at the existing DATA intensity assignment:

```python
mask = jnp.arange(n_rays) < photon_data['N']
if not _is_volume:
    mask = mask & get_inside_detector_flag(final_origins)
photon_intensities = mask.astype(jnp.float32)
```

This applies to the closed surface detectors and preserves open-medium string-telescope source behavior.
The audit wrapper implements the same initial intensity effect through the existing `arange < N` expression without changing the production file.
The source-veto test used actual configured transport and sensor response before digitization and trigger selection.

To rerun the numerical and actual DATA checks, use the exact `uv run` invocation above with these filenames in the same directory:

- `check_cherenkov_index.py` for the source table against independent water dispersion.
- `compare_index_ab.py` for primary Cherenkov counts from the two real Geant4 ROOT source variants.
- `check_outside_population.py` for the escaping source populations and saved worst-placement arrays.
- `check_actual_source_mask.py` for the unchanged-key real muon DATA comparison.
- `check_source_boundary.py` for chunk conservation, timing preservation, and rotation properties.

The modified material binary is built by `build_index_variant.sh` inside an authorized CPU allocation.
The two measured-index source macros are `mu52_measured.mac` and `mu54_measured.mac`.
Their execution command inside the allocation is:

```bash
bash audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/photon_sources/PhotonSim_measured \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/photon_sources/mu54_measured.mac
```

The same command applies to `mu52_measured.mac`.
The original-source ROOT files and valid source runtime wrapper are retained by the Geant4 physics lane.
