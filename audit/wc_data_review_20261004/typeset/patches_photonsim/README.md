# PhotonSim candidate fixes

These patches target the separate PhotonSim repository at `ff73224a669fdc3598dc2993a8763223b11a3e5b`, pinned by the audited LUCiD container.
They do not target the LUCiD repository.
Only physics files are changed; the audit executable's headless entry-point changes are excluded.

| Patch | Tested correction | Regression |
| --- | --- | --- |
| `01_remove_late_track_cut.patch` | Remove the unconditional 10 us new-track kill | Matched-seed prompt/delayed gamma yield |
| `02_pion_endpoint.patch` | Current post-step pion position and global time | Parent endpoint equals replacement birth position |
| `03_genie_haar_rotation.patch` | Haar-density angle rejection with the existing uniform axis | Mean, hemisphere fraction and second moment |
| `04_thermal_neutron_elastic.patch` | HPT elastic constructor; retain original capture/inelastic physics | Independent pure-water thermal lifetime and late tail |

Each change was tested in isolated source variants in milano CPU allocations.
All four patches passed sequential `git apply --check` and actual application in a temporary source tree in milano job `39896290`.
The resulting physics files exactly match the experimental variants, as recorded in `validation.json`.
This is patch applicability validation, rather than an additional combined-physics simulation.

Apply from the pinned PhotonSim checkout inside an authorized allocation:

```bash
git apply --check /path/to/patches_photonsim/*.patch
git apply /path/to/patches_photonsim/*.patch
```

The capture-time benchmark depends on patch 01 as well as patch 04 because the original source time cut prevents an unbiased delayed-neutron measurement.
Retain `G4HadronPhysicsQGSP_BERT` and the tested Geant4 11.3 thermal datasets.
The existing `(G4_WATER, H)` material mapping selects the `h_water` thermal law in that version.
Broader HP-capture replacements were not validated and are not included.

Rebuild the candidate executable and regenerate the ROOT samples before running the physical source regressions.
Use the supplied audit macros and `run_photonsim.sh` runtime wrapper where the CVMFS ROOT module-path workaround is needed.
`test_source_physics.py` covers gamma yield, pion continuity and GENIE isotropy.
`test_neutron_capture.py` covers thermal water capture timing.
Their saved-original and saved-fixed ROOT products demonstrate fail-before/pass-after behavior; historical original products do not update when the executable changes.

No production repository was modified by this patch preparation.

The patches reproduce the tested code changes exactly and retain surrounding original comments.
Remove the obsolete late-track rationale and revise the pion stored-position comment during integration; those comments do not describe the corrected physics.
