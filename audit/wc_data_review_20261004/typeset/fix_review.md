# Fix and scope review for the typeset audit

Reviewed against checkout `82f8d249fd3fcbff1a26dd8a94465dc07e5f3839` and PhotonSim `ff73224a669fdc3598dc2993a8763223b11a3e5b`.
All inspection and artifact edits in this review ran inside milano allocation `39896290`.
No production files were changed.
The companion `fix_snippets.json` distinguishes local replacements from integration sketches.

## Required wording and scope

- QE: the demonstrated normalization error follows incident-QE semantics used by the cited photocathode reference and WCSim.
  The supplied curve's absolute experimental provenance is not established.
  State that the direct response loses about 25% at R=0.25, rather than assigning that percentage to all event charge including reflected late light.
  Angular models need the actual encounter reflectance; a scalar correction alone is insufficient for those models.
- Thermal neutrons: 997 original events have a tagged nCapture gamma.
  Two other events have neutron-induced products, so 997 is not a nuclear disappearance probability.
  The 5.918% late fraction is conditional on those tagged captures.
  The successful constructor replacement validates the tested pure-water thermal capture clock and tail, rather than all neutron physics.
- Water index: the 84.27% figure is the primary Cherenkov yield of the 54 MeV muon benchmark after changing phase-index knots.
  Secondary-electron light remains, and this is not an 84% GeV-event or campaign-wide deficit.
  Temperature and selected calibration curve must be stated before a production material patch.
- Geometry: 1.07% is a lost-or-wrong-first-PMT rate conditional on oracle-defined true sphere hits in isotropic probes over the standard vertex volume.
  It is not total PE or energy bias.
  A later-PMT assignment and extra flight time also change intervening optical transport.
- Exterior light: the 97 false PE measurement is one translated physical muon event.
  Birth masking is correct for the opaque-ID abstraction used here.
  An external-light model would require explicit exterior geometry and transmission boundaries.
- Time cuts: source and electronics cuts are distinct and their losses cannot be added.
  Removing the electronics constant alone is an executed causal correction, but finite, explicit acquisition windows are needed to bound dark-noise generation.
  Do not label a modern neutron-tagging DAQ as calibrated by this SK-I preset.
- Pion fix: endpoint continuity is restored.
  Different replacement counts are expected when corrected trajectories alter later histories.
- GENIE fix: coherent uniform SO(3) rotation applies to configurations requesting isotropy.
  Beam and physical-supernova directions must retain their requested anisotropy.

## Physical regression tests versus measurement witnesses

A test that asserts the present wrong output is a measurement witness, even when its function name starts with `test_`.
The following helpers deliberately pass on the faulty checkout and must not be presented as the requested failing regressions:

- `pmt_detection/reproduce.py`: boundary incident-QE, pencil-beam, and negative-TTS functions.
- `electronics/repro_electronics.py`: delayed-burst and wrong-charge timing functions.
- `geometry/reproduce_geometry.py`: bulk measurement assertions and output generation.

The fail-before/pass-after suite must call production code and assert the physical invariant directly.
Do not call a measurement helper whose internal assertion still demands a 0.75 QE ratio after the correction.
Use a fixed statistical tolerance justified by sample size for stochastic response tests.
The minimal exterior DATA test and `water_optics/minimal_qe.py` already assert the desired physical result.
The source regression files assert photon time-translation invariance, pion endpoint continuity, GENIE isotropy, and the independent thermal capture clock.

Source tests read ROOT products.
Rebuild the candidate PhotonSim executable and regenerate those products before evaluating a source fix.
An unchanged historical ROOT file cannot become correct when C++ changes.
Explicitly label saved-original versus isolated-correction ROOT runs as fixture verification, and provide regeneration commands for a developer regression.

## Fix validation evidence

Scalar conditional-QE sampling passed an isolated 100,000-photon check within statistical uncertainty.
The exterior mask gave zero exterior response with an identical interior control.
The complete bounded PMT-selection prototype matched all 22 saved failing rays; it was a correctness prototype rather than a production acceleration implementation.
Source variants restored identical delayed-gamma yield, zero pion endpoint gaps, and isotropic GENIE moments.
The thermal-elastic-only variant restored the tested capture lifetime and tail; broader HP-capture variants remained biased.
Electronics isolated changes restored delayed PE and the second gate, and charge-dependent timing matched the selected WCSim formula.
The original-wavelength QE lookup passed the two zero-QE spectral endpoints.

The compact snippets are suggested changes, not merged production fixes.
The scalar QE snippet requires calibration validation over all configured wavelengths and per-PMT corrections.
The complete ray-search and DAQ-window snippets state their assumptions and required integration explicitly.
