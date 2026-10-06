# ROOT-to-photon handoff audit: SK_WAND DATA

No new major production physics defect was confirmed in this lane at LUCiD
`20df309`. This is a bounded result from the checks below, not proof of complete
physical agreement with a real detector. Production files were not edited.

## Active path and code review

The selected production path is `run_job._run_lucid` ->
`generate_events_from_photonsim_particles` -> `root_reader._read_event_raw` ->
`run_event_process_pipeline` -> `_trace_event_bucketed` ->
`_simulation_with_data_impl` -> common propagation -> per-photon response ->
`gather_photon_deposits` -> digitization. `sources/event_io.py` contains legacy
drivers; its old per-particle count branch and old ROOT layout are not used by
this selected path.

Verified from executable statements:

- `root_reader.py:135-170` selects chunks, orders them by `ChunkStartID`, converts
  photon positions from mm to m, and retains direction, time in ns, and wavelength
  in nm. The accepted off-production EventID/entry-index limitation remains.
- `event_generation.py:454-475` rotates before translation, translates photon
  origins once in m, and translates raw segment endpoints by the same vector
  times 1000 because those endpoints are still in mm.
- `event_builder.py:252-256` converts photon origins m to cm at the kernel
  boundary, and `simulator.py:798` converts cm back to m. The previously reported
  factor-100 error is absent from this path.
- `simulator.py:847-867` uses input photon count for the padding mask and unit
  initial intensity. It directly uses input direction, emission time, and
  wavelength. Track energy, direction and vertex dummy parameters do not
  re-generate, re-weight, or re-time G4 DATA photons.
- `event_builder.py:265-344` forwards every active source photon across chunks,
  retains every kernel `(step,candidate,photon)` slab, drops padding, and maps
  slab rows to the correct global photon ID. The modulo mapping agrees with the
  actual `(K,candidate,photon)` reshape at `simulator.py:750-778`.
- `event_builder.py:698-707` joins particle and filtered-segment truth to those
  source IDs. Categorization cannot suppress the PE stream: production sets
  `compute_aggregate=False` (`event_generation.py:556`), and
  `gather_photon_deposits` selects only positive weight (`event_builder.py:767`),
  retaining orphan particle/segment sentinels. The legacy aggregate's sentinel
  filtering does not control the selected production sensor charge.
- `simulator.py:598-603` now initializes closed-detector photon survival from
  the transformed origin's bounds check. Translated outside-origin photons do
  not generate ID PE in the executed control.
- `event_generation.py:599-613` adds the event time offset downstream, after
  G4-frame propagation and digitization, without changing flight-time inputs.
  Its conversion to float64 precedes the addition.

## Executed independent checks

Script: [check_handoff.py](check_handoff.py).
Machine-readable output: [handoff_results.json](handoff_results.json).
Execution log: [cpu-39967547.log](cpu-39967547.log).
Slurm submission: [run.sbatch](run.sbatch).

These initial handoff diagnostics and LUCiD imports ran in milano CPU job
`39967547`, using four CPUs and no GPUs. The script asserts the partition before scientific
imports. No existing pytest suite was used as evidence.

| Independent invariant | Result |
| --- | --- |
| Real ROOT TTree containing reverse-ordered, noncontiguous chunks, with an unrelated event between them | Seven chosen photons recovered in the exact source order |
| Chosen dimensional positions/times/wavelengths through ROOT read | mm -> m correct; direction, ns and nm unchanged |
| Event with no matching chunks | Correct empty photon arrays |
| 523 uniquely tagged inputs in three padded chunks, with positive records only in a later kernel slab | No input lost or duplicated; padding absent; global IDs and segment IDs exact |
| Orphan truth attribution | Positive orphan PE retained for digitization |
| 10,000 uniform rotation draws and coherent photon/segment/track transforms | Passed moment and rigid-transform checks |
| Actual DATA kernel on SK_WAND geometry: 513 distinct radial near-sensor sources, 513 target PMTs, K=12, perfect optics to isolate the handoff | Exactly one detected record per input, correct source/segment/PMT IDs; 512.9994507 PE after float32 reductions |
| Arrival times against independent radial sphere entry distance divided by configured light speed, plus original emission time | Maximum residual 0.006282 ns over emission times 0--200,000 ns |
| 513 exterior sources passed through the same public host wrapper | Zero detected records |

The actual-beam diagnostic deliberately sets QE=1, no reflections, and
effectively infinite attenuation/scattering lengths. It verifies the units,
placement, timing and record conservation; it does not validate the SK optical
parameter calibration. Wavelength forwarding is tested as an exact host
invariant, while wavelength-dependent physical response is covered by the
optics lane.

## Additional absolute direct-light validation

[check_absolute_direct_light.py](check_absolute_direct_light.py) exercised
2.4 million photons through the actual DATA `per_segment` kernel with the
SK_WAND optical configuration. Slurm job `39969190` ran on milano, four CPUs,
zero GPUs. [Results](absolute_direct_light_results.json),
[log](absolute-39969190.log), [submission](run_absolute.sbatch).

There are 100,000 photons per point: six wavelengths (300, 350, 400, 450, 550,
600 nm), two water path lengths (5 and 20 m to PMT sphere entry), and two PMT
reflectances (0 and the configured 0.25). K=1 isolates direct light because a
scattered or reflected photon cannot return for another encounter. Each beam
aims normally at actual SK_WAND PMT 4111.

The independent probability is

`P(direct PE) = QE(lambda) * exp[-D * (alpha_abs + alpha_sym + alpha_asym)]`.

The scattering and blue-absorption coefficients come directly from
[Abe et al., Eqs. 14--17 and Table 3](https://arxiv.org/pdf/1307.0162), independently
implemented with the published numerical constants. Red absorption and QE use
the raw configured tabulated values. None of LUCiD's optical evaluation,
interpolation or attenuation helper functions supplies the expected result.
The represented photon origin/direction and the PMT sphere define D through an
independent float64 quadratic intersection. The probe wavelengths avoid the
452--476 nm splice blend.

| Wavelength | Water path | Expected direct PE | Observed R=0 | Observed R=0.25 |
| --- | --- | --- | --- | --- |
| 300 nm | 5 m | 481.24 | 502 | 462 |
| 350 nm | 5 m | 18610.14 | 18804 | 18658 |
| 400 nm | 5 m | 21700.54 | 21847 | 21819 |
| 450 nm | 5 m | 18187.96 | 18326 | 18225 |
| 550 nm | 5 m | 3477.35 | 3379 | 3470 |
| 600 nm | 5 m | 347.07 | 340 | 398 |
| 300 nm | 20 m | 322.16 | 317 | 340 |
| 350 nm | 20 m | 15371.09 | 15580 | 15421 |
| 400 nm | 20 m | 19162.76 | 19093 | 19125 |
| 450 nm | 20 m | 15597.42 | 15434 | 15546 |
| 550 nm | 20 m | 1452.30 | 1389 | 1435 |
| 600 nm | 20 m | 12.12 | 10 | 12 |

All 24 points agree within 2.739 binomial standard deviations; the combined
Pearson statistic is 28.944 for 24 degrees of freedom (p=0.222). The largest
R=0 versus R=0.25 count difference is 2.205 standard deviations. Each surviving
record has unit PE weight within float32 precision and belongs to the intended
PMT. This independently rejects the former additional reflection penalty,
double-QE suppression, and major water-attenuation unit errors for these
direct-light cases. The 600 nm/20 m endpoint has only about 12 expected PE and
therefore weak relative precision; its exact binomial comparison is retained
in the results.

Agreement verifies propagation of the supplied QE curve, not its absolute
experimental calibration or detector-wide angular/late-light response.

## Scope and uncertainties

No bug/fix pair is supplied because no new failing physical invariant was
found here. PhotonSim material and process physics are external to these
Python handoff functions and are audited by the upstream lane. Detailed
ancestry/containment and persisted charge/time truth are covered by the
production-integrity lane. SK_WAND uses only Cherenkov emission; scintillation
expansion was read for interactions with the common path but is not an active
SK_WAND emission source. The accepted legacy CLI, arbitrary EventID, displaced
input-vertex truth, late-light cuts, and supernova limitations are not reported
as new production defects.

These finite tests exclude concrete failure mechanisms in the chosen cases.
They cannot establish 100% physical correctness for all input files, optical
properties, source topologies or detector states.
