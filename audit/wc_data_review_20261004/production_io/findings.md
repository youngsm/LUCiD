# Production I/O audit findings

The reference is SK_WAND geometry and physics at the audited checkout.
No production source was edited.
CPU job 39894044 ran on milano, and follow-up measurements ran inside milano allocation 39894052 on sdfmilan269.
All Python commands used uv.
No GPU was requested.

## Confirmed conditional defect: supernova chunk boundaries duplicate simultaneous digits and dark noise

The supernova driver groups interactions using only their interaction start times, with a gap cutoff of max(integration_window, trigger_window + padding).
For SK_WAND this cutoff is 800 ns.
The arrival times of photons can be delayed relative to the interaction time by much more than 800 ns, and the current digitizer retains arrivals up to 100000 ns.
Two interactions in different chunks can therefore contribute light at the same PMT at the same absolute time.
Each chunk is digitized separately before concatenation, so these deposits do not share the physical electronics integration window.
The independent dark samples for overlapping arrival spans are also concatenated, adding the dark rate more than once over the same detector time interval.

Affected call path: generate_events_from_photonsim_supernova -> _group_interactions_by_gap -> _merge_pileup_streams -> digitize_and_decompose -> _concat_triggered_chunks.
The boundary is chosen at lucid/sources/event_generation.py:1596-1600 using the helper at lines 1338-1358.
Separate digitization happens at lines 1648-1654, and concatenation happens at line 1683.
Dark is generated independently for every merge at lucid/simulation/digitizer.py:524-537.

Independent expectation: simultaneous deposits on the same PMT must be integrated into one digit by the configured 200 ns integrator.
For two one-photoelectron deposits at 1100 ns, the unsmeared integrated charge is 2 PE, and there is one digit.
A Poisson dark process at rate r over one time span of width T has expected photoelectrons n_PMT * r * T.
These expectations do not depend on LUCiD comments or on any existing test.

Executed fixture: interaction t0 values are 0 and 1000 ns, and their photon arrival delays are 1100 and 100 ns.
Both photons therefore arrive at 1100 ns.
The current gap helper returns two chunks.
The current split pipeline writes two simultaneous 1 PE digits and two overlapping readout windows.
Pooling the same deposits through the same real _merge_pileup_streams API gives one 2 PE digit.
The defect doubles digit multiplicity in this fixture without changing total true charge.

The dark measurement used the actual SK_WAND sensor count of 11096 and the configured 4.2 kHz rate.
With three physics deposits per interaction and a 500 ns pad on each side, both chunks populate the same 1000 ns interval.
Across 256 independent ensembles, split processing produced 93.984375 dark PE on average, versus 46.484375 for one pooled processing pass.
The observed ratio was 2.0218487.
The independent one-span expectation is 11096 * 4.2e-6 * 1000 = 46.6032 dark PE.
Readout smearing was disabled to isolate this corruption of the event content.

Reproducers: reproduce_host.py:test_sn_chunk_overlap and reproduce_sn_dark.py.
Evidence: host-39894044.log, host_results.json, sn_dark.log, and sn_dark_results.json.
The HDF5 files split_sensor.h5 and split_labl.h5 also retain the duplicate digits and windows through the real writer-reader boundary.

Smallest useful reproducer, using the helpers defined in reproduce_host.py:

```python
streams = [stream(1, 0.0, 1100.0), stream(2, 1000.0, 100.0)]
groups = _group_interactions_by_gap(np.array([0.0, 1000.0]), 800.0)
assert groups == [[0], [1]]
assert [merge([s])["sensor_digits"]["PE"].tolist() for s in streams] == [[1.0], [1.0]]
assert merge(streams)["sensor_digits"]["PE"].tolist() == [2.0]
```

Exact reproduction inside a milano allocation:

```bash
export JAX_PLATFORMS=cpu
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit/wc_data_review_20261004/production_io/reproduce_host.py
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit/wc_data_review_20261004/production_io/reproduce_sn_dark.py
```

Minimal suggested fix: join chunks whenever their actual photon arrival/readout intervals overlap, including integration, deadtime and trigger padding, before either digitization or dark generation.
A conservative implementation can first extend the interaction-gap cutoff to include the maximum retained photon delay and all readout pads, though using actual deposit intervals is more precise.
Dark should be generated once over the union of retained readout intervals.
Merely merging the stored window labels after digitization does not repair the shared-charge or duplicate-dark defect.

Applicability: this is conditional on overlapping delayed light in the supernova driver.
It does not affect ordinary single-vertex GeV events processed by generate_events_from_photonsim_particles.
No rate or bulk percentage of corruption in the shipped 50 kpc burst has been measured, so the local factor-of-two result should not be presented as a whole-dataset factor-of-two defect.

## Checked host behavior

The real merge, writer and reader APIs preserve two different primary ancestors and interaction indices.
The same check preserved an exact fractional arrival time of 10000000080.375 ns in float64.
Simultaneous cross-vertex deposits were correctly pooled into one 2 PE digit when processed together.
These are limited executed invariants, not a claim of universal physical correctness.

run_job forwards the selected physics-config digitizer and trigger to the event generator.
The GENIE runner returns the actual number of RooTracker entries, so the stale variable name total_primaries does not currently overcount beamOn.
The earliest segment direction saved by the writer comes from the Geant4 pre-step momentum, according to the independently inspected PhotonSim call site.

## Confirmed secondary defect: direct CLI silently disables configured SK_WAND readout

A complete real PhotonSim ROOT-to-HDF5 execution confirmed that the direct CLI ignores configured readout.
The earlier failures were environment problems and were resolved using the Geant4 lane's audit-only Apptainer ROOT namespace workaround.
The successful repro used the unmodified pinned PhotonSim physics binary and a real 0.1 MeV electron event.
PhotonSim produced one ROOT entry, eight particle steps, and zero Cherenkov photons.
The explicit CLI arguments selected config/SK_WAND_geom_config.json and config/SK_WAND_physics_config.json.

The direct CLI completed and wrote an event with digitizer_model=basic, trigger=none, n_events=1 and n_hits=0.
The real run_job._run_lucid API consumed the same ROOT sample and selected digitizer_model=ski, trigger=sliding_window, dropping that untriggered event and writing n_events=0.
The source defect is at lucid/production/generate_events_with_particles.py:137-151, where digitizer and trigger are not forwarded to generate_events_from_photonsim_particles.
The defaults resolve to basic electronics and no trigger at lucid/sources/event_generation.py:173-183 and 302-303.
run_job forwards both correctly at lucid/production/run_job.py:393-395.

An additional executed real digitizer test quantified the readout consequence with two detected photoelectrons on one PMT at 0 and 2000 ns.
The basic model used by the direct CLI returned one 2 PE digit at 0 ns.
The selected SK_WAND ski model returned two 1 PE digits, at 0 and 2000 ns.
Thus the direct CLI can remove delayed hit timestamps and collapse independent PMT pulses, in addition to disabling configured dark noise, discriminator and PMT response.

Independent expectation: choosing an explicit detector physics configuration must use the electronics and readout trigger in that configuration.
The configured 200 ns integrator must distinguish arrivals separated by 2000 ns.
The configured N200 threshold of 25 cannot be crossed by an event with no recorded hit.

Reproducers and evidence: subthreshold.mac, subthreshold.root, subthreshold_photonsim.log, cli.log, reproduce_cli_comparison.py, cli_comparison.log, cli_comparison_results.json, reproduce_cli_readout.py and cli_readout_results.json.
All successful commands ran within milano allocation 39894052.

Minimal test after running the actual CLI command:

```python
with h5py.File("cli_output/sensor/wc_sensor_0000.h5") as f:
    assert f["config"].attrs["digitizer_model"] == "basic"  # expected ski
    assert f["config"].attrs["trigger"] == "none"           # expected sliding_window
    assert f["config"].attrs["n_events"] == 1              # actual 100 keV e- has zero hits
```

Exact successful CLI invocation inside the repository and milano allocation:

```bash
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python -m lucid.production.generate_events_with_particles --root-file audit/wc_data_review_20261004/production_io/subthreshold.root --config config/SK_WAND_geom_config.json --physics-config config/SK_WAND_physics_config.json --output audit/wc_data_review_20261004/production_io/cli_output --dataset-name SK_WAND_direct_cli_audit --master-seed 19 --n-events 1 --apply-smearing
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit/wc_data_review_20261004/production_io/reproduce_cli_comparison.py
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit/wc_data_review_20261004/production_io/reproduce_cli_readout.py
```

Minimal suggested fix: load and forward the physics file's digitizer and trigger blocks at the CLI's generator call.
The existing run_job helpers can be reused directly:

```python
digitizer=_read_digitizer_cfg({}, args.physics_config),
trigger=_read_trigger_cfg({}, args.physics_config),
```

Applicability: every direct generate_events_with_particles conversion is affected when its selected physics configuration specifies these readout blocks.
The ordinary run_job SK_WAND production path is unaffected.

## Confirmed secondary defect: nonzero input vertices are saved as zero source truth

The data writer preserves displaced photon positions and particle steps, but records the interaction vertex as the translation displacement alone.
When apply_translation=False, it sets that displacement to zero and writes the vertex as zero regardless of the actual input vertex.
This gives internally inconsistent event truth and can corrupt vertex-reconstruction targets.

A real 100 MeV PhotonSim electron event produced 23829 Cherenkov photons.
The audit-only C++ translate_root.cc copied its actual TTree schema, including vector<string>, and rigidly shifted every photon position, particle initial position, and segment endpoint by (5000,-2000,3000) mm.
It preserved energies, times, directions, counts, segment ownership, ancestry and all other branches.
The rigidly displaced event was then consumed through the actual SK_WAND run_job._run_lucid path.

The physical source and first saved step both were (5,-2,3) m.
The saved labl per_interaction vertex was (0,0,0) m.
The source-vertex error was 6.164414 m.
This test uses a translated real Geant4 shower and the real writer, without mocking the source, transport, readout or HDF5 APIs.

Affected call path: generate_events_from_photonsim_particles -> build_interaction_metadata -> save_labl_event.
The displacement becomes zero at lucid/sources/event_generation.py:423-432 and is passed as vertex_xyz at lines 645-647.
The shared pile-up/supernova stream path has the same metadata assumption at lines 926-929.

Independent expectation: the interaction vertex must equal the initial physical position of its parent_id==0 primary in the same detector coordinate frame as the photons and particle steps.
For a common source vertex, a rigid displacement d gives a vertex v_input+d.

Reproducer: displaced.mac, translate_root.cc and reproduce_vertex_truth.py.
Evidence: displaced_photonsim.log, translate_root.log, vertex_truth.log and vertex_truth_results.json.
The output is under displaced_output/{sensor,hits,step,labl}.
The unmodified pinned Geant4 binary and its ROOT namespace wrapper belong to the geant4_physics audit lane.
All execution took place in milano allocation 39894052.

Minimal assertion against the executed output:

```python
raw = _read_event_raw("displaced_shifted.root", 0)
source = next(t["position"] for t in raw["track_info_dict"].values() if t["parent_id"] == 0)
with h5py.File("displaced_output/labl/wc_labl_0000.h5") as f:
    pi = f["event_000/per_interaction"]
    saved = np.array([pi["vertex_" + axis][0] for axis in "xyz"])
assert np.allclose(source, [5.0, -2.0, 3.0])
assert np.array_equal(saved, [0.0, 0.0, 0.0])  # incorrect stored truth
```

The complete minimal executed assertion is in reproduce_vertex_truth.py.
Its exact Python command inside milano is:

```bash
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit/wc_data_review_20261004/production_io/reproduce_vertex_truth.py
```

Minimal suggested fix: derive the common input primary vertex from the raw TrackInfo positions after any rotation, add the displacement actually applied to photon and step positions, and pass that physical position to build_interaction_metadata.
If apply_translation is intended to place the source at a sampled absolute vertex, use displacement = sampled_vertex - input_vertex and save sampled_vertex.
Validate that all primaries assigned to the same source interaction share the same vertex, rather than silently substituting zero.
Apply the same correction to _simulate_interaction_stream.

Applicability: external or preprocessed Geant4 input whose source vertex is nonzero.
Current PhotonSim production intentionally begins every gun/bomb/GENIE event at zero before the LUCiD fiducial translation, so the standard SK_WAND production configuration is unaffected.
The photons and step positions in the reproduced event are correct; the confirmed corruption is the saved source-vertex truth.
