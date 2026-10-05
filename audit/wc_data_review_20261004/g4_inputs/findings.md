# ROOT input and particle association review

No new large physical error was found in the ordinary SK_WAND production ROOT conversion or Python Cherenkov genealogy path.
One conditional input defect was reproduced: retaining or reordering ROOT entries without renumbering stored EventID can silently remove every photon from an event.
This defect is separate from the previously fixed factor-100 origin conversion.

## Confirmed conditional defect: tree entry index is mistaken for EventID

Affected call paths are `root_reader.read_photon_data_from_photonsim` and `root_reader._read_event_raw`, followed by `_read_photons_for_event`.
The chunk selector at `lucid/sources/root_reader.py:137` compares raw `EventID` with the requested TTree entry index.
The public reader passes `entry_index` at line 253, and the production reader passes it at line 366.
Neither call reads the main-tree EventID to identify that entry's photons.

The real-schema TTree fixture `nonzero_event_id.root` stores event 7 in main-tree entry 0, with nine photons and nine valid photon-segment indices.
Both current reader APIs return zero photon origins for entry 0, while passing the stored EventID directly into the chunk reader returns all nine photons.
The measured photon loss is 100%.
The nine `Photon_SegmentIndex` values are still returned by `_read_event_raw`, so the metadata and photon array lengths also disagree.

Physical expectation follows from preserving the identity and contents of a simulated event under a change of file entry order.
Reading the retained event must return the same nine emitted photons, with the same directions, positions, wavelengths, and times.
The control reads the actual raw chunks using their shared stored EventID, independently of the affected entry-to-event selection.

Minimal failing test, using the supplied fixture:

```python
from lucid.sources.root_reader import read_photon_data_from_photonsim
x = read_photon_data_from_photonsim("nonzero_event_id.root", 0)
assert len(x["photon_origins"]) == 9  # Current result is 0.
```

Minimal suggested fix:

```python
# In both reader paths, include EventID in the main-tree branches.
event_id = int(tree_data["EventID"][0])
photon_positions, photon_directions, photon_times, photon_wavelengths = \
    _read_photons_for_event(raw_tree, event_id)
```

The ordinary SK_WAND production macro runs one serial `beamOn` sequence and creates entries whose EventID equals the entry index.
That ordinary production path is unaffected by this specific defect.
External files containing retained, skipped, or reordered events are affected.
Multiple runs with repeated EventID need a unique run/event key or an explicit rejection in addition to this minimal fix.

Executed in milano allocation 39894049 on `sdfmilan272`, with zero GPUs.
`reproduce_event_id.py` records the observed failure and the EventID-keyed control in `event_id_results.json`.
The execution log is `event-id-39894049.log`.
The complete repeatable command is `sbatch audit/wc_data_review_20261004/g4_inputs/run_reader.sbatch` from the repository root.

## Executed checks for the ordinary input path

`make_reader_fixture.cc` writes genuine ROOT TTree branches using PhotonSim's vector types, including `vector<string>` creator processes.
The fixture contains a prompt muon, its daughter at 2300 ns, an electron shower with a gamma-to-electron chain, and a pi0 with two independently categorized gamma showers.
It also contains a zero-photon neutron primary and a completely empty second event.
The optical chunks are written out of order to test selection by ChunkStartID.
Expected units and ancestry are specified directly in `check_reader.py`, independently of the transformations being tested.

All of the following checks passed:

- Millimetres become metres for photons, track vertices, and segment endpoints.
- MeV energies and ns times are retained.
- Nine input photons keep their expected source segment indices after chunk stitching.
- Five physical category branches receive the expected photon indices.
- The 2300 ns daughter remains a distinct delayed branch of the prompt muon.
- Electron shower descendants remain owned by the electron primary.
- The two pi0 gamma showers remain separate categorized branches.
- Zero-photon ancestors needed for genealogy are retained.
- Segment filtering preserves the expected track order and remapped indices.
- Kernel-flat records preserve source ancestry even when their global photon indices are reordered or repeated.
- Ten supplied detected deposits sum to ten deposits in both particle and segment aggregations.
- The empty event returns no particles, segments, or photons.

The measured delayed photon time is 2300.010009765625 ns, consistent with float32 serialization of the 2300.01 ns input.
The results are in `reader_results.json` and `check-39894049.log`.
These are finite transformation and ancestry checks, not a proof that Geant4's generated physics or every possible event topology is exact.

## Assumptions checked against actual C++

PhotonSim `DataManager.cc:412` iterates a `std::map` of tracks, and lines 419-436 write each track's complete segment block before advancing the offset.
This guarantees the sorted, contiguous `Segment_TrackID` blocks required by `particle_categorization.py:390-392` and the filtered offsets used in `event_builder.py:522`.
PhotonSim's serial run manager is selected at `PhotonSim.cc:81`.
The input photon positions and times are written as mm and ns in `DataManager.cc:546-551`, and energies are written as MeV in the track metadata at line 491.
The reader's present conversion follows those concrete writes.

## Environment repair and exclusions

Initial C++ ROOT fixture attempts with CERN LCG ROOT 6.34.02 and 6.32.02 produced malformed TTrees because their PCM files referenced an unavailable absolute build path.
Those failures were not counted as LUCiD bugs or physical evidence.
The final fixture was regenerated with `geant4_physics/run_photonsim.sh`, which uses an audit-only Apptainer root filesystem and binds the actual ROOT library directory onto that original PCM build path.
The valid generation log is `fixture-39894049-valid.log`, with no interpreter errors.
The cancelled initial batch job was 39894040; the executed checks ran in the persistent compute allocation 39894049 instead.
No production source was edited.

The intentional Cherenkov-only meaningful-track view omits unrelated non-light-producing tracks.
Whether that omission is appropriate for complete interaction truth is a separate writer/metadata review, coordinated with the production_io lane.
Delayed-particle truncation, charged-pion replacement, and Geant4 emission physics are owned by the geant4_physics lane.
Translation, rotation, and source emission-frame handling are owned by the photon_sources lane.
