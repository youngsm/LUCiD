"""Independent transformations/ancestry checks through the actual ROOT reader."""
from pathlib import Path
import json
import numpy as np

from lucid.sources.root_reader import _read_event_raw, read_particle_data_from_photonsim
from lucid.sources.event_builder import (
    _derive_views_from_segments, derive_particle_idx_per_track,
    derive_track_ancestor_and_interaction, _aggregate_from_photon_records,
)

BASE = Path(__file__).resolve().parent
ROOT = BASE / "reader_fixture.root"
raw = _read_event_raw(ROOT, 0)
# Independent data contracts: mm become m, MeV and ns are retained.
psi = np.array([0, 0, 1, 1, 2, 4, 8, 8, 10])
expected_pos = np.column_stack(((psi + 1), -(psi + 1) * .25, (psi + 1) * .5 + .005))
np.testing.assert_allclose(raw["photon_origins"], expected_pos, atol=1e-6)
np.testing.assert_allclose(raw["photon_wavelengths"], np.arange(350, 440, 10))
np.testing.assert_allclose(raw["photon_directions"], np.tile([.6, 0, .8], (9, 1)), atol=1e-7)
np.testing.assert_allclose(raw["photon_times"], [.01,.01,.01,.01,2300.01,2.11,.21,.21,.21], atol=1e-4)
np.testing.assert_allclose(raw["track_info_dict"][3]["position"], [3, -.75, 1.5])
assert raw["track_info_dict"][3]["time"] == 2300
assert raw["track_info_dict"][3]["energy"] == 30
assert raw["primary_energy"] == 2000
np.testing.assert_array_equal(raw["photon_segment_index_raw"], psi)

view = read_particle_data_from_photonsim(ROOT, 0)
# Physical ancestry expectations: prompt muon, its delayed daughter, electron
# shower, and two independently categorized pi0 decay gamma showers.
assert [p["genealogy"] for p in view["particles"]] == [[1],[1,3],[2],[7,8],[7,10]]
assert [p["photon_indices"] for p in view["particles"]] == [[0,1],[4],[2,3,5],[6,7],[8]]
assert list(view["meaningful_tracks"]) == [1,2,3,4,5,7,8,9,10,11]
np.testing.assert_array_equal(view["photon_segment_index"], [0,0,1,1,2,4,7,7,9])
np.testing.assert_allclose(view["segments"]["start_x"], [1,2,3,4,5,7,8,9,10,11])
np.testing.assert_array_equal(view["segments"]["group_id"], np.arange(10))

view["primary_to_interaction"] = {1:0,2:0,7:0}
ancestor, interaction = derive_track_ancestor_and_interaction(view)
np.testing.assert_array_equal(ancestor, [1,2,1,2,2,7,7,7,7,7])
np.testing.assert_array_equal(interaction, np.zeros(10))
np.testing.assert_array_equal(derive_particle_idx_per_track(view), [0,2,1,2,2,-1,3,3,4,4])

# Kernel-flat detection rows may reorder and duplicate global photon ids.
# Every deposit must retain its source particle and source segment.
gids = np.array([4,0,8,5,6,1,7,2,3,4])
records = {"photon_global_idx":gids,"qe_weight":np.ones(10),
    "qe_time":100 + raw["photon_times"][gids], "qe_time_reco":101 + raw["photon_times"][gids],
    "sensor_idx":np.arange(10)%2,"seg_idx_raw":psi[gids]}
with_records = _derive_views_from_segments(raw, records)["photon_records_filtered"]
np.testing.assert_array_equal(with_records["particle_idx"], [1,0,4,2,3,0,3,2,2,1])
np.testing.assert_array_equal(with_records["seg_idx_filtered"], np.array([0,0,1,1,2,4,7,7,9])[gids])
aggregate = _aggregate_from_photon_records(
    with_records["qe_weight"],with_records["qe_time"],with_records["sensor_idx"],
    with_records["seg_idx_filtered"],with_records["particle_idx"],5,2,
    with_records["qe_time_reco"])
assert aggregate["PE_per_particle"].sum() == 10
assert aggregate["segment_sensor_hits"]["PE"].sum() == 10

empty = read_particle_data_from_photonsim(ROOT, 1)
assert empty["n_particles"] == 0 and empty["segments"]["n_segments"] == 0
assert empty["photon_origins"].shape == (0,3)
result = {"status":"PASS", "input_photons":9,"kernel_flat_deposits":10,
    "derived_particles":5,"meaningful_tracks":10,
    "delayed_daughter_photon_time_ns":float(raw["photon_times"][4]),
    "contracts":["mm-to-m positions", "ns time retention", "MeV energy retention",
        "chunk ordering", "photon-segment mapping", "delayed daughter ownership",
        "shower inheritance", "pi0 gamma separation", "meaningful ancestor retention",
        "group offsets", "kernel-flat source mapping", "charge conservation", "empty events"]}
print(json.dumps(result,indent=2))
(BASE / "reader_results.json").write_text(json.dumps(result,indent=2)+"\n")
