"""A retained PhotonSim event keeps EventID even if its TTree entry changes."""
from pathlib import Path
import json
import numpy as np
import uproot
from lucid.sources.root_reader import read_photon_data_from_photonsim, _read_event_raw, _read_photons_for_event

base = Path(__file__).resolve().parent
path = base / "nonzero_event_id.root"
with uproot.open(path) as f:
    event_id = int(f["OpticalPhotons"]["EventID"].array(library="np")[0])
    expected = int(f["OpticalPhotons"]["NOpticalPhotons"].array(library="np")[0])
    physical_photons = _read_photons_for_event(f["OpticalPhotonsRaw"], event_id)[0]
public = read_photon_data_from_photonsim(path, 0)
raw = _read_event_raw(path, 0)
assert event_id == 7 and expected == 9 and len(physical_photons) == 9
assert len(public["photon_origins"]) == 0
assert len(raw["photon_origins"]) == 0
assert len(raw["photon_segment_index_raw"]) == 9
result = {"tree_entry":0,"stored_event_id":event_id,"expected_photons":expected,
          "public_reader_photons":len(public["photon_origins"]),
          "production_reader_photons":len(raw["photon_origins"]),
          "lost_fraction":1.0,"event_id_keyed_control_photons":len(physical_photons)}
print(json.dumps(result,indent=2))
(base / "event_id_results.json").write_text(json.dumps(result,indent=2)+"\n")
