"""Execute with uv on a milano/roma allocation; no production edits.

prepare creates a ROOT input for the real PhotonSim GENIE path.
analyze inspects the real PhotonSim ROOT output of the accompanying macros.
"""
from pathlib import Path
import json
import sys

import numpy as np
import uproot


BASE = Path(__file__).resolve().parent


def prepare():
    n = 20_000
    pdg = np.tile(np.array([14, 14], np.int32), (n, 1))
    status = np.tile(np.array([0, 1], np.int32), (n, 1))
    p4 = np.zeros((n, 2, 4), np.float64)
    p4[:, :, 2:] = 0.001  # +z, one MeV incoming and outgoing neutrino.
    with uproot.recreate(BASE / "forward_neutrinos.gtrac.root") as f:
        t = f.mktree("gRooTracker", {
            "StdHepN": "int32", "StdHepPdg": "2 * int32",
            "StdHepStatus": "2 * int32", "StdHepP4": "2 * 4 * float64",
        })
        t.extend({"StdHepN": np.full(n, 2, np.int32),
                  "StdHepPdg": pdg, "StdHepStatus": status, "StdHepP4": p4})
    (BASE / "genie_isotropic.mac").write_text(
        f"/output/filename genie_isotropic.root\n/run/initialize\n"
        f"/random/setSeeds 314159 271828\n/photon/storeIndividual true\n"
        f"/gun/clearPrimaries\n/gun/genieInput {BASE / 'forward_neutrinos.gtrac.root'}\n"
        f"/gun/genieIsotropic true\n/run/beamOn {n}\n")


def analyze():
    result = {}
    with uproot.open(BASE / "genie_isotropic.root") as f:
        z = np.array([a[0] for a in f["OpticalPhotons"]["TrackInfo_DirZ"].array(library="np")])
        result["genie_isotropic"] = {"n": len(z), "mean_cos_theta": float(z.mean()),
            "fraction_positive_z": float(np.mean(z > 0)),
            "expected_mean": 0., "expected_positive_z": .5,
            "histogram_10_bins": np.histogram(z, bins=np.linspace(-1, 1, 11))[0].tolist()}
        assert abs(z.mean()) > 0.2, "Angular-bias reproducer no longer fails isotropy"
    counts = {}
    for suffix in ("prompt", "delayed"):
        with uproot.open(BASE / f"gamma_{suffix}.root") as f:
            counts[suffix] = f["OpticalPhotons"]["NOpticalPhotons"].array(library="np")
    result["late_gamma"] = {"prompt_total": int(counts["prompt"].sum()),
        "delayed_total": int(counts["delayed"].sum()),
        "prompt_mean": float(counts["prompt"].mean()),
        "delayed_mean": float(counts["delayed"].mean()),
        "photon_loss_fraction": float(1. - counts["delayed"].sum() / counts["prompt"].sum()),
        "prompt_per_event": counts["prompt"].tolist(), "delayed_per_event": counts["delayed"].tolist()}
    assert counts["delayed"].sum() < counts["prompt"].sum() * .5
    gaps = []
    with uproot.open(BASE / "pion_deflection.root") as f:
        t = f["OpticalPhotons"]
        fields = ["TrackInfo_TrackID", "TrackInfo_ParentTrackID", "TrackInfo_CreatorProcess",
                  "TrackInfo_PosX", "TrackInfo_PosY", "TrackInfo_PosZ", "TrackInfo_Time",
                  "Segment_TrackID", "Segment_EndX", "Segment_EndY", "Segment_EndZ", "Segment_Time"]
        data = t.arrays(fields, library="np")
        for ev in range(t.num_entries):
            for row, process in enumerate(data["TrackInfo_CreatorProcess"][ev]):
                if not str(process).startswith("Deflection_"):
                    continue
                parent = data["TrackInfo_ParentTrackID"][ev][row]
                parent_segments = np.flatnonzero(data["Segment_TrackID"][ev] == parent)
                assert len(parent_segments)
                final = parent_segments[-1]
                endpoint = np.array([data[f"Segment_End{k}"][ev][final] for k in "XYZ"])
                newborn = np.array([data[f"TrackInfo_Pos{k}"][ev][row] for k in "XYZ"])
                gap_mm = np.linalg.norm(newborn - endpoint)
                gaps.append({"event": ev, "track": int(data["TrackInfo_TrackID"][ev][row]),
                    "parent": int(parent), "process": str(process), "gap_mm": float(gap_mm),
                    "parent_endpoint_mm": endpoint.tolist(), "replacement_vertex_mm": newborn.tolist()})
    result["pion_deflection"] = {"n_replacements": len(gaps),
        "max_gap_mm": max(g["gap_mm"] for g in gaps) if gaps else None,
        "median_gap_mm": float(np.median([g["gap_mm"] for g in gaps])) if gaps else None,
        "replacements": gaps}
    assert gaps and max(g["gap_mm"] for g in gaps) > 1e-3
    print(json.dumps(result, indent=2))
    (BASE / "source_results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    {"prepare": prepare, "analyze": analyze}[sys.argv[1]]()
