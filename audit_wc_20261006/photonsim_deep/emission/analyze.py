"""Compare independent G4 creation snapshots against optical ROOT, every photon."""
import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import json
from pathlib import Path
import numpy as np
import uproot
from lucid.sources.root_reader import _read_photons_for_event

BASE=Path(__file__).resolve().parent
dtype=np.dtype([("event","<i4"),("track","<i4"),("parent","<i4"),("cherenkov","<i4"),
                ("creation","<f8",(8,)),("record","<f8",(8,))])
assert dtype.itemsize==144
rows=[]
for name in json.loads((BASE/"scenarios.json").read_text()):
    modes={}
    for mode in ("stream","buffer"):
        tag=f"{name}_{mode}"
        observers=np.fromfile(BASE/f"{tag}.bin",dtype=dtype)
        summary=[json.loads(s) for s in (BASE/f"{tag}.jsonl").read_text().splitlines()]
        f=uproot.open(BASE/f"{tag}.root")
        counts=f["OpticalPhotons"]["NOpticalPhotons"].array(library="np")
        track_data=f["OpticalPhotons"].arrays(["TrackInfo_TrackID","TrackInfo_ParentTrackID",
            "TrackInfo_CreatorProcess","Segment_TrackID","Photon_SegmentIndex"],library="np")
        chunk_starts=f["OpticalPhotonsRaw"]["ChunkStartID"].array(library="np")
        chunk_events=f["OpticalPhotonsRaw"]["EventID"].array(library="np")
        maxima=np.zeros(8)
        outputs=[]
        deflection_children=0
        for eid,s in enumerate(summary):
            ob=observers[observers["event"]==eid]
            assert s["created"]==s["recorded"]==len(ob)==counts[eid]
            assert s["unrecorded"]==s["duplicate_classifications"]==0
            track_ids=np.asarray(track_data["TrackInfo_TrackID"][eid])
            parents=np.asarray(track_data["TrackInfo_ParentTrackID"][eid])
            assert np.all(np.isin(parents[parents>0],track_ids))
            assert np.all(np.isin(ob["parent"],track_ids))
            psi=np.asarray(track_data["Photon_SegmentIndex"][eid])
            assert np.all(psi>=0)
            seg_owner=np.asarray(track_data["Segment_TrackID"][eid])
            np.testing.assert_array_equal(seg_owner[psi],ob["parent"])
            deflection_children+=sum(str(p).startswith("Deflection_") for p in track_data["TrackInfo_CreatorProcess"][eid])
            if len(ob):
                assert np.unique(ob["track"]).size==len(ob)
                maxima=np.maximum(maxima,np.max(abs(ob["record"]-ob["creation"]),axis=0))
            po,pd,pt,pw=_read_photons_for_event(f["OpticalPhotonsRaw"],eid)
            # Compare ROOT raw float arrays, including creation position in mm.
            chunks=f["OpticalPhotonsRaw"].arrays(["PhotonPosX","PhotonPosY","PhotonPosZ",
                "PhotonDirX","PhotonDirY","PhotonDirZ","PhotonTime","PhotonWavelength"],library="np")
            indices=np.flatnonzero(chunk_events==eid)
            indices=indices[np.argsort(chunk_starts[indices])]
            raw=[]
            for col in ("PhotonPosX","PhotonPosY","PhotonPosZ","PhotonDirX","PhotonDirY","PhotonDirZ","PhotonTime","PhotonWavelength"):
                raw.append(np.concatenate([chunks[col][i] for i in indices]) if indices.size else np.empty(0,np.float32))
            raw=np.stack(raw,axis=1)
            np.testing.assert_array_equal(raw,ob["record"].astype(np.float32))
            np.testing.assert_allclose(raw[:,:6],ob["creation"][:,:6],rtol=1e-7,atol=2e-4)
            np.testing.assert_allclose(raw[:,6:],ob["creation"][:,6:],rtol=1e-7,atol=2e-7)
            np.testing.assert_array_equal(po,raw[:,:3]/np.float32(1000))
            np.testing.assert_array_equal(pd,raw[:,3:6])
            np.testing.assert_array_equal(pt,raw[:,6])
            np.testing.assert_array_equal(pw,raw[:,7])
            if len(ob):
                assert np.all(ob["cherenkov"]==1)
                assert np.max(abs(np.linalg.norm(ob["creation"][:,3:6],axis=1)-1))<1e-12
                assert np.min(ob["creation"][:,7])>270 and np.max(ob["creation"][:,7])<675
            expected_starts=np.arange(0,len(ob),100000) if mode=="stream" else (np.array([0]) if len(ob) else np.empty(0,int))
            np.testing.assert_array_equal(chunk_starts[indices],expected_starts)
            outputs.append(raw)
        modes[mode]=outputs
        rows.append({"scenario":tag,"n_events":len(summary),"photons":int(counts.sum()),
            "raw_chunks":len(chunk_events),"creation_to_first_step_max_delta":maxima.tolist(),
            "root_exactly_equals_float32_record_payload":True,"root_reader_units_verified":True,
            "all_emitting_parents_registered":True,"all_track_ancestry_edges_registered":True,
            "assigned_segment_parent_matches_actual_parent":True,"deflection_children_checked":int(deflection_children),
            "event_observers":summary})
    assert len(modes["stream"])==len(modes["buffer"])
    for a,b in zip(modes["stream"],modes["buffer"]):
        np.testing.assert_array_equal(a,b)
result={"slurm_job":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],
        "chunking_AB_bitwise_equal":True,"total_photons_checked":sum(r["photons"] for r in rows),"scenarios":rows}
(BASE/"results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="scenarios"},indent=2))
for row in rows:
    print(row["scenario"],row["photons"],row["raw_chunks"],row["creation_to_first_step_max_delta"])
