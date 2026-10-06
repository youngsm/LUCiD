"""Observer neutrality and whether first-step subtraction causes segment errors."""
import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import json
from pathlib import Path
import numpy as np
import uproot
from lucid.sources.root_reader import _read_photons_for_event

BASE=Path(__file__).resolve().parent
UP=BASE.parents[1]/"photonsim_physics"
dtype=np.dtype([("event","<i4"),("track","<i4"),("parent","<i4"),("cherenkov","<i4"),
                ("creation","<f8",(8,)),("record","<f8",(8,))])
rows=[]
for name in ("electron","muon","pion","gamma"):
    old=uproot.open(UP/f"{name}.root")
    observed=uproot.open(BASE/f"{name}_stream.root")
    obs=np.fromfile(BASE/f"{name}_stream.bin",dtype=dtype)
    evs=observed["OpticalPhotons"].arrays(["Segment_TrackID","Segment_Time","Photon_SegmentIndex","Segment_NCherenkov"],library="np")
    for eid in range(observed["OpticalPhotons"].num_entries):
        a=_read_photons_for_event(old["OpticalPhotonsRaw"],eid)
        b=_read_photons_for_event(observed["OpticalPhotonsRaw"],eid)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        ob=obs[obs["event"]==eid]
        seg_track=evs["Segment_TrackID"][eid]
        seg_time=evs["Segment_Time"][eid]
        stored=evs["Photon_SegmentIndex"][eid]
        joined_creation=np.full(len(ob),-1,np.int64)
        joined_record=np.full(len(ob),-1,np.int64)
        for parent in np.unique(ob["parent"]):
            sel=ob["parent"]==parent
            indices=np.flatnonzero(seg_track==parent)
            if not len(indices): continue
            for source,out in (("creation",joined_creation),("record",joined_record)):
                idx=np.searchsorted(seg_time[indices],ob[source][sel,6],side="right")-1
                idx=np.maximum(0,idx)
                out[sel]=indices[idx]
        np.testing.assert_array_equal(stored,joined_record)
        wanted=np.asarray(evs["Segment_NCherenkov"][eid],np.int64)
        stored_counts=np.bincount(stored[stored>=0],minlength=len(wanted))
        birth_counts=np.bincount(joined_creation[joined_creation>=0],minlength=len(wanted))
        rows.append({"scenario":name,"event":eid,"photons":len(ob),"baseline_identical":True,
            "time_rewind_changed_segment_assignments":int(np.count_nonzero(joined_creation!=stored)),
            "segment_L1_using_current_record_times":int(abs(stored_counts-wanted).sum()),
            "segment_L1_using_exact_creation_times":int(abs(birth_counts-wanted).sum())})
result={"slurm_job":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],
        "observer_neutrality_verified":True,"rows":rows}
(BASE/"baseline_timestamps_results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
