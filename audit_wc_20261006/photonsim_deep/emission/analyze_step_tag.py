import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
import json
from pathlib import Path
import numpy as np
import uproot
from lucid.sources.root_reader import _read_photons_for_event
BASE=Path(__file__).resolve().parent
dt=np.dtype([("event","<i4"),("track","<i4"),("parent","<i4"),("local_step","<i4"),("birth","<f8",(12,))])
assert dt.itemsize==112
rows=[]
for name in ("gamma","electron","pion","muon"):
    root={k:uproot.open(BASE/f"{name}_tag_{k}.root") for k in ("baseline","fixed")}
    data={k:np.fromfile(BASE/f"{name}_tag_{k}.bin",dtype=dt) for k in root}
    np.testing.assert_array_equal(data["baseline"],data["fixed"])
    trees={k:f["OpticalPhotons"].arrays(["Segment_TrackID","Segment_Time","Segment_NCherenkov","Photon_SegmentIndex"],library="np") for k,f in root.items()}
    for eid in range(root["baseline"]["OpticalPhotons"].num_entries):
        photons0=_read_photons_for_event(root["baseline"]["OpticalPhotonsRaw"],eid)
        photons1=_read_photons_for_event(root["fixed"]["OpticalPhotonsRaw"],eid)
        for a,b in zip(photons0,photons1): np.testing.assert_array_equal(a,b)
        ob=data["baseline"][data["baseline"]["event"]==eid]
        expected=np.zeros(len(ob),np.int64)
        seg_track=trees["baseline"]["Segment_TrackID"][eid]
        segtime=trees["baseline"]["Segment_Time"][eid]
        ncher=trees["baseline"]["Segment_NCherenkov"][eid]
        for pid in np.unique(ob["parent"]):
            sel=ob["parent"]==pid
            indices=np.flatnonzero(seg_track==pid)
            expected[sel]=indices[ob["local_step"][sel]]
        old=np.asarray(trees["baseline"]["Photon_SegmentIndex"][eid])
        new=np.asarray(trees["fixed"]["Photon_SegmentIndex"][eid])
        np.testing.assert_array_equal(new,expected)
        np.testing.assert_array_equal(np.bincount(new,minlength=len(ncher)),ncher)
        bad=old!=expected
        t=ob["birth"][:,0]; pre=ob["birth"][:,1]; post=ob["birth"][:,2]
        rows.append({"scenario":name,"event":eid,"photons":len(ob),
          "wrong_original_step":int(bad.sum()),"wrong_fixed_step":int(np.count_nonzero(new!=expected)),
          "photon_arrays_bitwise_unchanged":True,"step_bincount_matches_physics_NCherenkov":True,
          "wrong_step_birth_before_parent_pre":int(np.count_nonzero(bad&(t<pre-1e-14))),
          "wrong_step_birth_after_parent_post":int(np.count_nonzero(bad&(t>post+1e-14))),
          "wrong_step_new_minus_old_range":([int((expected-old)[bad].min()),int((expected-old)[bad].max())] if bad.any() else []),
          "birth_time_minus_parent_post_ns_max":float(np.max(t-post)) if len(t) else 0,
          "original_segment_bincount_L1":int(abs(np.bincount(old[old>=0],minlength=len(ncher))-ncher).sum())})
result={"slurm_job":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],"rows":rows,
    "total_photons":sum(r["photons"] for r in rows),"wrong_original":sum(r["wrong_original_step"] for r in rows),
    "wrong_fixed":sum(r["wrong_fixed_step"] for r in rows)}
(BASE/"step_tag_results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
