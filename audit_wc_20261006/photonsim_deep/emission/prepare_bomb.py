"""Use the exact production GeV/01 bomb settings in the emission observer."""
import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import json
from pathlib import Path
from lucid.production.generate_macro import generate_macro
BASE=Path(__file__).resolve().parent
repo=BASE.parents[2]
cfg=json.loads((repo/"lucid/production/configs/GeV/01_pbomb.json").read_text())
for stream in (True,False):
    mode="stream" if stream else "buffer"
    tag=f"bomb_{mode}"
    text=generate_macro(config=cfg,output_root_file=str(BASE/f"{tag}.root"),
        n_events=3,photonsim_seeds=(314159,271828))
    text=text.replace('/photon/storeIndividual true','/photon/storeIndividual true\n/photon/storeProcessName true\n/photon/streamPhotonsChunked '+str(stream).lower())
    (BASE/f"{tag}.mac").write_text(text)
scenarios=json.loads((BASE/"scenarios.json").read_text())
scenarios["bomb"]={"production_config":"lucid/production/configs/GeV/01_pbomb.json","n":3}
(BASE/"scenarios.json").write_text(json.dumps(scenarios,indent=2)+"\n")
