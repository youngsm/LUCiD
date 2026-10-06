"""Create isolated observer source copies and production-derived macros on CPU."""
import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
import json
from pathlib import Path
from lucid.production.generate_macro import generate_macro

BASE=Path(__file__).resolve().parent
UP=BASE.parents[1]/"photonsim_physics"/"upstream"
for filename in ("StackingAction.cc", "SteppingAction.cc", "EventAction.cc"):
    text=(UP/"src"/filename).read_text()
    text='#include "AuditEmission.hh"\n'+text
    if filename=="StackingAction.cc":
        old='G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* track)\n{'
        new=old+'\n  EmissionAudit::Created(track);'
    elif filename=="SteppingAction.cc":
        old='  // Optical photons: record on first step (creation) only.'
        new='  EmissionAudit::Recorded(step);\n'+old
    else:
        old='  fEdep = 0.;'
        new=old+'\n  EmissionAudit::Begin(event->GetEventID());'
        text=text.replace('  dataManager->EndEvent();','  dataManager->EndEvent();\n  EmissionAudit::End();')
    assert text.count(old)==1
    (BASE/filename).write_text(text.replace(old,new))

configs={
 "electron": {"particles":[{"type":"e-","energy_MeV":100}],"n":3},
 "muon": {"particles":[{"type":"mu-","energy_MeV":1000}],"n":2},
 "pion": {"particles":[{"type":"pi+","energy_MeV":1000}],"n":3},
 "positron": {"particles":[{"type":"e+","energy_MeV":2}],"n":3},
 "gamma": {"particles":[{"type":"gamma","energy_MeV":2.2}],"n":3},
 "dark": {"particles":[{"type":"e-","energy_MeV":0.01}],"n":2},
}
for name, conf in configs.items():
    cfg={"name":name,"material":"water","energy_distribution":"monoenergetic",
         "store_individual_photons":True,"disable_decays":False,"fixed_direction_z":True,
         "particles":conf["particles"]}
    for stream in (True,False):
        mode="stream" if stream else "buffer"
        tag=f"{name}_{mode}"
        text=generate_macro(config=cfg,output_root_file=str(BASE/f"{tag}.root"),
            n_events=conf["n"],photonsim_seeds=(314159,271828))
        text=text.replace('/photon/storeIndividual true','/photon/storeIndividual true\n/photon/storeProcessName true\n/photon/streamPhotonsChunked '+str(stream).lower())
        (BASE/f"{tag}.mac").write_text(text)
(BASE/"scenarios.json").write_text(json.dumps(configs,indent=2)+"\n")
