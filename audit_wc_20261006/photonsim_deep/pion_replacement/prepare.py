"""Audit-only source generation; run inside milano/roma."""
import os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
out=Path(__file__).resolve().parent
upstream=out.parents[1]/"photonsim_physics"/"upstream"
step=(upstream/"src/SteppingAction.cc").read_text()
step=step.replace('#include "SteppingAction.hh"','#include "SteppingAction.hh"\n#include "PionAudit.hh"')
step=step.replace('  // Get the track and particle information','  PionAudit::Step(step);\n  // Get the track and particle information')
step=step.replace('            // Kill current track','            PionAudit::Replacement(track, static_cast<int>(originalStatus));\n            // Kill current track')
(out/"SteppingAction_on.cc").write_text(step)
assert step.count('if (angle > 5.0 * deg) {')==1
(out/"SteppingAction_off.cc").write_text(step.replace('if (angle > 5.0 * deg) {','if (false && angle > 5.0 * deg) {'))
event=(upstream/"src/EventAction.cc").read_text()
event=event.replace('#include "EventAction.hh"','#include "EventAction.hh"\n#include "PionAudit.hh"')
event=event.replace('  fEdep = 0.;','  PionAudit::Begin(event->GetEventID());\n  fEdep = 0.;')
event=event.replace('  // accumulate statistics in run action','  PionAudit::End();\n  // accumulate statistics in run action')
(out/"EventAction_audit.cc").write_text(event)
for mode in ("on","off"):
    for pdg,name in [(211,"pi+"),(-211,"pi-")]:
        for energy,count in [(2,500),(200,300),(1000,300),(2000,300)]:
            stem=f'{mode}_{pdg}_{energy}'
            macro=f'''/output/filename {out/stem}.root
/run/initialize
/random/setSeeds 516231 272391
/photon/storeIndividual false
/gun/clearPrimaries
/gun/addPrimary {name} {energy} MeV
/gun/position 0 0 0 m
/gun/direction 0 0 1
/run/beamOn {count}
'''
            (out/f'{stem}.mac').write_text(macro)
