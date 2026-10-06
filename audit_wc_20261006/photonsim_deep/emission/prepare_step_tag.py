import os
assert os.environ.get("SLURM_JOB_PARTITION") in ("milano","roma")
from pathlib import Path
import difflib
BASE=Path(__file__).resolve().parent
UP=BASE.parents[1]/"photonsim_physics"/"upstream"
dst=BASE/"step_tag"
dst.mkdir(exist_ok=True)
(dst/"EmissionStepInfo.hh").write_text((BASE/"EmissionStepInfo.hh").read_text())
originals={}
for name in ("DataManager.hh","DataManager.cc","SteppingAction.cc"):
    folder="include" if name.endswith(".hh") else "src"
    text=(UP/folder/name).read_text(); originals[name]=text
    if name=="DataManager.hh":
        text=text.replace('G4int immediateParentTrackID);','G4int immediateParentTrackID, G4int emissionStepIndex = -1);')
        text=text.replace('std::vector<G4int> fPhotonImmediateParentTrackID;',
                          'std::vector<G4int> fPhotonImmediateParentTrackID;\n    std::vector<G4int> fPhotonEmissionStepIndex;')
    elif name=="DataManager.cc":
        text='#include <cstdlib>\n#include <stdexcept>\n'+text
        text=text.replace('G4int immediateParentTrackID)\n{','G4int immediateParentTrackID, G4int emissionStepIndex)\n{')
        text=text.replace('fPhotonImmediateParentTrackID.push_back(immediateParentTrackID);',
                          'fPhotonImmediateParentTrackID.push_back(immediateParentTrackID);\n    fPhotonEmissionStepIndex.push_back(emissionStepIndex);')
        text=text.replace('fPhotonImmediateParentTrackID.clear();','fPhotonImmediateParentTrackID.clear();\n  fPhotonEmissionStepIndex.clear();')
        old='    // fPhotonTimeRetained[p] is in ns (set in AddOpticalPhoton via time/ns).'
        new='''    // Audit A/B switch: exact emission-step fix versus untouched time search.
    if (std::getenv("AUDIT_EXACT_STEP")) {
      const int exact = fPhotonEmissionStepIndex[p];
      if (exact < 0 || exact >= static_cast<int>(segs.size()))
        throw std::runtime_error("Invalid exact emission-step tag");
      fPhoton_SegmentIndex[p] = baseIt->second + exact;
      continue;
    }

'''+old
        assert text.count(old)==1; text=text.replace(old,new)
    else:
        text='#include "EmissionStepInfo.hh"\n'+text
        text=text.replace('void SteppingAction::UserSteppingAction(const G4Step* step)\n{',
            'void SteppingAction::UserSteppingAction(const G4Step* step)\n{\n  TagEmissionStep(step);')
        text=text.replace('                                   parentID);','                                   parentID, ReadAndRecordEmissionStep(track));')
    assert text!=originals[name]
    (dst/name).write_text(text)
diff=[]
for name,old in originals.items():
    diff.extend(difflib.unified_diff(old.splitlines(True),(dst/name).read_text().splitlines(True),
        fromfile="original/"+name,tofile="prototype/"+name))
(BASE/"step_tag_prototype.diff").write_text(''.join(diff))
for name in ("gamma","electron","pion","muon"):
    for mode in ("baseline","fixed"):
        macro=(BASE/f"{name}_stream.mac").read_text()
        macro=macro.replace(str(BASE/f"{name}_stream.root"),str(BASE/f"{name}_tag_{mode}.root"))
        (BASE/f"{name}_tag_{mode}.mac").write_text(macro)
