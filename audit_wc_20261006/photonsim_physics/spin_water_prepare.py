"""Audit-only effective water depolarization variant; run on milano/roma."""
from pathlib import Path
base = Path(__file__).resolve().parent
source = (base/'upstream/src/SteppingAction.cc').read_text()
source = source.replace('#include "G4SteppingManager.hh"', '#include "G4SteppingManager.hh"\n#include "Randomize.hh"')
needle = '  DataManager* dataManager = DataManager::GetInstance();'
addition = '''  // Effective ensemble retention for stopped positive muons in water.
  // SK, PRD 110 (2024) 082008, Table VI: r_H2O^+ = 0.718 +/- 0.007.
  // Apply once at the positive-KE -> at-rest transition. G4's spin decay
  // channel expects a UNIT spin; randomizing a fraction is essential.
  if (particle->GetParticleName() == "mu+" &&
      track->GetTrackStatus() == fStopButAlive &&
      step->GetPreStepPoint()->GetKineticEnergy() > 0.0 &&
      step->GetPostStepPoint()->GetKineticEnergy() == 0.0 &&
      step->GetPreStepPoint()->GetMaterial()->GetName() == "G4_WATER") {
    if (G4UniformRand() > 0.718) track->SetPolarization(G4ThreeVector());
  }

'''
assert needle in source
(base/'SteppingAction_spin_water.cc').write_text(source.replace(needle, addition+needle))
macro = (base/'pion_rest_spin.mac').read_text().replace('pion_rest_spin.root','pion_rest_water.root')
(base/'pion_rest_water.mac').write_text(macro)
