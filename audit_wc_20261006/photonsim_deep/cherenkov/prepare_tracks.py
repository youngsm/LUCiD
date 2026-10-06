"""Prepare instrumented, otherwise identical, pinned PhotonSim builds on Slurm."""
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') in {'milano','roma'}
base=Path(__file__).resolve().parent
up=base/'../../photonsim_physics/upstream'
src=(up/'src/DetectorConstruction.cc').read_text()
needle='  waterMPT->AddProperty("RINDEX", photonEnergy, refractiveIndex, nEntries);'
assert src.count(needle)==1
addition='''
  // AUDIT: densify exactly the same piecewise-linear phase index.
  std::vector<G4double> auditEnergy, auditIndex;
  for (int j=0;j<nEntries-1;++j) {
    const int divisions=1000;
    for (int k=0;k<divisions;++k) {
      const double f=double(k)/divisions;
      auditEnergy.push_back(photonEnergy[j]+f*(photonEnergy[j+1]-photonEnergy[j]));
      auditIndex.push_back(refractiveIndex[j]+f*(refractiveIndex[j+1]-refractiveIndex[j]));
    }
  }
  auditEnergy.push_back(photonEnergy[nEntries-1]);
  auditIndex.push_back(refractiveIndex[nEntries-1]);
  waterMPT->RemoveProperty("RINDEX");
  waterMPT->AddProperty("RINDEX",auditEnergy,auditIndex);
'''
(base/'DetectorConstruction_dense.cc').write_text('#include <vector>\n'+src.replace(needle,needle+addition))
src=(up/'src/SteppingAction.cc').read_text()
needle='  G4ParticleDefinition* particle = track->GetDefinition();'
assert src.count(needle)==1
addition='''
  // AUDIT ONLY: observe actual step endpoints/length and photons before storage.
  static std::ofstream auditSteps(std::getenv("AUDIT_STEPS"));
  static std::ofstream auditPhotons(std::getenv("AUDIT_PHOTONS"));
  static long auditPhotonRows=0;
  if (particle->GetPDGCharge()!=0) {
    const auto* a=step->GetPreStepPoint();
    const auto* b=step->GetPostStepPoint();
    const auto* secondaries=step->GetSecondaryInCurrentStep();
    int nCherenkov=0;
    for (const auto* s:*secondaries) {
      const auto* cp=s->GetCreatorProcess();
      if (s->GetDefinition()!=G4OpticalPhoton::Definition() || !cp || cp->GetProcessName()!="Cerenkov") continue;
      ++nCherenkov;
      if (auditPhotonRows++<200000) {
        auto delta=step->GetDeltaPosition();
        double fraction=delta.mag2()>0 ? (s->GetPosition()-a->GetPosition()).dot(delta)/delta.mag2() : 0;
        auditPhotons<<std::setprecision(16)<<a->GetBeta()<<","<<b->GetBeta()<<","<<s->GetKineticEnergy()/eV<<","<<s->GetMomentumDirection().dot(delta.unit())<<","<<fraction<<","<<(s->GetGlobalTime()-a->GetGlobalTime())/ns<<","<<step->GetStepLength()/mm<<"\\n";
      }
    }
    auditSteps<<std::setprecision(16)<<G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID()<<","<<track->GetTrackID()<<","<<particle->GetPDGEncoding()<<","<<a->GetBeta()<<","<<b->GetBeta()<<","<<step->GetStepLength()/mm<<","<<particle->GetPDGCharge()/eplus<<","<<nCherenkov<<"\\n";
    // quick_exit in the existing headless harness requires explicit flushing.
    auditSteps.flush(); auditPhotons.flush();
  }
'''
(base/'SteppingAction_observe.cc').write_text('#include <fstream>\n#include <iomanip>\n#include <cstdlib>\n'+src.replace(needle,needle+addition))
scenarios=[('e025','e-',.25,1000),('e030','e-',.30,1000),('e035','e-',.35,1000),('e050','e-',.5,1000),('e1','e-',1,300),('e10','e-',10,100),('e100','e-',100,10),('mu60','mu-',60,30),('mu100','mu-',100,30),('mu1000','mu-',1000,5),('p500','proton',500,20),('p1000','proton',1000,10)]
for variant in ('coarse','dense','fine'):
    for name,particle,energy,count in scenarios:
        (base/f'{variant}_{name}.mac').write_text(f'''/output/filename {base}/{variant}_{name}.root
{('/process/optical/cerenkov/setMaxBetaChange 0.1' if variant=='fine' else '')}
/run/initialize
/random/setSeeds 718192 192817
/photon/storeIndividual false
/gun/clearPrimaries
/gun/addPrimary {particle} {energy} MeV
/gun/position 0 0 0 m
/gun/direction 0 0 1
/run/beamOn {count}
''')
