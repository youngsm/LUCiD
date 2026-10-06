// In PhotonSim::PhysicsList::ConstructProcess(), after its base call:
// G4VModularPhysicsList::ConstructProcess();
// Include the two Water*ElementSelector.hh audit examples and headers below.
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4MuonMinusCapture.hh"
#include "G4HadronicAbsorptionBertini.hh"
#include "WaterMuonElementSelector.hh"
#include "WaterPionElementSelector.hh"

void InstallWaterCaptureSelectors() {
  auto* particles=G4ParticleTable::GetParticleTable();
  auto* mu=particles->FindParticle("mu-")->GetProcessManager()->GetProcessList();
  for(int i=0;i<mu->size();++i)
    if(auto* process=dynamic_cast<G4MuonMinusCapture*>((*mu)[i]))
      process->SetElementSelector(new WaterMuonElementSelector());
  auto* pi=particles->FindParticle("pi-")->GetProcessManager()->GetProcessList();
  for(int i=0;i<pi->size();++i)
    if(auto* process=dynamic_cast<G4HadronicAbsorptionBertini*>((*pi)[i]))
      process->SetElementSelector(new WaterPionElementSelector());
}
