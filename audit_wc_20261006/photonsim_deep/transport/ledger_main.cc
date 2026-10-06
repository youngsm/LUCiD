#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "StackingAction.hh"
#include "DataManager.hh"
#include "DataManagerMessenger.hh"
#include "RandomSeedMessenger.hh"
#include "G4RunManagerFactory.hh"
#include "G4VUserActionInitialization.hh"
#include "G4UImanager.hh"
#include "G4EmExtraPhysics.hh"
#include "G4GammaGeneralProcess.hh"
#include "G4HadronicProcess.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"
#include "Randomize.hh"
#include "TROOT.h"
#include <fstream>
#include <iomanip>
#include <cstdlib>

struct Ledger {
  long photons=0, gamma_nuc=0, electron_nuc=0, muon_nuc=0;
  long neutrons=0, protons=0, charged_pions=0, secondary_muons=0;
  long primary_decay_e=0, primary_nuclear=0;
  double edep=0, primary_death_ns=-1, primary_range_mm=-1, primary_stop_ns=-1;
  double primary_decay_e_ke=-1;
};
static Ledger ledger;
static std::ofstream output;

class AuditEvent : public PhotonSim::EventAction {
public:
  explicit AuditEvent(PhotonSim::RunAction* r): PhotonSim::EventAction(r) {}
  void BeginOfEventAction(const G4Event* e) override {
    ledger = Ledger(); PhotonSim::EventAction::BeginOfEventAction(e);
  }
  void EndOfEventAction(const G4Event* e) override {
    PhotonSim::EventAction::EndOfEventAction(e);
    output << e->GetEventID() << ',' << ledger.photons << ',' << ledger.edep/MeV
      << ',' << ledger.gamma_nuc << ',' << ledger.electron_nuc << ',' << ledger.muon_nuc
      << ',' << ledger.neutrons << ',' << ledger.protons << ',' << ledger.charged_pions
      << ',' << ledger.secondary_muons << ',' << ledger.primary_decay_e
      << ',' << ledger.primary_nuclear << ',' << ledger.primary_death_ns
      << ',' << ledger.primary_range_mm << ',' << ledger.primary_stop_ns
      << ',' << ledger.primary_decay_e_ke << '\n';
  }
};

class AuditStep : public PhotonSim::SteppingAction {
public:
  explicit AuditStep(PhotonSim::EventAction* e): PhotonSim::SteppingAction(e) {}
  void UserSteppingAction(const G4Step* step) override {
    G4Track* track = step->GetTrack();
    auto particle = track->GetDefinition();
    const int pdg = particle->GetPDGEncoding();
    const auto* process = step->GetPostStepPoint()->GetProcessDefinedStep();
    const G4String name = process ? process->GetProcessName() : "none";
    ledger.edep += step->GetTotalEnergyDeposit();
    if (track->GetGlobalTime() < 10000*ns) {
      if (name == "photonNuclear") ++ledger.gamma_nuc;
      if (name == "electronNuclear" || name == "positronNuclear") ++ledger.electron_nuc;
      if (name == "muonNuclear") ++ledger.muon_nuc;
      if (track->GetParentID()==0 && (name == "photonNuclear" || name == "electronNuclear" || name == "positronNuclear" || name == "muonNuclear")) ++ledger.primary_nuclear;
    }
    if (track->GetCurrentStepNumber()==1 && track->GetParentID()!=0) {
      if (pdg==2112) ++ledger.neutrons;
      if (pdg==2212) ++ledger.protons;
      if (std::abs(pdg)==211) ++ledger.charged_pions;
      if (std::abs(pdg)==13) ++ledger.secondary_muons;
      const auto* creator = track->GetCreatorProcess();
      const auto cname = creator ? creator->GetProcessName() : "none";
      if (std::abs(pdg)==11 && track->GetParentID()==1 && track->GetVertexKineticEnergy()>1*MeV &&
          (cname=="Decay" || cname=="muMinusCaptureAtRest")) {
        ++ledger.primary_decay_e; ledger.primary_decay_e_ke=track->GetVertexKineticEnergy()/MeV;
      }
    }
    const auto* secondaries=step->GetSecondaryInCurrentStep();
    if(secondaries) for(const auto* sec:*secondaries) {
      if(sec->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition()) ++ledger.photons;
    }
    if (track->GetParentID()==0) {
      if (track->GetTrackStatus()==fStopButAlive && ledger.primary_stop_ns<0) ledger.primary_stop_ns=track->GetGlobalTime()/ns;
      if (track->GetTrackStatus()==fStopAndKill) {
        ledger.primary_death_ns=track->GetGlobalTime()/ns;
        ledger.primary_range_mm=track->GetTrackLength()/mm;
      }
    }
    PhotonSim::SteppingAction::UserSteppingAction(step);
  }
};

class AuditStack : public PhotonSim::StackingAction {
public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    // Cherenkov creation is unchanged and counted in the parent's step.
    // Remove only subsequent optical transport/storage, which cannot feed
    // back into charged/hadronic transport in this water configuration.
    if(t->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition()) return fKill;
    return PhotonSim::StackingAction::ClassifyNewTrack(t);
  }
};
class AuditActions : public G4VUserActionInitialization {
public:
  void Build() const override {
    SetUserAction(new PhotonSim::PrimaryGeneratorAction());
    auto* r=new PhotonSim::RunAction(); SetUserAction(r);
    auto* e=new AuditEvent(r); SetUserAction(e);
    SetUserAction(new AuditStep(e)); SetUserAction(new AuditStack());
  }
};

static void dump_processes() {
  std::cout << "AUDIT_VERSION " << G4Version << '\n';
  for(const char* species:{"gamma","e-","e+","mu-","mu+","proton","neutron","pi-","pi+","kaon-","kaon+","kaon0L","alpha"}) {
    auto* particle=G4ParticleTable::GetParticleTable()->FindParticle(species);
    auto* manager=particle->GetProcessManager();
    auto* processes=manager->GetProcessList();
    std::cout << "AUDIT_PROCESSES " << species;
    for(int i=0;i<manager->GetProcessListLength();++i) {
      auto* p=(*processes)[i]; std::cout << ' ' << p->GetProcessName();
      if(auto* general=dynamic_cast<G4GammaGeneralProcess*>(p)) {
        auto* nuclear=general->GetGammaNuclear();
        std::cout << "{nuclear=" << (nuclear?nuclear->GetProcessName():"NULL") << '}';
      }
    }
    std::cout << '\n';
  }
}
int main(int argc,char** argv) {
  if(argc!=4) return 2;
  gROOT->Reset();
  auto* random=new PhotonSim::RandomSeedMessenger();
  long seeds[]={314159,271828}; CLHEP::HepRandom::setTheSeeds(seeds);
  auto* run=G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  run->SetUserInitialization(new PhotonSim::DetectorConstruction());
  auto* physics=new PhotonSim::PhysicsList();
  if(G4String(argv[3])=="extra") physics->RegisterPhysics(new G4EmExtraPhysics(0));
  run->SetUserInitialization(physics); run->SetUserInitialization(new AuditActions());
  auto* data=new PhotonSim::DataManagerMessenger();
  auto* ui=G4UImanager::GetUIpointer();
  ui->ApplyCommand("/run/initialize"); dump_processes();
  output.open(argv[2]); output<<std::setprecision(16);
  output<<"event,photons,edep_MeV,gamma_nuclear,electron_nuclear,muon_nuclear,neutrons,protons,charged_pions,secondary_muons,primary_decay_e,primary_nuclear,primary_death_ns,primary_range_mm,primary_stop_ns,primary_decay_e_ke_MeV\n";
  int status=ui->ApplyCommand(G4String("/control/execute ")+argv[1]);
  PhotonSim::DataManager::GetInstance()->Finalize(); output.close();
  delete random; delete data; delete run; PhotonSim::DataManager::DeleteInstance();
  std::quick_exit(status==0?0:1);
}
