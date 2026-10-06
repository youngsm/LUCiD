// Exact production primary generator, without transporting the resulting events.
#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "Randomize.hh"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  auto* run = new G4RunManager;
  run->SetUserInitialization(new PhotonSim::DetectorConstruction);
  run->SetUserInitialization(new PhotonSim::PhysicsList);
  auto* generator = new PhotonSim::PrimaryGeneratorAction;
  run->SetUserAction(generator);
  run->Initialize();
  std::cout << "AUDIT_ENGINE " << G4Random::getTheEngine()->name() << '\n';
  auto* table=G4ParticleTable::GetParticleTable();
  std::cout << "AUDIT_O16_PRESENT_BEFORE " << (table->FindParticle(1000080160)!=nullptr) << '\n';
  if (argc > 4 && std::string(argv[4])=="preload-ion") {
    auto* ion=table->GetIonTable()->GetIon(1000080160);
    std::cout << "AUDIT_ION_CREATED " << (ion!=nullptr) << '\n';
  }
  std::ifstream commands(argv[1]);
  std::string line;
  while (std::getline(commands,line)) {
    if (line.empty() || line[0]=='#') continue;
    const int status=G4UImanager::GetUIpointer()->ApplyCommand(line);
    if (status) {
      std::cerr << "AUDIT_COMMAND_FAILURE " << status << ' ' << line << '\n';
      return 3;
    }
  }
  std::ofstream out(argv[3]);
  out << std::setprecision(17);
  out << "event\tvertex\tprimary\tpdg\tkinetic_MeV\tpx_MeV\tpy_MeV\tpz_MeV\tx_mm\ty_mm\tz_mm\tt_ns\tevent_true_MeV\trootracker_entry\tspin_x\tspin_y\tspin_z\n";
  for (int e=0;e<std::stoi(argv[2]);++e) {
    G4Event event(e);
    generator->GeneratePrimaries(&event);
    int np=0;
    for (int vi=0;vi<event.GetNumberOfPrimaryVertex();++vi) {
      auto* vertex=event.GetPrimaryVertex(vi);
      for (int pi=0;pi<vertex->GetNumberOfParticle();++pi) {
        auto* p=vertex->GetPrimary(pi);
        out << e << '\t' << vi << '\t' << np++ << '\t' << p->GetPDGcode()
            << '\t' << p->GetKineticEnergy() << '\t' << p->GetPx() << '\t' << p->GetPy() << '\t' << p->GetPz()
            << '\t' << vertex->GetX0() << '\t' << vertex->GetY0() << '\t' << vertex->GetZ0() << '\t' << vertex->GetT0()
            << '\t' << generator->GetTrueEnergy() << '\t' << generator->GetCurrentGenieEntryID() << '\t' << p->GetPolX() << '\t' << p->GetPolY() << '\t' << p->GetPolZ() << '\n';
      }
    }
    std::cout << "AUDIT_EVENT " << e << " n=" << np << '\n';
  }
  out.close();
  // Match the existing PhotonSim teardown workaround; output is already closed.
  std::quick_exit(0);
}
