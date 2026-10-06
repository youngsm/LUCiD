#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "DataManager.hh"
#include "DataManagerMessenger.hh"
#include "RandomSeedMessenger.hh"
#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UImanager.hh"
#include "Randomize.hh"
#include "TROOT.h"
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  gROOT->Reset();
  G4SteppingVerbose::UseBestUnit(4);
  auto randomMessenger = new PhotonSim::RandomSeedMessenger();
  long seeds[] = {314159, 271828};
  CLHEP::HepRandom::setTheSeeds(seeds);
  auto runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  runManager->SetUserInitialization(new PhotonSim::DetectorConstruction());
  runManager->SetUserInitialization(new PhotonSim::PhysicsList());
  runManager->SetUserInitialization(new PhotonSim::ActionInitialization());
  auto dataMessenger = new PhotonSim::DataManagerMessenger();
  int status = G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  PhotonSim::DataManager::GetInstance()->Finalize();
  delete randomMessenger;
  delete dataMessenger;
  delete runManager;
  PhotonSim::DataManager::DeleteInstance();
  std::quick_exit(status == 0 ? 0 : 1);
}
