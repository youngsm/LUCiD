#pragma once
// Audit prototype for preserving the exact parent G4 emission step.
#include "G4VUserTrackInformation.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4OpticalPhoton.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <array>

class EmissionStepInfo : public G4VUserTrackInformation {
public:
  int parent, local_step;
  std::array<double,12> birth;
  EmissionStepInfo(const G4Step* step,const G4Track* photon):
    parent(step->GetTrack()->GetTrackID()),local_step(step->GetTrack()->GetCurrentStepNumber()-1) {
    auto p=photon->GetPosition(), pre=step->GetPreStepPoint()->GetPosition(), post=step->GetPostStepPoint()->GetPosition();
    birth={{photon->GetGlobalTime()/ns,step->GetPreStepPoint()->GetGlobalTime()/ns,
            step->GetPostStepPoint()->GetGlobalTime()/ns,
            p.x()/mm,p.y()/mm,p.z()/mm,pre.x()/mm,pre.y()/mm,pre.z()/mm,
            post.x()/mm,post.y()/mm,post.z()/mm}};
  }
  void Print() const override {}
};

inline void TagEmissionStep(const G4Step* step) {
  if(step->GetTrack()->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition()) return;
  const auto secondary=step->GetSecondaryInCurrentStep();
  if(!secondary) return;
  for(const G4Track* photon:*secondary) {
    if(photon->GetDefinition()!=G4OpticalPhoton::OpticalPhotonDefinition()) continue;
    if(photon->GetUserInformation()) throw std::runtime_error("Existing photon user information");
    const_cast<G4Track*>(photon)->SetUserInformation(new EmissionStepInfo(step,photon));
  }
}

inline int ReadAndRecordEmissionStep(const G4Track* photon) {
  const auto info=dynamic_cast<const EmissionStepInfo*>(photon->GetUserInformation());
  if(!info) throw std::runtime_error("Optical photon missing emission step tag");
  static FILE* f=nullptr;
  if(!f) {
    f=std::fopen(std::getenv("AUDIT_STEP_BINARY"),"wb");
    if(!f) throw std::runtime_error("Cannot create emission step ledger");
  }
  int header[4]={G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID(),
    photon->GetTrackID(),photon->GetParentID(),info->local_step};
  if(header[2]!=info->parent) throw std::runtime_error("Emission step parent mismatch");
  std::fwrite(header,sizeof(int),4,f);
  std::fwrite(info->birth.data(),sizeof(double),12,f);
  // Baseline main uses quick_exit, so flush here for complete audit evidence.
  std::fflush(f);
  return info->local_step;
}
