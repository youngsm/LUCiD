#pragma once
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <string>

namespace PionAudit {
struct Endpoint { G4ThreeVector position; double time; double energy; int status; };
struct State {
  int event=0;
  long photons=0, chainPhotons=0, pionSteps=0, chainTracks=0, replacements=0;
  long resumed=0, resumedStopped=0, terminals=0, decays=0, captures=0, inelastic=0;
  long muonBirths=0, secondaryBirths=0, piPlusBirths=0, piMinusBirths=0;
  double length=0, endTime=0, endRadius=0, maxRadius=0;
  double maxBirthPositionError=0, maxBirthTimeError=0, maxBirthEnergyError=0;
  std::array<long,7> replacementStatus{};
  std::set<int> chain;
  std::map<int,Endpoint> replacementsByParent;
  std::string terminalProcess="none";
};
inline State state;
inline std::ofstream output;
inline void Begin(int event) {
  state=State{}; state.event=event;
  if (!output.is_open()) {
    const char* filename=std::getenv("AUDIT_PION_CSV");
    output.open(filename ? filename : "pion_audit.csv");
    output << "event,photons,chain_photons,pion_steps,chain_tracks,replacements,resumed,resumed_stopped,terminals,decays,captures,inelastic,muon_births,secondary_births,pi_plus_births,pi_minus_births,chain_length_mm,end_time_ns,end_radius_mm,max_radius_mm,max_birth_position_error_mm,max_birth_time_error_ns,max_birth_energy_error_mev,replace_alive,replace_stopbutalive,replace_stopkill,replace_killall,replace_suspend,replace_suspendwait,replace_postpone,terminal_process\n";
  }
}
inline void End() {
  output << std::setprecision(17) << state.event << ',' << state.photons << ',' << state.chainPhotons << ',' << state.pionSteps << ',' << state.chainTracks << ',' << state.replacements << ',' << state.resumed << ',' << state.resumedStopped << ',' << state.terminals << ',' << state.decays << ',' << state.captures << ',' << state.inelastic << ',' << state.muonBirths << ',' << state.secondaryBirths << ',' << state.piPlusBirths << ',' << state.piMinusBirths << ',' << state.length/mm << ',' << state.endTime/ns << ',' << state.endRadius/mm << ',' << state.maxRadius/mm << ',' << state.maxBirthPositionError/mm << ',' << state.maxBirthTimeError/ns << ',' << state.maxBirthEnergyError/MeV;
  for (auto n: state.replacementStatus) output << ',' << n;
  output << ',' << state.terminalProcess << '\n'; output.flush();
}
inline void Replacement(const G4Track* track, int status) {
  if (!state.chain.count(track->GetTrackID())) return;
  ++state.replacements;
  if (status>=0 && status<7) ++state.replacementStatus[status];
  state.replacementsByParent[track->GetTrackID()]={track->GetPosition(),track->GetGlobalTime(),track->GetKineticEnergy(),status};
}
inline void Step(const G4Step* step) {
  const auto* t=step->GetTrack();
  const int pdg=t->GetDefinition()->GetPDGEncoding();
  const auto* creator=t->GetCreatorProcess();
  const std::string creation=creator ? std::string(creator->GetProcessName()) : "Primary";
  if (t->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition()) {
    if(t->GetCurrentStepNumber()==1) {
      ++state.photons;
      if(state.chain.count(t->GetParentID())) ++state.chainPhotons;
    }
    return;
  }
  if(t->GetCurrentStepNumber()==1) {
    // Count genuine physical pion births, excluding the audit target's
    // artificial deflection children even for secondary pion chains.
    if(creation.rfind("Deflection_",0)!=0) {
      if(pdg==211) ++state.piPlusBirths;
      if(pdg==-211) ++state.piMinusBirths;
    }
    const bool inherited=creation.rfind("Deflection_",0)==0 && state.chain.count(t->GetParentID());
    if((std::abs(pdg)==211 && t->GetParentID()==0) || inherited) {
      state.chain.insert(t->GetTrackID()); ++state.chainTracks;
      if(inherited) {
        ++state.resumed;
        if(step->GetPreStepPoint()->GetKineticEnergy()==0) ++state.resumedStopped;
        const auto& old=state.replacementsByParent.at(t->GetParentID());
        state.maxBirthPositionError=std::max(state.maxBirthPositionError,(step->GetPreStepPoint()->GetPosition()-old.position).mag());
        state.maxBirthTimeError=std::max(state.maxBirthTimeError,std::abs(step->GetPreStepPoint()->GetGlobalTime()-old.time));
        state.maxBirthEnergyError=std::max(state.maxBirthEnergyError,std::abs(step->GetPreStepPoint()->GetKineticEnergy()-old.energy));
      }
    } else if(state.chain.count(t->GetParentID())) {
      ++state.secondaryBirths;
      if(std::abs(pdg)==13 && creation=="Decay") ++state.muonBirths;
    }
  }
  if(!state.chain.count(t->GetTrackID())) return;
  ++state.pionSteps; state.length+=step->GetStepLength();
  state.maxRadius=std::max(state.maxRadius,t->GetPosition().mag());
  if(t->GetTrackStatus()==fStopAndKill || t->GetTrackStatus()==fKillTrackAndSecondaries) {
    ++state.terminals;state.endTime=t->GetGlobalTime();state.endRadius=t->GetPosition().mag();
    const auto* process=step->GetPostStepPoint()->GetProcessDefinedStep();
    const std::string name=process ? std::string(process->GetProcessName()) : "None";
    state.terminalProcess=name;
    if(name=="Decay") ++state.decays;
    if(name.find("Capture")!=std::string::npos || name.find("capture")!=std::string::npos) ++state.captures;
    if(name.find("Inelastic")!=std::string::npos || name.find("inelastic")!=std::string::npos) ++state.inelastic;
  }
}
}
