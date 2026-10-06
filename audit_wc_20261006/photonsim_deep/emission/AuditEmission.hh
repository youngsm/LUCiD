#pragma once
// Audit-only observer: records creation snapshots without changing any G4 state
// or consuming random numbers. Included in instrumented local source copies.
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4OpticalPhoton.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <string>

namespace EmissionAudit {
struct Birth { std::array<double,8> v; std::array<double,3> pol; int parent; std::string process; };
inline std::map<int,Birth> photons;
inline int event = -1;
inline unsigned long created = 0, recorded = 0, duplicate = 0;
inline std::array<double,8> max_delta{};
inline double max_polarization_change = 0.;
inline std::map<std::string,unsigned long> first_process;
inline std::map<std::string,unsigned long> creator_process;
inline FILE* bin = nullptr;
inline FILE* summary = nullptr;

inline void Begin(int event_id) {
  if (!bin) {
    const char* path = std::getenv("AUDIT_EMISSION_BINARY");
    const char* summary_path = std::getenv("AUDIT_EMISSION_SUMMARY");
    if (!path || !summary_path) throw std::runtime_error("Missing audit output paths");
    bin = std::fopen(path,"wb"); summary = std::fopen(summary_path,"w");
    if (!bin || !summary) throw std::runtime_error("Cannot open audit files");
  }
  event=event_id; created=recorded=duplicate=0; photons.clear();
  max_delta.fill(0.); max_polarization_change=0.; first_process.clear(); creator_process.clear();
}

inline void Created(const G4Track* track) {
  if (track->GetDefinition()!=G4OpticalPhoton::OpticalPhotonDefinition()) return;
  const auto p=track->GetPosition(), d=track->GetMomentumDirection(), pol=track->GetPolarization();
  const auto cp=track->GetCreatorProcess();
  Birth b{{p.x()/mm,p.y()/mm,p.z()/mm,d.x(),d.y(),d.z(),track->GetGlobalTime()/ns,
            h_Planck*c_light/track->GetKineticEnergy()/nm},
          {pol.x(),pol.y(),pol.z()},track->GetParentID(), cp ? std::string(cp->GetProcessName()) : "Primary"};
  if (photons.count(track->GetTrackID())) ++duplicate;
  photons[track->GetTrackID()]=b; ++created; ++creator_process[b.process];
}

inline void Recorded(const G4Step* step) {
  const auto track=step->GetTrack();
  if (track->GetDefinition()!=G4OpticalPhoton::OpticalPhotonDefinition() || track->GetCurrentStepNumber()!=1) return;
  const auto it=photons.find(track->GetTrackID());
  if (it==photons.end()) throw std::runtime_error("Optical first step has no creation snapshot");
  const Birth b=it->second;
  if (track->GetParentID()!=b.parent) throw std::runtime_error("Optical parent changed before record");
  const auto p=track->GetVertexPosition(), d=track->GetVertexMomentumDirection();
  std::array<double,8> actual{{p.x()/mm,p.y()/mm,p.z()/mm,d.x(),d.y(),d.z(),
    (track->GetGlobalTime()-step->GetDeltaTime())/ns,h_Planck*c_light/track->GetKineticEnergy()/nm}};
  for (int i=0;i<8;++i) max_delta[i]=std::max(max_delta[i],std::abs(actual[i]-b.v[i]));
  const auto pol=track->GetPolarization();
  max_polarization_change=std::max(max_polarization_change,
    std::sqrt(std::pow(pol.x()-b.pol[0],2)+std::pow(pol.y()-b.pol[1],2)+std::pow(pol.z()-b.pol[2],2)));
  const auto process=step->GetPostStepPoint()->GetProcessDefinedStep();
  ++first_process[process ? std::string(process->GetProcessName()) : "none"];
  // Record consists of 4 int32 fields and 16 float64 fields (144 bytes).
  int header[4]={event,track->GetTrackID(),track->GetParentID(),b.process=="Cerenkov" ? 1 : 0};
  std::fwrite(header,sizeof(int),4,bin);
  std::fwrite(b.v.data(),sizeof(double),8,bin);
  std::fwrite(actual.data(),sizeof(double),8,bin);
  photons.erase(it); ++recorded;
}

inline void End() {
  std::fprintf(summary,"{\"event\":%d,\"created\":%lu,\"recorded\":%lu,\"unrecorded\":%zu,\"duplicate_classifications\":%lu,\"max_creation_vs_record_delta\":[",
    event,created,recorded,photons.size(),duplicate);
  for(int i=0;i<8;++i) std::fprintf(summary,"%s%.17g",i ? ",":"",max_delta[i]);
  std::fprintf(summary,"],\"max_polarization_change\":%.17g,\"first_step_processes\":{",max_polarization_change);
  bool first=true;
  for(const auto& kv:first_process) { std::fprintf(summary,"%s\"%s\":%lu",first?"":",",kv.first.c_str(),kv.second); first=false; }
  std::fprintf(summary,"},\"creator_processes\":{"); first=true;
  for(const auto& kv:creator_process) { std::fprintf(summary,"%s\"%s\":%lu",first?"":",",kv.first.c_str(),kv.second); first=false; }
  std::fprintf(summary,"}}\n"); std::fflush(summary); std::fflush(bin);
}
}
