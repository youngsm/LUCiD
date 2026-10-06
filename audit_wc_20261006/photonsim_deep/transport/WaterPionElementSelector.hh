#include "G4ElementSelector.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "Randomize.hh"

// Stopped pi- capture in liquid water: measured H probability 0.00445.
// Preserve isotope sampling and all final-state models after choosing target.
class WaterPionElementSelector : public G4ElementSelector {
public:
  const G4Element* SelectZandA(const G4Track& track, G4Nucleus* target) override {
    if(track.GetDefinition()->GetPDGEncoding()!=-211 || track.GetMaterial()->GetName()!="G4_WATER")
      return G4ElementSelector::SelectZandA(track,target);
    const int z=G4UniformRand()<0.00445 ? 1 : 8;
    const G4Element* element;
    do { element=G4ElementSelector::SelectZandA(track,target); }
    while(element->GetZasInt()!=z);
    return element;
  }
};
