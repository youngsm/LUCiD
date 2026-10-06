#include "G4ElementSelector.hh"
#include "G4Material.hh"

// Water-specific muonic hydrogen -> oxygen transfer before disappearance.
// Attach only to G4MuonMinusCapture; do not alter pion/kaon selectors.
class WaterMuonElementSelector : public G4ElementSelector {
public:
  const G4Element* SelectZandA(const G4Track& track, G4Nucleus* target) override {
    const G4Element* element = G4ElementSelector::SelectZandA(track, target);
    if (track.GetMaterial()->GetName() == "G4_WATER") {
      while (element->GetZasInt() != 8) {
        element = G4ElementSelector::SelectZandA(track, target);
      }
    }
    return element;
  }
};
