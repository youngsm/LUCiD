#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include <iostream>
int main() {
  PhotonSim::DetectorConstruction detector;
  auto* water=G4Material::GetMaterial("G4_WATER");
  std::cout << "Water density_g_cm3=" << water->GetDensity()/(CLHEP::g/CLHEP::cm3)
            << " temperature_K=" << water->GetTemperature()/CLHEP::kelvin
            << " state=" << water->GetState() << '\n';
  auto* elements=water->GetElementVector();
  auto* densities=water->GetVecNbOfAtomsPerVolume();
  for (size_t i=0;i<water->GetNumberOfElements();++i)
    std::cout << "element=" << (*elements)[i]->GetName() << " number_density_m3="
              << densities[i]*CLHEP::m*CLHEP::m*CLHEP::m << '\n';
}
