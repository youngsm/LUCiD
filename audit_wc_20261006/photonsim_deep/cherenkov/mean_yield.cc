#include "DetectorConstruction.hh"
#include "G4Cerenkov.hh"
#include "G4Electron.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalParameters.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

double sellmeier(double energyEV) {
  const double A[] = {.5684027565,.1726177391,.02086189578,.1130748688};
  const double C[] = {.005101829712,.01821153936,.02620722293,10.69792721};
  const double l = 1.2398419843320026 / energyEV;
  double nsq=1;
  for(int i=0;i<4;++i) nsq += A[i]*l*l/(l*l-C[i]);
  return std::sqrt(nsq);
}

int main() {
  PhotonSim::DetectorConstruction detector;
  auto* water=G4Material::GetMaterial("G4_WATER");
  auto* ri=water->GetMaterialPropertiesTable()->GetProperty("RINDEX");
  std::vector<double> energies, linearRI, physicalRI;
  for(int i=0;i<=2670;++i) {
    const double e=(1.84+.001*i)*eV;
    energies.push_back(e); linearRI.push_back(ri->Value(e));
    physicalRI.push_back(sellmeier(e/eV));
  }
  auto makeMaterial=[&](const char* name,std::vector<double>& r) {
    auto* mat=new G4Material(name,water->GetDensity(),1);
    mat->AddMaterial(water,1.0);
    auto* mpt=new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX",energies,r);
    mat->SetMaterialPropertiesTable(mpt); return mat;
  };
  auto* linear=makeMaterial("AuditDenseLinearWater",linearRI);
  auto* physical=makeMaterial("AuditDenseSellmeierWater",physicalRI);
  G4Cerenkov process;
  process.BuildPhysicsTable(*G4Electron::Definition());
  std::ofstream out("mean_yield.csv");
  out<<std::setprecision(15)<<"beta,coarse_per_mm,dense_linear_per_mm,dense_physical_per_mm\n";
  for(int i=0;i<=5400;++i) {
    const double beta=.73+.00005*i;
    out<<beta<<","<<process.GetAverageNumberOfPhotons(eplus,beta,water,ri)<<",";
    out<<process.GetAverageNumberOfPhotons(eplus,beta,linear,linear->GetMaterialPropertiesTable()->GetProperty("RINDEX"))<<",";
    out<<process.GetAverageNumberOfPhotons(eplus,beta,physical,physical->GetMaterialPropertiesTable()->GetProperty("RINDEX"))<<"\n";
  }
  std::ofstream material("material.txt");
  material<<std::setprecision(15)<<*water<<"\n";
  material<<"Density_gcm3="<<water->GetDensity()/(g/cm3)<<"\nTemperature_K="<<water->GetTemperature()/kelvin;
  material<<"\nCerenkovMaxPhotons="<<process.GetMaxNumPhotonsPerStep();
  material<<"\nCerenkovMaxBetaChangeFraction="<<process.GetMaxBetaChangePerStep()<<"\n";
  for(size_t i=0;i<ri->GetVectorLength();++i)
    material<<"Knot,"<<ri->Energy(i)/eV<<","<<(*ri)[i]<<","<<sellmeier(ri->Energy(i)/eV)<<"\n";
}
