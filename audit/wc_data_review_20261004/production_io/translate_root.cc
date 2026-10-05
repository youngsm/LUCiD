#include <TFile.h>
#include <TTree.h>
#include <array>
#include <vector>
#include <string>

int main(int argc, char **argv) {
  TFile input(argv[1], "READ");
  TFile output(argv[2], "RECREATE");
  auto event = static_cast<TTree*>(input.Get("OpticalPhotons"));
  const std::array<std::string,9> names = {
    "TrackInfo_PosX", "TrackInfo_PosY", "TrackInfo_PosZ",
    "Segment_StartX", "Segment_StartY", "Segment_StartZ",
    "Segment_EndX", "Segment_EndY", "Segment_EndZ"};
  const std::array<double,3> delta = {5000., -2000., 3000.};
  std::array<std::vector<double>*,9> coords{};
  for (size_t i=0; i<names.size(); ++i)
    event->SetBranchAddress(names[i].c_str(), &coords[i]);
  output.cd();
  auto copy = event->CloneTree(0);
  for (Long64_t entry=0; entry<event->GetEntries(); ++entry) {
    event->GetEntry(entry);
    for (size_t i=0; i<names.size(); ++i)
      for (auto &x : *coords[i]) x += delta[i%3];
    copy->Fill();
  }
  copy->Write();
  auto raw = static_cast<TTree*>(input.Get("OpticalPhotonsRaw"));
  std::array<std::vector<float>*,3> photons{};
  for (int i=0; i<3; ++i)
    raw->SetBranchAddress(("PhotonPos"+std::string(1,"XYZ"[i])).c_str(), &photons[i]);
  output.cd();
  auto rcopy = raw->CloneTree(0);
  for (Long64_t entry=0; entry<raw->GetEntries(); ++entry) {
    raw->GetEntry(entry);
    for (int i=0; i<3; ++i)
      for (auto &x : *photons[i]) x += delta[i];
    rcopy->Fill();
  }
  rcopy->Write();
  output.Close();
}
