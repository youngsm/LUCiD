#include <TFile.h>
#include <TTree.h>
#include <vector>
#include <string>
#include <cstdlib>

// A real TTree with the same vector branch types written by PhotonSim.
int main(int argc, char** argv) {
  TFile output(argv[1], "RECREATE");
  TTree event("OpticalPhotons", "OpticalPhotons");
  int eid=argc>2?std::atoi(argv[2]):0, nphotons=9, nsegments=11, entry=-1, nu=0;
  double energy=2000, nuke=0;
  std::vector<int> tid={1,2,3,4,5,6,7,8,9,10,11};
  std::vector<int> parent={0,0,1,2,4,0,0,7,8,7,10};
  std::vector<int> pdg={-13,11,-11,22,11,2112,111,22,11,22,11};
  std::vector<double> ke={1000,500,30,40,20,1,500,250,20,200,20};
  std::vector<double> times={0,0,2300,2,2.1,0,0,.1,.2,.1,.2};
  std::vector<std::string> process={"Primary","Primary","Decay","eBrem","conv","Primary","Primary","Decay","conv","Decay","conv"};
  std::vector<double> sx(11), sy(11), sz(11), ex(11), ey(11), ez(11);
  std::vector<double> dx(11,0), dy(11,0), dz(11,1), edep(11,.1), beta(11,.99);
  std::vector<int> nch={2,2,1,0,1,0,0,0,2,0,1};
  std::vector<int> psi={0,0,1,1,2,4,8,8,10};
  for(int i=0;i<11;++i) {
    sx[i]=(i+1)*1000; sy[i]=-(i+1)*250; sz[i]=(i+1)*500;
    ex[i]=sx[i]; ey[i]=sy[i]; ez[i]=sz[i]+10;
  }
  event.Branch("EventID", &eid);
  event.Branch("NOpticalPhotons", &nphotons);
  event.Branch("NSegments", &nsegments);
  event.Branch("PrimaryEnergy", &energy);
  event.Branch("RooTrackerEntryID", &entry);
  event.Branch("IncomingNuPdg", &nu);
  event.Branch("IncomingNuKE", &nuke);
  event.Branch("TrackInfo_TrackID", &tid);
  event.Branch("TrackInfo_ParentTrackID", &parent);
  event.Branch("TrackInfo_PDG", &pdg);
  event.Branch("TrackInfo_Energy", &ke);
  event.Branch("TrackInfo_Time", &times);
  event.Branch("TrackInfo_CreatorProcess", &process);
  event.Branch("TrackInfo_PosX", &sx); event.Branch("TrackInfo_PosY", &sy); event.Branch("TrackInfo_PosZ", &sz);
  event.Branch("TrackInfo_DirX", &dx); event.Branch("TrackInfo_DirY", &dy); event.Branch("TrackInfo_DirZ", &dz);
  event.Branch("Segment_TrackID", &tid);
  event.Branch("Segment_StartX", &sx); event.Branch("Segment_StartY", &sy); event.Branch("Segment_StartZ", &sz);
  event.Branch("Segment_EndX", &ex); event.Branch("Segment_EndY", &ey); event.Branch("Segment_EndZ", &ez);
  event.Branch("Segment_DirX", &dx); event.Branch("Segment_DirY", &dy); event.Branch("Segment_DirZ", &dz);
  event.Branch("Segment_Edep", &edep); event.Branch("Segment_Time", &times);
  event.Branch("Segment_BetaStart", &beta); event.Branch("Segment_NCherenkov", &nch);
  event.Branch("Photon_SegmentIndex", &psi);
  event.Fill();

  TTree raw("OpticalPhotonsRaw", "OpticalPhotonsRaw");
  long long start=0;
  std::vector<float> px, py, pz, pdx, pdy, pdz, pt, wl;
  raw.Branch("EventID", &eid); raw.Branch("ChunkStartID", &start);
  raw.Branch("PhotonPosX", &px); raw.Branch("PhotonPosY", &py); raw.Branch("PhotonPosZ", &pz);
  raw.Branch("PhotonDirX", &pdx); raw.Branch("PhotonDirY", &pdy); raw.Branch("PhotonDirZ", &pdz);
  raw.Branch("PhotonTime", &pt); raw.Branch("PhotonWavelength", &wl);
  // Deliberately reorder chunks to exercise ChunkStartID ordering.
  for(const auto& bounds : std::vector<std::pair<int,int>>{{5,9},{0,3},{3,5}}) {
    start=bounds.first;
    px.clear(); py.clear(); pz.clear(); pdx.clear(); pdy.clear(); pdz.clear(); pt.clear(); wl.clear();
    for(int i=bounds.first;i<bounds.second;++i) {
      int seg=psi[i];
      px.push_back(sx[seg]); py.push_back(sy[seg]); pz.push_back(sz[seg]+5);
      pdx.push_back(.6); pdy.push_back(0); pdz.push_back(.8);
      pt.push_back(times[seg]+.01); wl.push_back(350+i*10);
    }
    raw.Fill();
  }
  // Legitimate zero-photon, zero-track event.
  eid+=1; nphotons=0; nsegments=0; energy=0;
  tid.clear(); parent.clear(); pdg.clear(); ke.clear(); times.clear(); process.clear();
  sx.clear(); sy.clear(); sz.clear(); ex.clear(); ey.clear(); ez.clear();
  dx.clear(); dy.clear(); dz.clear(); edep.clear(); beta.clear(); nch.clear(); psi.clear();
  event.Fill();
  output.cd(); event.Write(); raw.Write(); output.Close();
}
