source geant4_env.sh
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
g++ -O2 -std=c++17 -IPhotonSim_HPT/include -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) headless_main.cc PhotonSim_HPT/src/*.cc -o PhotonSim_HPT_audit -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
