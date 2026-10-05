# Source from a milano/roma Slurm job.
source /cvmfs/sft.cern.ch/lcg/releases/gcc/11.3.0/x86_64-el8/setup.sh
wc_lcg_view=/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt
export PATH="$wc_lcg_view/bin:$PATH"
export LD_LIBRARY_PATH="$wc_lcg_view/lib:$wc_lcg_view/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$wc_lcg_view:${CMAKE_PREFIX_PATH:-}"
export ROOTSYS=/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.34.02-18eb6/x86_64-el8-gcc11-opt
source /cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt/bin/geant4.sh
