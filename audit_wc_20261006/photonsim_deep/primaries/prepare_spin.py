"""Prepare a GENIE-export-shaped spin fixture and isolated minimal repair."""
from pathlib import Path
import os, json, shutil
import numpy as np
import uproot
from lucid.production.generate_macro import generate_macro
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano','roma')
B=Path(__file__).resolve().parent
U=B.parents[1]/'photonsim_physics/upstream'
P=B/'spin_patch'; (P/'include').mkdir(parents=True,exist_ok=True); (P/'src').mkdir(exist_ok=True)
def replace_once(s,old,new):
    assert s.count(old)==1,(old,s.count(old)); return s.replace(old,new)
h=(U/'include/RooTrackerReader.hh').read_text()
h=replace_once(h,'  G4double E;','  G4double E;\n  G4double polx = 0., poly = 0., polz = 0.;')
h=replace_once(h,'    Double_t fStdHepP4[kMaxParticles][4] = {};','    Double_t fStdHepP4[kMaxParticles][4] = {};\n    Double_t fStdHepPolz[kMaxParticles][3] = {};\n    G4bool fHasPolarization = false;')
(P/'include/RooTrackerReader.hh').write_text(h)
r=(U/'src/RooTrackerReader.cc').read_text()
r=replace_once(r,'  fPath = path;','  fHasPolarization = false;\n  fPath = path;')
r=replace_once(r,'  fTree->SetBranchAddress("StdHepP4",      fStdHepP4);','  fTree->SetBranchAddress("StdHepP4",      fStdHepP4);\n  if (fTree->GetBranch("StdHepPolz"))\n    fHasPolarization = fTree->SetBranchAddress("StdHepPolz", fStdHepPolz) >= 0;')
r=replace_once(r,'    fFinalState.push_back({pdg, px, py, pz, E});','    fFinalState.push_back({pdg, px, py, pz, E,\n      fHasPolarization ? fStdHepPolz[k][0] : 0.,\n      fHasPolarization ? fStdHepPolz[k][1] : 0.,\n      fHasPolarization ? fStdHepPolz[k][2] : 0.});')
(P/'src/RooTrackerReader.cc').write_text(r)
g=(U/'src/PrimaryGeneratorAction.cc').read_text()
g=replace_once(g,'      G4ThreeVector mom(p.px * MeV, p.py * MeV, p.pz * MeV);','      G4ThreeVector mom(p.px * MeV, p.py * MeV, p.pz * MeV);\n      G4ThreeVector spin(p.polx, p.poly, p.polz);')
g=replace_once(g,'        mom = rot * mom;','        mom = rot * mom;\n        spin = rot * spin;')
g=replace_once(g,'      auto* primary = new G4PrimaryParticle(pdef, mom.x(), mom.y(), mom.z());','      auto* primary = new G4PrimaryParticle(pdef, mom.x(), mom.y(), mom.z());\n      primary->SetPolarization(spin);')
(P/'src/PrimaryGeneratorAction.cc').write_text(g)
probe=(B/'primary_probe.cc').read_text()
probe=replace_once(probe,'rootracker_entry\\n','rootracker_entry\\tspin_x\\tspin_y\\tspin_z\\n')
probe=replace_once(probe,"generator->GetCurrentGenieEntryID() << '\\n';","generator->GetCurrentGenieEntryID() << '\\t' << p->GetPolX() << '\\t' << p->GetPolY() << '\\t' << p->GetPolZ() << '\\n';")
(B/'spin_probe.cc').write_text(probe)
n=256
pdg=np.tile(np.array([14,13,2212],dtype=np.int32),(n,1)); pdg[1::2,:2]*=-1
status=np.tile(np.array([0,1,1],dtype=np.int32),(n,1))
mom=np.array([[0.,0.,1000.],[100.,200.,300.],[-100.,-200.,700.]])
mass=np.array([0.,105.6583715,938.27208816])
p4=np.column_stack((mom,np.sqrt(np.sum(mom*mom,axis=1)+mass*mass)))/1000.
pol=np.zeros((n,3,3)); unit=mom[1]/np.linalg.norm(mom[1]); pol[:,1,:]=-unit; pol[1::2,1,:]=unit
with uproot.recreate(B/'polarized.gtrac.root') as f:
    t=f.mktree('gRooTracker',{'StdHepN':'int32','StdHepPdg':'3 * int32','StdHepStatus':'3 * int32',
        'StdHepP4':'3 * 4 * float64','StdHepPolz':'3 * 3 * float64'})
    t.extend({'StdHepN':np.full(n,3,np.int32),'StdHepPdg':pdg,'StdHepStatus':status,
        'StdHepP4':np.tile(p4,(n,1,1)),'StdHepPolz':pol})
for name,direction in [('spin_random','isotropic'),('spin_fixed','beam')]:
    cfg={'name':name,'primary_source':'genie','genie':{'direction':direction}}
    mac=generate_macro(cfg,'spin_unused.root',n,genie_rootracker=str(B/'polarized.gtrac.root'),photonsim_seeds=(1171,2237))
    (B/f'{name}.commands').write_text('\n'.join(l for l in mac.splitlines() if l.startswith(('/random/','/gun/')))+'\n')
(B/'spin_fixture_description.json').write_text(json.dumps({'events':n,'polarization_rule':'mu-=-p_hat, mu+=+p_hat; nucleon unset=0','synthetic':True},indent=2))
