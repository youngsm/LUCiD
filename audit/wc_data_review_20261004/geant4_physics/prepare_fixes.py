from pathlib import Path
base=Path(__file__).resolve().parent
step=base/'PhotonSim_fixed/src/SteppingAction.cc'
s=step.read_text()
cut='''    if (time > 10000.0 * ns) {
      track->SetTrackStatus(fStopAndKill);
      return;  // Don't register or process this track
    }
'''
assert s.count(cut)==1
s=s.replace(cut,'')
s=s.replace('G4ThreeVector kinkPosition = info->preMomentumPos;', 'G4ThreeVector kinkPosition = track->GetPosition();')
s=s.replace('G4double kinkTime = info->preMomentumTime;', 'G4double kinkTime = track->GetGlobalTime();')
step.write_text(s)
primary=base/'PhotonSim_fixed/src/PrimaryGeneratorAction.cc'
s=primary.read_text()
old='      fGenieRotAngle = 2.0 * M_PI * G4UniformRand();'
new='''      do {
        fGenieRotAngle = 2.0 * M_PI * G4UniformRand();
      } while (G4UniformRand() > std::pow(std::sin(fGenieRotAngle / 2.0), 2));'''
assert s.count(old)==1
primary.write_text(s.replace(old,new))
for name in ('gamma_prompt','gamma_delayed','pion_deflection','genie_isotropic','neutron'):
    s=(base/f'{name}.mac').read_text()
    assert f'/output/filename {name}.root' in s
    (base/f'{name}_fixed.mac').write_text(s.replace(f'/output/filename {name}.root',f'/output/filename {name}_fixed.root'))
