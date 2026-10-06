# Independent water-transport oracles

Research/static review, 2026-10-06. No simulation, import, build, or numerical script was executed by this agent. The parent transport lane owns the Slurm milano/roma reproductions. This file separates external physical measurements from predictions made by reading Geant4 11.3.0. It does not claim a Monte Carlo model can be proved universally physically exact.

## Stopped negative muons: target, branching fraction, and time

For pure water, use **1.7954 ± 0.0020 µs** for the stopped negative-muon disappearance lifetime and **0.184 ± 0.001** for the nuclear-capture probability. These numbers are stated on printed pp.63–64 of the [SK Kitagawa thesis (2022)](https://www-sk.icrr.u-tokyo.ac.jp/sk/_pdf/articles/2022/Master_Thesis_Kitagawa.ver8_.pdf), citing [Suzuki, Measday and Roalsvig, PRC 35, 2212 (1987)](https://doi.org/10.1103/PhysRevC.35.2212). The thesis was read; the original 1987 full text was not successfully retrieved.

The lifetime convention is `tau = 1 / (lambda_decay + lambda_capture)`, and the capture probability is `lambda_capture / (lambda_decay + lambda_capture)`. The decay-electron time distribution has this same disappearance lifetime for a single bound species. The reciprocal capture rate alone is **not** the observed lifetime. A mixture of H-bound and O-bound muons is not one exponential, and conditioning that mixture on decay changes its component weights.

The physical target oracle is independently supported by the [SK cosmic-muon charge-ratio/polarization paper, PRD 110, 082008 (2024), printed p.9](https://eprints.gla.ac.uk/340602/1/340602.pdf): a muon initially bound to hydrogen in water transfers to oxygen much faster than it decays. Its water decay-in-orbit treatment therefore uses oxygen. This supports an effective oxygen-bound model at WAND time scales; neglecting the short molecular transfer time is still an approximation.

### What the pinned code actually does

Source links below are pinned to v11.3.0. Line numbers refer to raw source files, including their license headers, as stored beside this note.

1. [G4StoppingPhysics.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/stopping/src/G4StoppingPhysics.cc), lines103–104 and124–125, constructs and registers `G4MuonMinusCapture` on `mu-`.
2. [G4HadronStoppingProcess.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc), lines63–67, creates the default `G4ElementSelector`; lines139–140 select one target nucleus.
3. [G4ElementSelector.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4ElementSelector.cc), lines73–98, uses atom-number-density times Z, except that oxygen receives a factor0.56. For H2O this means **H:O weights2:4.48**, hence **P(H)=2/6.48≈30.86%**. Lines104–120 then sample the chosen element's isotope and write its A,Z into the target.
4. `G4HadronStoppingProcess.cc`, lines155 and166–169, passes that target through `G4EmCaptureCascade`, then `G4MuonMinusBoundDecay`; lines181–191 and the following capture call retain the same target. Reading [G4EmCaptureCascade.cc, `ApplyYourself`](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4EmCaptureCascade.cc) finds atomic de-excitation using the given A,Z, with no material transfer or target reassignment. There is no hidden H→O transfer in this call chain.
5. [G4MuonMinusBoundDecay.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc), lines88–105, samples delay and branch from those two rates. Lines205 and216 give capture rates0.000725/µs for H1 and0.10242/µs for O16. With decay rates near0.455/µs, the source predicts about **12.7% capture**, rather than18.4%, and about **7% too many decay electrons per stopped mu−**. These are code-derived predictions, not executed results.

Switching to [G4MuonMinusAtomicCapture.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4MuonMinusAtomicCapture.cc) alone is not a demonstrated fix: its constructor also uses `G4ElementSelector`, and its atom construction uses that selected nucleus.

### Minimal reproduction and correction

Create stopped mu− in `G4_WATER`, record the target Z and all direct secondaries at their creation, and count creator-model IDs ending in `_DIO` and `_NuclearCapture`. Both branches have the process name `muMinusCaptureAtRest`; counting only creator-process `Decay` is wrong. Record branch delays from the stop time **before** production stacking/time cuts. Compare the unconditional capture fraction and oxygen-bound lifetime with the measurements above; separately verify accepted electrons under the WAND cut.

The supported local correction is an element-selector override attached only to the mu− capture process and only to the known pure-water material. Reusing base selection until oxygen is returned preserves its conditional isotope sampling:

```cpp
class WaterMuonElementSelector : public G4ElementSelector {
 public:
  const G4Element* SelectZandA(const G4Track& t, G4Nucleus* n) override {
    const bool water = t.GetMaterial()->GetName() == "G4_WATER";
    const G4Element* e;
    do { e = G4ElementSelector::SelectZandA(t, n); }
    while (water && e->GetZasInt() != 8);
    return e;
  }
};
// On the mu− stopping-process instance after ordinary process construction:
// muonCapture->SetElementSelector(new WaterMuonElementSelector());
```

Use the actual water-material pointer/composition in a generalized production implementation. The pinned installed `G4ElementSelector.hh` declares virtual `SelectZandA`; [G4HadronStoppingProcess.hh](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/include/G4HadronStoppingProcess.hh), lines89 and121–127, exposes `SetElementSelector` and owns the replacement. This corrects the effective atomic target without replacing nuclear capture kinematics. It does not validate every oxygen nuclear final state or DIO spectral detail. Do not apply an all-oxygen correction to other stopped species without separate measurements.

## New independent stopped-pion hypothesis

[Berridge et al., PRA 75, 034501 (2007)](https://doi.org/10.1103/PhysRevA.75.034501) measured the fraction of stopped π− undergoing **nuclear capture on hydrogen in water** as **(4.45 ± 0.24)×10⁻³**. The primary paper's [author-uploaded full text](https://www.researchgate.net/publication/242310160_p-nuclear_capture_ratio_on_hydrogen_and_oxygen_in_water), pp.1–2, defines this fraction and uses the π0 yield: the hydrogen reaction π−p→π0n is allowed, while the corresponding stopped-pion oxygen charge exchange has negative Q. Thus the physical H-capture probability is about0.445%, not the default selector's30.86%. For the Panofsky ratio and its internal-conversion convention, see [the subsequent radiative-capture note](pion_radiative_oracles.md).

The installed 11.3.0 `G4HadronicAbsorptionBertini.hh` shows inheritance from `G4HadronStoppingProcess`, with no `AtRestDoIt` override. `G4StoppingPhysics.cc`, lines107 and153–160, creates this process and attaches it to π− as well as K− and several hyperons. The [Geant4 constructor source mirrored by BNL](https://eic-code-browser.sdcc.bnl.gov/lxr/source/geant4/source/processes/hadronic/stopping/src/G4HadronicAbsorptionBertini.cc), lines45–51, installs Bertini without replacing the default selector. Exact pinned-runtime target counts remain the decisive check; the v11.3.0 constructor URL was not retrievable by the web tool.

**Prediction pending execution:** if that selector is retained, the H-capture probability is about69 times the measurement, potentially making prompt π0/energetic-gamma events far too common. Run π− at rest in water, score target Z and direct π0/gamma/nucleon daughters before stacking cuts, and compare with the measured probability. A target-selection fix should use the measured π− hydrogen fraction with its uncertainty, and preserve each element's conditional isotope distribution. It should be species-specific: `G4StoppingPhysics` shares the Bertini process across several particle types. Recheck the π−p channel branching separately rather than assuming Bertini reproduces it.

## Range and missing-interaction cross-checks

The [PDG liquid-water muon table](https://pdg.lbl.gov/2020/AtomicNuclearProperties/MUE/muE_water_liquid.pdf) gives kinetic energy200/400/800/1000/1400/2000MeV CSDA mass ranges78.94/179.6/377.1/473.2/661.1/935.3g cm⁻². At2GeV its photonuclear stopping power is0.001 versus total2.218MeV cm² g⁻¹: missing muon nuclear interactions are not an order-one mean-energy-loss explanation in this energy range, although rare topologies can matter.

The [NIST ICRU90 update, TableA.9, PDF p.11](https://www.nist.gov/system/files/documents/2017/04/26/newstar.pdf) gives proton kinetic energy200/400/500/800/1000/2000MeV ranges26.09/82.63/117.6/238.2/326.8/807.9g cm⁻². Convert mass ranges using the simulated density. CSDA is path length, not endpoint displacement or an unconditional proton sample with inelastic reactions. An EM-only diagnostic and a separate inelastic-rate check are preferable to requiring every physical proton track to have the tabulated length.

PhotonSim lacks `G4EmExtraPhysics`; its actual process ledger must establish whether gamma/e±/mu nuclear channels are absent. The [official QGSP_BERT description](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/reference_PL/QGSP_BERT.html) includes those channels in the complete reference list, which is more than its hadron-inelastic constructor alone. [Burgov et al., JETP16(1963)50](https://www.jetp.ras.ru/cgi-bin/dn/e_016_01_0050.pdf) measured oxygen photonuclear absorption with water at18.9–26.6MeV and found a few-percent contribution to total absorption. Missing channels can immediately change prompt electromagnetic energy and topology; neutron-tagging acceptance does not remove that effect. The actual mean/rare-tail WAND light impact requires A/B transport results.

## Scope of neutron/model conclusions

Lack of an HP constructor alone does not prove missing neutron processes: the standard list includes elastic, inelastic and capture models. Under the accepted10µs WAND cut, thermal transport/capture differences are not promoted here into new bugs. Prompt oxygen de-excitation following fast-neutron inelastic scattering still warrants measurement if a large topology discrepancy appears. The [official QGSP_BERT guide](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/reference_PL/QGSP_BERT.html) also cautions that its post11.2 HP variant differs from other HP lists and is under validation; blindly adding that variant is not an independently justified fix.

The [current Geant4 guide](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/physicslistguide.html) notes a Bertini11.2 compatibility switch introduced in11.3.2 because11.3 model changes did not consistently improve thin-target comparisons. This is a model-validation uncertainty for pinned11.3.0, not by itself a PhotonSim bug.
