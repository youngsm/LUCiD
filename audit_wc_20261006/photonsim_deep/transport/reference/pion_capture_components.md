# Geant4 11.3.0 components for stopped-pion capture

Static source review only; no builds, imports, simulations, or numerical scripts were run for these notes. The code sketch below has not been compiled. All source links are pinned to **v11.3.0** and were retrieved through the GitHub connector; installed headers were also inspected at `/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt/include/Geant4/`.

## Why a physics-constructor swap does not supply the missing channel

- [`G4CascadeT31piNChannel.cc`, lines 34–36](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/models/cascade/cascade/src/G4CascadeT31piNChannel.cc#L34): the five π−p two-body final states are π−p, π0n, K0Λ, K0Σ0, and K+Σ−. There is no γn channel. Lines 1710–1714 instantiate the π−p table with those states.
- [`G4ElementaryParticleCollider.cc`, lines 205–212 and 362–412](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/models/cascade/cascade/src/G4ElementaryParticleCollider.cc#L205): ordinary nucleon collisions use the channel table and final-state generator. Its separate low-energy pion absorption branch, lines 557–620, produces an outgoing nucleon against the residual nucleus; it does not add a capture photon.
- [`G4HadronicAbsorptionBertini.cc`, lines 45–51](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4HadronicAbsorptionBertini.cc#L45): the stopping process registers a `G4CascadeInterface`, with minimum energy zero and Precompound de-excitation.
- [`G4StoppingPhysicsWithINCLXX.cc`, lines 97 and 153–160](https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/stopping/src/G4StoppingPhysicsWithINCLXX.cc#L153), and [`G4StoppingPhysicsFritiofWithBinaryCascade.cc`, lines 97 and 154–161](https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/stopping/src/G4StoppingPhysicsFritiofWithBinaryCascade.cc#L154), still register Bertini for π−. Those alternatives change antihadron treatment, not pion capture.

The legacy `G4PionMinusAbsorptionAtRest` had a crude hydrogen radiative branch, but is absent from the pinned installation and is not a supported replacement. `G4PionRadiativeDecayChannel` models free pion decay, not pion capture. No ready-to-register pinned component supplying the required pion radiative capture was found.

## Minimal hydrogen branch

Spuller et al. measured `P = 1.546 ± 0.009`. On p. 479 they define this as the **π0 family**, including π0 Dalitz decay, divided by the **radiative family**, including direct `e+e−n` internal conversion. Thus `1/(1+P)` is the radiative-family probability, not exactly the direct-real-photon probability. A leading correction replaces the wholly absent radiative family by γn with probability `1/(1+P)` and π0n otherwise; it still omits capture internal conversion. The p. 481 internal-conversion ratio correction of 0.999 reflects a cancellation and does not determine the individual missing branching fractions. [Primary paper, Physics Letters B 67 (1977) 479–482](https://doi.org/10.1016/0370-2693(77)90449-X); [full paper](https://muon.npl.washington.edu/exp/WildIdeas/DarkLiterature/piHe/spuller77.pdf).

Use a dedicated `G4HadronStoppingProcess` registered only on π−, with a custom `G4HadronicInteraction` that treats **A=1, Z=1** and delegates other targets to `G4CascadeInterface` configured with `usePreCompoundDeexcitation()`. Replace the old stopping process only in the π− process manager. Do not delete the old shared Bertini process: [`G4StoppingPhysics.cc`, lines107 and153–160](https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/stopping/src/G4StoppingPhysics.cc#L153) also registers it on K− and hyperons. The material-dependent water target selector is a separate required correction.

This preserves Geant4's atomic-cascade/process integration. [`G4HadronStoppingProcess.cc`, lines 139–166](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc#L139) selects the isotope, performs the atomic cascade, and passes its binding energy to the nuclear model; lines 190–211 select/call the registered model. Register one dispatching model, rather than two overlapping models and hoping their selection implements a branching ratio.

Core of an uncompiled **leading two-channel correction** in `ApplyYourself`, with a member `G4HadFinalState result`:

```cpp
// Dispatch here only for stopped pi-, target A=1 and Z=1.
result.Clear();
result.SetStatusChange(stopAndKill);
const double P = 1.546; // Inclusive family ratio; conversion is omitted here.
const G4ParticleDefinition* x = G4Gamma::Definition();
if (G4UniformRand() >= 1.0/(1.0+P)) x = G4PionZero::Definition();
const auto* n = G4Neutron::Definition();
const double W = projectile.GetDefinition()->GetPDGMass()
    + G4Proton::Definition()->GetPDGMass() - projectile.GetBoundEnergy();
const double mn = n->GetPDGMass(), mx = x->GetPDGMass();
const double p = std::sqrt((W*W-(mn+mx)*(mn+mx))
                         *(W*W-(mn-mx)*(mn-mx))) / (2*W);
const G4ThreeVector q = p * G4RandomDirection();
result.AddSecondary(new G4DynamicParticle(
    x, G4LorentzVector(q, std::sqrt(mx*mx+p*p))));
result.AddSecondary(new G4DynamicParticle(
    n, G4LorentzVector(-q, std::sqrt(mn*mn+p*p))));
return &result;
```

This conserves the specified stopped-atom branch energy and momentum and lets the registered π0 decay supply its photons. Include the relevant Geant4 headers and retain normal model initialization, applicability, creator/time metadata, and energy checks. Natural deuterium is **not** an A=1 target and must not enter this two-body branch. This sketch is a suggested correction to the dominant channel omission, not a claim of complete atomic/capture physics.

Minimal validation: in a pure protium target, count direct nuclear-model daughters before neutron transport and π0 decay; verify the specified branch probability, exactly one recoil neutron per two-body branch, the radiative-photon energy from two-body kinematics, and four-momentum/charge/baryon conservation. A complete physical comparison must count the **inclusive families** matching Spuller's definition and include or quantify omitted internal-conversion channels. Then repeat in water with independently validated capture-target fractions. Hydrogen branching and target selection must be tested independently.

## Oxygen: existing de-excitation support, but no turnkey physical spectrum

Adding a hard photon on top of an unchanged nonradiative Bertini capture would double-spend energy. A radiative event must replace the nuclear final state and supply its residual system.

For an inclusive radiative branch at rest, define `W = M(A,Z) + mπ − B`, sample a physically allowed photon four-vector `k`, and set `pResidual = (0,0,0,W) − k`. Its conserved nuclear identity before subsequent emission is `(A,Z−1)`, hence **16N*** for a 16O target. Require `pResidual.mass() >= M(A,Z−1)`; equivalently `Eγ <= (W²−M(A,Z−1)²)/(2W)`. A photon continuum leaves an excited recoil system, not an arbitrarily ground-state nucleus. Individual measured capture lines require their identified residual states.

Available pinned plumbing:

- [`G4Fragment.cc`, lines 99–120](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/util/src/G4Fragment.cc#L99) constructs a fragment from `(A,Z,p4)` and derives excitation from its mass.
- [`G4ExcitationHandler.hh`, line 70](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/models/de_excitation/handler/include/G4ExcitationHandler.hh#L70) exposes `BreakItUp(const G4Fragment&)`; [implementation lines 459–477](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/models/de_excitation/handler/src/G4ExcitationHandler.cc#L459) handles Fermi breakup, evaporation, and photon evaporation. This can transport a conserved excited residual once a capture model has specified it.
- [`G4MuMinusCapturePrecompound.cc`, lines 171–206 and 232–248](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4MuMinusCapturePrecompound.cc#L232) provides a concrete integration example: subtract the escaping particle's four-vector, construct `(A,Z−1)` fragment, invoke Precompound de-excitation, and turn its reaction products into secondaries. Its muon weak-interaction spectrum is **not** a pion-capture model and must not be reused as one.

A gamma plus excited-16N bookkeeping construction conserves total charge, baryon number, and four-momentum, but a generic statistical de-excitation handler does not by itself establish the measured capture-line weights, continuum spectrum, neutron multiplicities, or correlations. The primary oxygen measurements must supply/validate those ingredients. Therefore no one-line oxygen fix is justified here. In particular, do not replace a measured total radiative branching fraction by the same probability above an arbitrary photon-energy threshold; its normalization and low-energy extrapolation must match the cited measurement.
