# LUCiD SK_WAND DATA-mode physics audit

**6 October 2026 · Consolidated report, including the expanded PhotonSim investigation**

## Main conclusion

**There are substantial physical errors in the current PhotonSim source simulation.** The largest newly demonstrated effect is incorrect negative-pion capture targets in water: correcting target selection reduces emitted Cherenkov light by **25.0% in a 200 MeV π⁻ sample**. Negative-muon capture and lifetime are also wrong. Electromagnetic nuclear interactions and radiative pion-capture channels are missing. Separately, an invalid GENIE input can silently produce an entire sample of default 5 MeV electrons through the real production runner.

These findings supersede the first audit pass's assessment that no comparably large source error had been demonstrated. Earlier findings in optical transport, spin correlations, PMT response, and photon truth remain valid and are included below.

**Scope:** SK_WAND DATA mode, including both supplied particle-bomb/single-particle configurations **and GENIE inputs**, as confirmed by the user. All builds, imports, numerical analysis, simulations, and tests ran on **milano CPUs; zero GPUs**. Production code and configurations were not changed. Fixes are isolated prototypes or proposals, distinguished below.

**Published package:** large raw files and local software environments are intentionally omitted from Git, per the user's instruction. Compact results, source, tests and patches are included. See [package contents](README.md), [file manifest](FILE_MANIFEST.json) and [runtime/regeneration guide](packaging/RUNTIME.md) before running a snippet that requires raw fixtures. Recorded scientific measurements are unchanged; packaging validation additionally strengthened the GENIE absent-polarization-branch check.

The numbers below describe specific tested populations. In particular, **emitted-photon changes are not detector-PE changes or measured campaign-wide biases**. Finite tests cannot certify exact agreement with a real detector; unresolved physical and calibration questions are listed explicitly.

## Findings at a glance

| Finding | Demonstrated consequence | Priority / scope | Fix status |
|---|---|---|---|
| [1. Wrong π⁻ capture targets](#1-negative-pion-capture-targets-in-water) | H capture 29.33% versus measured 0.445%; spurious π⁰ flashes; 25.0% source-light reduction after correction at 200 MeV | High; particle and GENIE events containing stopping π⁻ | Target-only prototype executed |
| [2. Wrong μ⁻ capture targets](#2-negative-muon-capture-and-lifetime) | Capture 12.86% versus 18.4%; lifetime 1926.5 versus 1795.4 ns | High; stopping μ⁻ population and Michel timing | Target-only prototype executed |
| [3. Bad GENIE input becomes electrons](#3-invalid-genie-input-silently-becomes-5-mev-electrons) | Real production runner returns success and stores unrelated 5 MeV e⁻ events | High; conditional input failure | Fail-fast fix proposed |
| [4. Missing EM nuclear interactions](#4-electromagnetic-nuclear-interactions-are-absent) | Missing nuclear shower tails; mean source light changes −3.36% at 25 MeV γ, about −0.77% at 2 GeV e/γ | Material missing physics | Constructor fix executed |
| [5. Missing radiative π⁻ capture](#5-radiative-pion-capture-is-missing) | No direct hard capture γ; measured oxygen hard lines alone predict about 12 in the tested sample | Missing prompt topology; remains after target fix | Final-state model proposed |
| [6. Missing decay-spin correlations](#6-spin-correlations-in-pion-and-muon-decay) | Selected Michel forward fraction 0.5297 versus water oracle 0.3835 | Material joint-angular error | Isolated spin/water prototype executed |
| [6.1. GENIE muon spin is discarded](#61-genie-primary-muon-polarization-is-also-discarded) | All 256 imported polarized muons start with zero spin | Interface loss; must accompany spin-aware decay correction | Import/rotation prototype executed |
| [7. Missing high-charge PMT response](#7-high-charge-pmt-compression-is-absent) | Readout stays linear; actual production pulse reaches 1152 PE within 1 ns | Bright-pulse response; calibration needed | Calibrated response proposed |
| [8. Reflected PMT self-encounters](#8-reflected-photons-can-numerically-hit-the-same-pmt-again) | 3.96% of targeted reflected rays affected; small measured ensemble charge change | Small numerical effect | Stable-intersection prototype executed |
| [9. Wrong photon step labels](#9-photon-emitting-step-truth-can-be-wrong) | 1206 wrong step labels among 766427 photons; total photons intact | Fine-grained truth error | Exact step-tag prototype executed |
| [10. Other bounded defects](#10-other-confirmed-defects-with-limited-or-secondary-scope) | Dynamic GENIE ions, threshold discretization, phase-law mismatch, diagnostic truncation | Explicitly bounded below | Mixed |
| [11. Absolute PMT efficiency](#11-unresolved-absolute-pmt-efficiency) | Potentially large normalization uncertainty; convention unknown | Unresolved, **not a confirmed bug** | Needs provenance/calibration |

## 1. Negative-pion capture targets in water

### Physical failure and impact

Geant4's generic element selector weights H:O in water as `2:4.48`, assigning **30.86%** of stopped π⁻ captures to hydrogen. The selected nucleus passes unchanged through the stopping process and cascade; there is no later molecular transfer that corrects the choice.

The measured **final nuclear-capture fraction on H is 0.445 ± 0.024%**. This is already corrected for capture branching and is not an initial atomic-capture fraction. [Berridge et al., PRA 75, 034501](https://doi.org/10.1103/PhysRevA.75.034501).

| Executed sample | Baseline | Water-target prototype |
|---|---:|---:|
| H captures, 3000 cold π⁻ | 880 / 3000 = 29.33% | 19 / 3000 = 0.633% |
| Mean photons per cold π⁻ | 9435.45 ± 243.12 | 1079.61 ± 45.99 |
| H captures, 1000 injected 200 MeV π⁻ | 162 among 497 stops | 2 among 502 stops |
| Mean photons per injected 200 MeV π⁻ | 24807.81 ± 601.12 | 18603.74 ± 515.18 |

Every baseline H capture produced a π⁰. The excessive H assignment therefore creates an excess of prompt π⁰ showers; the conditional final-state branching error is treated separately in finding 5. The **25.0% reduction** in the 200 MeV sample includes ordinary in-flight transport; it is not an estimate for every pion energy, a full GENIE distribution, or detected PE.

### Minimal reproduction and fix

After generating the retained fixtures with [run_capture.sbatch](photonsim_deep/transport/run_capture.sbatch), this physical assertion fails on baseline and passes when the filename is changed to `pi_minus_cold_water.csv`:

```python
import numpy as np
d = np.genfromtxt("pi_minus_cold_baseline.csv", delimiter=",", names=True)
p_h = d["pion_hydrogen"].sum() / d["pion_captures"].sum()
assert abs(p_h - 0.00445) < 0.02
```

**Minimal tested target fix:** install a water-specific final-target selector with `P(H)=0.00445`, `P(O)=1-P(H)`, preserving isotope sampling. **Both the π⁻ species and water-material guards are essential:** the Bertini stopping process is shared with negative kaons and hyperons. This is a measured effective target model, not a microscopic atomic-transfer simulation.

[Selector implementation](photonsim_deep/transport/WaterPionElementSelector.hh) · [registration snippet](photonsim_deep/transport/minimal_water_capture_registration.cc) · [full physical regression](photonsim_deep/transport/capture_regression.py) · [results](photonsim_deep/transport/capture_results.json).

The runtime variant passed the target-fraction oracle; the standalone registration snippet was compile-checked. **This does not fix the missing radiative final states in finding 5.**

## 2. Negative-muon capture and lifetime

### Physical failure and impact

The generic selector also leaves about 31% of stopped μ⁻ assigned to H. In water, muonic hydrogen transfers its muon to oxygen much faster than muon disappearance. Independent water values are **18.4 ± 0.1% nuclear capture** and **1795.4 ± 2.0 ns disappearance lifetime**. [SK 2024, transfer discussion and capture measurement](https://arxiv.org/html/2403.08619v2#S3.SS2); lifetime is in Table XII of the paper.

| 10000 genuinely stopping low-KE μ⁻ per variant | Baseline | Oxygen-target prototype |
|---|---:|---:|
| Nuclear-capture fraction | 12.86 ± 0.335% | 18.17 ± 0.386% |
| Mean disappearance delay | 1926.5 ± 19.4 ns | 1814.9 ± 18.1 ns |
| Tracked Michel daughter, KE > 1 MeV and before 10 µs | 86.49% | 81.53% |

The baseline capture discrepancy is **15.9σ**. Separate 200 MeV samples also fail baseline/pass the water oracle; μ⁺ controls agree with free decay. The Michel fraction is a source-daughter count, **not detector/tagging efficiency**.

### Minimal reproduction and fix

```python
import json
c = json.load(open("capture_results.json"))
assert c["mu_minus_cold_baseline"]["passes_water_oracle"]  # fails
assert c["mu_minus_cold_water"]["passes_water_oracle"]     # passes
```

The [regression](photonsim_deep/transport/capture_regression.py) derives this oracle from actual creator-model IDs and daughter birth times **before** the accepted late cut. It excludes in-flight decays from stopped denominators and avoids mistaking the muon's kill time for its disappearance time.

**Minimal tested fix:** attach [WaterMuonElementSelector.hh](photonsim_deep/transport/WaterMuonElementSelector.hh) to `G4MuonMinusCapture`, selecting O for water and preserving isotopes. This represents rapid transfer at the final-target level. Switching to `G4MuonMinusAtomicCapture` alone retains the same selector problem.

**Residual:** this does not establish the full oxygen decay-in-orbit response. Pinned `G4MuonMinusBoundDecay` uses a boosted free-Michel approximation with binding-energy subtraction and isotropic directions, rather than the oxygen-specific spectrum/asymmetry treatment in the SK reference. Its boost already permits some electrons above the free endpoint. The spectral/PE discrepancy is **unquantified**, and is retained as a validation gap rather than another measured large error. [Pinned source](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc).

## 3. Invalid GENIE input silently becomes 5 MeV electrons

### Failure and production witness

`PrimaryGeneratorAction::SetGenieInput` resets the reader when `Open` fails. `GeneratePrimaries` then falls through to its default electron gun. The production existing-file guard does not prevent an invalid ROOT tree.

The **real** `lucid.production.run_job` with a GENIE configuration, `--detector SK_WAND --skip-genie --skip-lucid`, and real PhotonSim transport returned **exit 0** and wrote two **5 MeV e⁻** events containing **1021 and 1160 photons**. `RooTrackerEntryID=-1` and `IncomingNuPdg=0`. This reproduced with and without the cached count marker. Optical digitization was intentionally skipped; the physics was already replaced in the produced ROOT file.

This is a conditional input-integrity failure, not a measured incidence among the user's valid GENIE files.

### Minimal reproduction and fix

```python
import numpy as np
import uproot
with uproot.recreate("wrong_tree.gtrac.root") as f:
    f["unrelated_tree"] = {"x": np.array([1], dtype=np.int32)}
```

Then select that file:

```text
/gun/clearPrimaries
/gun/genieInput /absolute/path/to/wrong_tree.gtrac.root
```

Generating an event produces the unrequested default electron. [Full production reproducer](photonsim_deep/primaries/repro_run_job_wrong_tree.py) · [Slurm launcher](photonsim_deep/primaries/run_production_validation.sbatch) · [persisted results](photonsim_deep/primaries/production_wrong_tree_results.json).

**Minimal proposed fix:** fail immediately when an explicitly requested source cannot be opened, and validate all required branch bindings before reporting success:

```cpp
if (!fGenieReader->Open(path))
    G4Exception("PrimaryGeneratorAction::SetGenieInput", "BadGenieInput",
                FatalException, "Cannot load requested rooTracker input");
```

The fatal-exception patch is proposed, not applied or compiled.

## 4. Electromagnetic nuclear interactions are absent

PhotonSim registers EM option 4 and hadronic constructors but omits **`G4EmExtraPhysics`**. Actual process managers show `GammaGeneralProc{nuclear=NULL}` and no `electronNuclear`, `positronNuclear`, or `muonNuclear`. These processes belong to the complete reference list. [Geant4 QGSP_BERT guide](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/reference_PL/QGSP_BERT.html).

| Source; events per variant | With added constructor | Mean source-light change |
|---|---|---:|
| 25 MeV γ; 2000 | 75 photonuclear events; restored near-zero-light tail | −3.36%, 7.81σ |
| 2 GeV γ; 400 | 30.75% of showers contain photonuclear interactions | −0.771%, 3.05σ |
| 2 GeV e⁻; 400 | 28.25% of showers contain photonuclear interactions | −0.775%, 5.46σ |
| 2 GeV μ⁺; 1000 | No direct muon-nuclear interaction observed | No significant change |

The GeV percentages count an interaction **anywhere in the shower**, not the primary interaction probability. This is particularly a missing-tail/topology problem; a large universal light-scale error is not demonstrated. Adding the constructor does not independently calibrate all restored cross sections.

**Minimal executed fix:**

```cpp
#include "G4EmExtraPhysics.hh"
// Alongside the existing EM constructor:
RegisterPhysics(new G4EmExtraPhysics(0));
```

**Minimal test:** require a non-null gamma nuclear process and the missing charged-lepton nuclear processes after initialization. The [executed test](photonsim_deep/transport/test_new_findings.py) checks both baseline/fixed process dumps and restored low-light events. [Fixture launcher](photonsim_deep/transport/run_transport.sbatch) · [patch](photonsim_deep/transport/minimal_em_extra_fix.diff) · [results](photonsim_deep/transport/em_results.json).

## 5. Radiative pion capture is missing

**Hydrogen:** all 880 baseline H captures produced π⁰+n. The pinned cascade table has no γ+n channel. The measured Panofsky ratio ≈1.546 implies about **39.3% electromagnetic-capture-family branching**. That family includes a small internal-conversion component; it is not exactly a real-photon-only branching fraction. After the target correction, this family affects approximately **0.175% of all water pion stops**. [Primary capture measurement](https://doi.org/10.1103/PhysRevA.75.034501).

**Oxygen:** there were **zero direct capture γ above 10 MeV among 2981 O captures** after target correction. Measured hard components alone have branches **0.15 ± 0.03% at 128.10 MeV** and **0.25 ± 0.06% near 117–121 MeV**, predicting about **12 photons** in that sample before the continuum. Shared normalization errors are retained; using their sum conservatively gives a 2.1% zero-count probability even at a three-error-bound low rate. The quoted total 2.24 ± 0.48% branch includes an extrapolation below 50 MeV and must not be called a directly measured above-50-MeV rate. [Bistirlich et al., author manuscript, printed p. 24](https://escholarship.org/content/qt5835t658/qt5835t658.pdf).

**Minimal test:** count direct hard γ daughters in the retained stopped-pion ledger:

```python
import numpy as np
d = np.genfromtxt("pi_minus_cold_water.csv", delimiter=",", names=True)
assert d["pion_hard_gamma"].sum() > 0  # fails: observed zero
```

[Physical oracle with measured uncertainties](photonsim_deep/transport/test_radiative_oracle.py) · [results](photonsim_deep/transport/radiative_results.json) · [source/model hooks and primary references](photonsim_deep/transport/reference/pion_radiative_oracles.md).

**Minimal adequate fix proposal:** a dedicated π⁻ stopping interaction can sample measured H branching with energy-conserving two-body γ+n/π⁰+n kinematics, while preserving shared kaon/hyperon processes. Oxygen needs measured radiative lines/continuum plus a consistent recoiling excited residual nucleus. Adding a photon to an unchanged absorption event would violate its energy budget. **Neither radiative final-state fix was implemented or validated.**

## 6. Spin correlations in pion and muon decay

PhotonSim registers ordinary `G4DecayPhysics` without the spin-aware pion/muon decay processes. It loses the joint angular correlation between a pion-decay muon and its Michel positron. This is separate from the accepted approximation of unpolarized **optical scattering**.

For 2000 stopped π⁺ chains, selecting positrons above 40 MeV:

| Forward hemisphere fraction relative to the muon's birth direction | Value |
|---|---:|
| Baseline | 0.529670 |
| Independent water Michel oracle | 0.383488 |
| Isolated spin + water-depolarization prototype | 0.384522 |

The baseline discrepancy is about nine reference standard errors. The water oracle uses measured positive-muon polarization retention adopted by SK and a leading-order Michel distribution; radiative/finite-mass corrections do not explain the discrepancy. [SK polarization treatment](https://arxiv.org/html/2403.08619v2).

Production reachability was checked: 20 actual 1 GeV π⁺ events contained three π→μ→Michel chains with Cherenkov-emitting muons. The controlled stopped-pion test itself has subthreshold muons; the failure is **joint angular physics**, not a demonstrated total-light loss or known campaign frequency.

**Minimal physical test**, using retained fixtures:

```bash
python audit_wc_20261006/photonsim_physics/test_spin_physics.py baseline # fails
python audit_wc_20261006/photonsim_physics/test_spin_physics.py water    # passes
```

**Minimal adequate fix:** register `G4SpinDecayPhysics` after ordinary decay, include calibrated stopping-water depolarization, and add `"DecayWithSpin"` to LUCiD's `DECAY_PROCESSES` categorization. The illustrative executed μ⁺ model retains full spin with probability 0.718 and randomizes the remainder. Scaling the polarization vector by 0.718 is insufficient because the relevant Geant4 decay channel uses it as an axis, not an asymmetry magnitude.

[Minimal patch](photonsim_physics/minimal_spin_fix.diff) · [test](photonsim_physics/test_spin_physics.py) · [results](photonsim_physics/spin_results.json). The PhotonSim variants were executed; the LUCiD process-category integration change remains proposed. Negative-muon bound-decay response needs separate treatment, as described in finding 2.

### 6.1 GENIE primary-muon polarization is also discarded

Pinned GENIE 3.04 QE/MEC/common lepton generators populate outgoing-lepton polarization, and ordinary ROOTRacker export writes it to `StdHepPolz`. PhotonSim's reader ignores that branch and never calls `G4PrimaryParticle::SetPolarization`, so imported muons start with zero spin. This follows the actual source chain, not an assumption that all ROOTRacker files must contain polarization. [GENIE primary-lepton utility](https://github.com/GENIE-MC/Generator/blob/R-3_04_00/src/Physics/Common/PrimaryLeptonUtils.cxx#L44) · [exporter](https://github.com/GENIE-MC/Generator/blob/R-3_04_00/src/Apps/gNtpConv.cxx#L2387).

**Executed witness:** all **256 polarized muon primaries** had zero baseline spin. The isolated reader/generator repair restored unit polarization, preserved momenta bit-for-bit, and retained the μ⁻/μ⁺ helicity under common event rotation to **6.66×10⁻¹⁶**. Beam-direction and missing-optional-branch cases also passed.

```python
import numpy as np
a = np.genfromtxt("spin_baseline.tsv", delimiter="\t", names=True)
mu = np.abs(a["pdg"]) == 13
spin = np.column_stack([a[k] for k in ("spin_x", "spin_y", "spin_z")])[mu]
np.testing.assert_allclose(np.linalg.norm(spin, axis=1), 1.)  # baseline fails
```

**Minimal tested fix:** read optional `StdHepPolz`, preserve zero when absent/unset, apply exactly the same rotation to momentum and polarization, and call `SetPolarization`. Polarization is dimensionless: do not apply MeV/GeV conversion or multiply by 1/2. Production integration should diagnose malformed/nonfinite values and prevent stale branch data.

[Executed patch](photonsim_deep/primaries/spin_preserving_fix.diff) · [fixture/build launcher](photonsim_deep/primaries/run_spin.sbatch) · [regression](photonsim_deep/primaries/analyze_spin.py) · [results](photonsim_deep/primaries/spin_results.json) · [pinned source review](photonsim_deep/primaries/genie_polarization_review.txt).

This is an independent interface defect, but default spin-insensitive decay already erases its observable effect; **do not count a second measured detector bias**. Imported spin must accompany spin-aware decay and water depolarization. GENIE's inspected unit-longitudinal-helicity convention is itself an approximation, so preserving it does not certify exact finite-mass neutrino–muon spin physics.

## 7. High-charge PMT compression is absent

The actual SK_WAND readout remains linear: 300000 trials at each of 30, 200, 1190, and 5000 input PE show relative gain **1.000093 at 1190 PE**, normalized to 30 PE. SK calibration measures PMT charge compression exceeding 10% above 1000 PE; that does not define the correction at exactly 1190 PE or for every pulse width. This is separate from the accepted one-PE charge-spectrum shape. [SK calibration, §3.1.7 and Fig. 19](https://arxiv.org/pdf/1307.0162).

A ten-event production-bomb cohort contains one ≥1000-PE digit among 58690 digits. It has **1190 truth PE**, including **1152 PE within a 1 ns window**; the central 90% span is 0.766 ns before PMT TTS. This establishes a prompt high-occupancy production witness. Its 0.413% share of cohort charge is **not** a missing-charge estimate or reliable campaign frequency.

**Minimal current-code probe:**

```python
import json
import numpy as np
from lucid.simulation.digitizer import apply_readout_resolution, resolve_model_config
cfg = json.load(open("config/SK_WAND_physics_config.json"))
model = resolve_model_config(cfg["digitizer"])
rng = np.random.default_rng(1901190)
for n in (30, 1190):
    q, _ = apply_readout_resolution(np.full(300_000, n),
                                   np.zeros(300_000), model, rng)
    print(n, q.mean() / n)  # remains approximately one
```

**Minimal physically justified fix:** add a measured, detector-period-appropriate charge-response function in `apply_readout_resolution`, before charge-dependent timing; preserve true PE. Pulse-width dependence needs measurement or a supported model. An arbitrary clamp or universal 10% reduction is not an established fix.

[Executed probe](digitizer/probe_high_charge_response.py) · [results](digitizer/high_charge_results.json) · [production pulse timing](end_to_end/bright_digit_timing.json).

## 8. Reflected photons can numerically hit the same PMT again

The float32 sphere discriminant `b*b - 4*a*c` suffers cancellation at SK distances. The computed entry point can lie about 1 mm inside a PMT; the 0.1 mm reflection nudge leaves the ray inside. A later search can then treat its exit as another surface encounter, repeating a reflection/absorption decision.

With pinned JAX/jaxlib **0.4.38**, **1780/44896 targeted reflected rays (3.96%)** re-encountered their original sensor; the stable formula gave **0/44906**. JAX 0.11 independently reproduced it. This is within the accepted spherical-PMT model and separate from endpoint-search approximation.

A six-million-photon paired DATA test under JAX 0.11 found corrected-minus-current charge **−0.0744 ± 0.0182%**, with no significant late-light difference. This measures the entire discriminant replacement, including changed first intersections/normals; it is not an isolated measurement of repeat-encounter loss or a Geant4 event-cohort bias.

**Minimal reproducer**, inside an allowed allocation:

```bash
audit_wc_20261006/runtime_jax0438/venv/bin/python \
  audit_wc_20261006/geometry/repro_reflection_selfhit_pinned.py
# Repeat with --fixed for the isolated in-memory correction.
```

The pinned witness spuriously re-encounters PMT 1027 after 0.285 mm; the corrected ray reaches the next real surface 0.703 m away.

**Minimal tested fix**, retaining existing root selection:

```python
perp = oc - (jnp.sum(oc * ray_d, axis=1) / a)[:, None] * ray_d
discriminant = 4 * a * (sensor_radius**2 - jnp.sum(perp**2, axis=1))
```

[Reproducer](geometry/repro_reflection_selfhit_pinned.py) · [pinned population results](runtime_jax0438/fused_results_jax0.4.38.json) · [charge comparison](geometry/charge_bias_summary.json).

## 9. Photon emitting-step truth can be wrong

PhotonSim infers the emitting step from photon birth time. An independent creation observer finds **1206 wrong step labels among 766427 photons**. Every wrong photon already has a Geant4 birth time slightly later than its actual emitting step's endpoint, by at most 0.000562 ns; time-based lookup chooses one to three later steps of the same parent. First-step timestamp subtraction is not the cause.

Total light and parent-track ownership survive. The isolated exact-step-tag fix yields **zero wrong labels**, correct segment counts, and bitwise unchanged photon position/direction/wavelength/time arrays.

**Minimal regression** on the retained gamma fixture:

```python
import numpy as np
import uproot
t = uproot.open("gamma_tag_baseline.root")["OpticalPhotons"]
a = t.arrays(["Photon_SegmentIndex", "Segment_NCherenkov"], library="np")
expected = a["Segment_NCherenkov"][0]
actual = np.bincount(a["Photon_SegmentIndex"][0], minlength=len(expected))
np.testing.assert_array_equal(actual, expected)  # fails; fixed fixture passes
```

**Minimal tested fix:** attach `parent.GetCurrentStepNumber()-1` to each optical secondary at creation and translate it through the parent's segment-base offset. Preserve any existing user track information. The full test compares exact per-photon tags, since count equality alone can hide swaps.

[Executed correction](photonsim_deep/emission/step_tag_prototype.diff) · [analysis](photonsim_deep/emission/analyze_step_tag.py) · [results](photonsim_deep/emission/step_tag_results.json) · [launcher](photonsim_deep/emission/run_step_tag.sbatch).

## 10. Other confirmed defects with limited or secondary scope

### 10.1 Supported GENIE ion primaries can disappear

The reader uses only `FindParticle(PDG)` and silently skips null results. Some supported ions require dynamic `G4IonTable` materialization. A fixture with status-1 e⁻ and O16 injects only the electron; calling `GetIon(1000080160)` first restores the **1.3426646 MeV O16**. Ion-cache history can affect whether it is present.

**Minimal fix:** resolve supported nuclear PDGs through `G4IonTable`, then fail with entry/PDG context for unsupported required primaries. Validate isomer/anti-ion handling rather than silently skipping them.

```cpp
auto* pdef = particleTable->FindParticle(p.pdg);
if (!pdef && p.pdg >= 1000000000)
    pdef = particleTable->GetIonTable()->GetIon(p.pdg);
if (!pdef) { /* fatal error with PDG and entry context */ }
```

[Fixture generator](photonsim_deep/primaries/prepare.py) · [actual primary probe](photonsim_deep/primaries/primary_probe.cc) · [results](photonsim_deep/primaries/results.json).

Source review establishes status-1 recoil nuclei on GENIE coherent paths, but actual campaign frequency and detector-light impact are unmeasured; the example oxygen recoil is subthreshold. Standard GENIE oxygen-deexcitation photons are already status 1 and are retained. A status-15 remnant is not automatically a missing final-state ion. [Pinned GENIE source provenance](photonsim_deep/primaries/reference/GENIE_deexcitation_sources_manifest.json).

### 10.2 Near-threshold Cerenkov step discretization

With unchanged PhotonSim and its original index table, 10000 electrons at 0.3 MeV produce **8891 photons** at the default 10% maximum beta change versus **9767** at 0.1%. Independent dense-index controls converge between 0.1% and 0.01%; default expected light is **9.45% low**, but only **about 0.093 emitted photons per electron**. No large normal-event bias is demonstrated.

**Minimal reproducer:** [default macro](photonsim_deep/cherenkov/min_default.mac), [small-step macro](photonsim_deep/cherenkov/min_small.mac), [launcher](photonsim_deep/cherenkov/run_minimal.sbatch).

**Minimal candidate setting:** `/process/optical/cerenkov/setMaxBetaChange 0.1`. The argument is **percent**. Check energy-dependent convergence and performance before adopting it broadly.

### 10.3 Sparse refractive-index quadrature

The ten-point table makes G4's cumulative integral interpolation generate negative emission means immediately above the physical threshold; emission is clipped to zero over roughly β=0.73165–0.73370. At the slightly larger β=0.734, the mean is positive but suppressed: **0.006591 versus analytic 0.025961 photons/mm**. For a linear index interval, the exact integral is:

```text
integral[dE / n(E)^2] = (E_b - E_a) / (n_a * n_b)
```

**Minimal test:** [mean_yield.cc](photonsim_deep/cherenkov/mean_yield.cc), launched by [run_mean.sbatch](photonsim_deep/cherenkov/run_mean.sbatch). **Minimal mitigation:** densify the existing piecewise-linear index curve, preserving its physical values.

Whole-track errors are small: **0.0843% at 0.3 MeV e⁻**, **0.000854% at 10 MeV e⁻**, and **0.000936% at 1 GeV μ⁻**. This is separate from step discretization. Production QE-weighted comparisons include the accepted wavelength clipping.

### 10.4 Water asymmetric-scattering phase law does not match its fitted coefficient

The coefficient comes from an SK forward law `p(mu)=2*mu` on `[0,1]`, while code samples Henyey–Greenstein with `g=.95`: mean cosine **.95 instead of 2/3**. The estimated affected fraction is small, about **0.30% over 20 m**, spectrum/QE/absorption weighted; this is not a measured total-charge bias.

```python
import jax.numpy as jnp
from lucid.wavelength.scattering import hg_sample_cos_theta
u = (jnp.arange(1_000_000) + .5) / 1_000_000
print(hg_sample_cos_theta(u, .95).mean(), jnp.sqrt(u).mean())
```

**Minimal fix:** sample `mu=sqrt(U)` with uniform azimuth for this fitted law, or refit the coefficient and HG law together. Changing `g` alone does not reproduce the law. [Executed optical-reference test](physics_crosscheck/verify_optical_reference.py) · [results](physics_crosscheck/optical_reference_results.json) · [SK optical calibration](https://arxiv.org/pdf/1307.0162).

### 10.5 Diagnostic scat10x transport truncation

The intended fiTQun attenuation workflow for `SK_WAND_physics_config_scat10x.json` still uses `K=12`. Relative to `K=96`, the probe loses **31.8% at 300 nm**, **11.5% at 325 nm**, and **2.63% at 350 nm**. Canonical SK_WAND showed no additional detected light between `K=12` and `K=24` in its test.

**Minimal reproduction:** run [check_statistics.py](simulator_statistics/check_statistics.py) with `AUDIT_PHYSICS_CONFIG=config/SK_WAND_physics_config_scat10x.json`, `AUDIT_K=96`, and a distinct `AUDIT_RESULT_NAME` to preserve baseline evidence. [Results](simulator_statistics/scat10x_K96_results.json).

**Minimal fix:** expose the existing `--K` option through the diagnostic job config and choose a convergence-checked value. `K=96` had no detected tail in its last 24 iterations in this sample. This is a large **diagnostic-config** effect, not a demonstrated canonical SK_WAND deficit.

### 10.6 Secondary metadata paths

| Defect | Minimal witness | Minimal fix / scope |
|---|---|---|
| Pileup drops segment group IDs | Two streams `[0,0]` become `[0,1,2,3]`, not `[0,0,1,1]`; [reproducer](production_integrity/repro_pileup_group_ids.py) | Preserve IDs and offset groups between streams. No charge/timing effect; off supplied single-vertex path. |
| Heterogeneous gun energy truth is overwritten | μ⁻ 500 + e⁺ 25 + γ 10 MeV records 10 rather than 535 MeV; [probe](photonsim_deep/primaries/primary_probe.cc), `multigun.tsv` fixture | Reset then sum successfully injected KE. Particles transport correctly; supplied single-particle configs, bomb and GENIE branches are unaffected. |

## 11. Unresolved absolute PMT efficiency

`SK_QE.json` is used as incident-photon detection probability with `qe_corrections=1`; its value at 400 nm is **0.2261907458**. There is no separate collection-efficiency factor. Neither the file, git provenance, nor the user establishes whether the curve already includes collection efficiency.

**Conditional diagnostic, not a confirmed bug:**

```python
from lucid.wavelength.medium import load_qe_curve
q = float(load_qe_curve("config/pmt/SK_QE.json")(400.))
print(q, q * .73, 1 / .73 - 1)
# 0.2261907458, 0.1651192445, 0.3698630137
```

If this is bare cathode QE, collection efficiencies of .73 or .67 imply current direct-PE normalization **37% or 49% too high**. If collection is included, another factor would double-count it. Similarity to a published peak cannot settle the convention; published collection numbers also depend on illumination/aperture.

**Resolution:** recover the exact curve provenance or compare absolute response with calibrated photon flux. Then use the appropriate full detection-efficiency curve or measured collection correction. **No arbitrary factor was applied.** [Provenance/reference analysis](physics_crosscheck/findings.md) · [SK calibration](https://arxiv.org/pdf/1307.0162).

## Validation coverage and remaining limits

### Independent checks that passed

| Area | Executed evidence | What this establishes |
|---|---|---|
| Optical laws and QE | 27 checks pass under JAX 0.11 and pinned 0.4.38; full DATA detection matches single-QE expectation within 1.44σ | No reproduced double-QE error in the current tested path |
| Photon exclusivity and transport iteration | 65536 photons; 13017 valid detections; no duplicate valid hit; canonical K=12 versus K=24 has no extra detected tail | Tested orchestration preserves exclusivity and converges for this canonical sample |
| Flattening, buckets and padding | Exact synthetic candidate/iteration/photon/segment labels | No tested reordering or padding leakage |
| Primary bomb generation | 20000 actual-config events; 59806 primaries; multiplicity/species/KE/directions/energy sums checked | Config routing and primary kinematics pass these independent checks |
| GENIE momentum interface | 20000 correlated fixtures; common rotations preserve dot products/chirality, units, KE, entry order | No independent-particle rotation or GeV/MeV failure in the tested route |
| Photon creation → ROOT → LUCiD | 4262268 records from 2131134 distinct histories run in two storage modes | No missing/duplicate photons; creation position/direction/wavelength conserved; max time error 1.819e−12 ns |
| Output lifecycle | Streaming/buffered arrays bitwise equal; production bombs span many 100k-photon chunks; zero-light/reset/ancestor cases pass | No tested chunk/event-loss defect; throwaway G4 optical absorption does not thin source again |
| Cerenkov model | Dispersion knots, conditional spectra, cone angles and emission times checked independently | Formula-level agreement within the specified model, subject to threshold limits above |
| Muon range | 200/400/1000/2000 MeV means agree with independent PDG water CSDA within 0.65% | No large stopping-range/unit-scale defect in these samples |
| Pion kill-and-replace | 5600 A/B events; all 2645 replacements resume; exact position/time; every continuation chain terminates once | No consequential replacement bug established; light differences <1.44σ with 6–12 percentage-point confidence widths |
| Actual ROOT → HDF5 datasets | Thirty single-particle and ten bomb events; ancestry, charge attribution, units and causal timing checked | No orphan detected PE or material charge-accounting failure in these cohorts |
| Source-world boundary census | Ten bomb showers × 17 placements; 90420856 inside-ID prompt photons; none with prior full-water-tank exit | No observed outside-tank returning ancestry in this sample; these are ten showers, not 170 independent events |

Detailed machine-readable evidence: [optical checks](runtime_jax0438/transport_checks.json), [statistics](simulator_statistics/results.json), [primary injection](photonsim_deep/primaries/results.json), [emission](photonsim_deep/emission/results.json), [pion replacement](photonsim_deep/pion_replacement/results.json), [production integration](end_to_end/results_bomb.json), [boundary census](photonsim_deep/boundary/history_census.json).

### Explicitly unvalidated physical questions

- Absolute PMT efficiency, calibrated angular/reflection response, and pulse-dependent high-charge response.
- Oxygen bound-muon energy/spin distributions beyond the capture-rate correction; full atomic transfer/cascade details.
- Differential nuclear final states and cross sections over all species/energies. Presence of a Geant4 process is not experimental validation.
- Exterior steel/concrete/rock effects. The source world is homogeneous water; no full external-material variant was built. Crossing the ID into outer-detector water is not itself a bug.
- Neutrino interaction physics inside GENIE itself. This audit interrogates its LUCiD/PhotonSim handoff and relevant pinned source semantics, not the entirety of GENIE cross-section and nuclear modeling.
- Production-container/backend equivalence for every small numerical effect. Pinned JAX checks were performed, but the complete container did not successfully launch; the six-million-photon charge study uses JAX 0.11.

Source-reviewed input risks without executed physical witnesses remain separate: original PhotonSim ignores macro-command failure and can return success; stale output reuse may then be possible. The reader's fixed 4096-particle arrays are clamped only after `GetEntry`, and malformed branch/oversized-entry handling needs validation. These are not promoted to demonstrated normal-data physics failures.

### Accepted approximations and excluded paths

The following were not reclassified as bugs: endpoint PMT search; spherical PMTs; negligible QE wavelength clamping within the emission band; 10 µs / 100 µs late cuts with no neutron tagging; `c/1.33` propagation speed; unpolarized optical scattering; one-PE charge-spectrum shape. The user also excluded supernova chunks, direct CLI, EventID, input-vertex truth, and realistic-mode TTS from production scope.

## Reproduction, versions, and handoff

| Component | Audited version / setting |
|---|---|
| LUCiD | `20df3094bd563162d40c6ca34c5ad25e1e1d3656` |
| PhotonSim | v1.0.2, `9232a7dae46ae36d9c81dd5f1b4e9f7653803514`; [preserved source](photonsim_physics/upstream/PhotonSim.cc) |
| Detector | `config/SK_WAND_geom_config.json`, `config/SK_WAND_physics_config.json` |
| Physics runtime | Geant4 11.3.0; ROOT 6.34.02 rather than production container ROOT 6.30 |
| Optical runtimes | JAX 0.11; separate Python 3.12/JAX-jaxlib 0.4.38 confirmation |
| Resource use | All execution on milano CPU allocations; zero GPUs |

Use the preserved `sbatch` launchers. To run an individual snippet, first obtain an allowed allocation:

```bash
srun --partition=milano --account=neutrino:ml-dev@milano --qos=preemptable \
  --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --pty bash --noprofile --norc
```

Then use the interpreter/environment documented by its lane. From the repository root set `JAX_PLATFORMS=cpu`, `PYTHONPATH=$PWD`, and `PYTHONDONTWRITEBYTECODE=1`; retain that absolute `PYTHONPATH` if changing directories. Launch full scripts from the repository root. Small snippets reading bare CSV/JSON/ROOT/TSV filenames must run in the linked fixture directory, or use full fixture paths. The isolated pinned environment additionally uses `PYTHONNOUSERSITE=1`. **Do not run the tests on login CPUs.** Roma with its matching account also satisfies the user's CPU constraint.

The PhotonSim source/build macros and CSV/ROOT results are retained, but this directory does not bundle a portable Geant4 installation. Exact CVMFS and existing Apptainer-wrapper paths are in the launchers and [geant4_env.sh](photonsim_physics/geant4_env.sh). The headless main removes GUI startup and checks command status; it cannot reproduce the original main's ignored error exit status. Transport ledgers count creation and kill optical tracks afterward, so comparisons are independent ensembles rather than paired event identity.

Key completed allocation IDs: transport **39971259**; capture **39971585**; physical assertions/fix compilation **39972145**; radiative check **39973045**; final fixture validation **39973141**; primary generation **39971299**; production GENIE failure **39972545**; GENIE polarization **39973136**; emission **39971140/39971237/39971301**; exact step fix **39971599**; pion replacement **39971593**; boundary census **39971218**; Cerenkov **39971297/39971407/39971598/39971973/39972343/39972392**. Earlier pinned-runtime, geometry, readout and integration logs remain beside their scripts.

Some small physical assertions intentionally fail on baseline. Comparison drivers instead assert that baseline fails and an isolated correction passes. Failed setup attempts are retained as logs and are not counted as validation.

**Recommended implementation order:** repair and regression-test the water capture targets and GENIE input failure first; restore EM nuclear and spin handling; implement measured radiative-capture final states; apply the small geometry/step-provenance fixes; resolve absolute efficiency and high-charge calibration before claiming detector-level physical agreement. This report is the consolidated handoff; lane files are supporting evidence and working notes.
