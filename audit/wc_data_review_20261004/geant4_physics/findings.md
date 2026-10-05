# PhotonSim physics findings for SK_WAND data mode

The default published container pins PhotonSim `v1.0.0`, commit `ff73224a669fdc3598dc2993a8763223b11a3e5b`, in `.github/workflows/container.yml:25`.
The container requests Geant4 11.3, and accepted experiments use Geant4 11.3.0 with these unchanged source files.
The audit executable removes GUI initialization and fixes startup seeds; each macro resets seeds to `314159 271828`.
All physics, source generation, stepping and output code in the baseline are unchanged.
`PHOTONSIM_BIN` is unset here, so an unseen custom host binary is not covered by this provenance check.
All source simulations ran in authorized milano CPU allocation `39894010` on `sdfmilan269`, with no GPUs; final ROOT regressions, tail summaries and native material inspection ran in replacement milano allocation `39895962`.
Job `39893936` was an unsuccessful initial link attempt.
All source inspection and edits after the user tightened the no-login-operations rule ran inside these authorized milano allocations.
Every output used as evidence is a valid ROOT TTree verified with uproot.
`run_photonsim.sh` repairs CVMFS ROOT's stale module import path through an ephemeral Apptainer bind, with no physics changes.

## Confirmed: 10 microsecond cutoff destroys delayed light

Affected source: [SteppingAction.cc:102](https://github.com/cesarjesusvalls/PhotonSim/blob/ff73224a669fdc3598dc2993a8763223b11a3e5b/src/SteppingAction.cc#L102).
The first step of every newly created non-optical track kills it if its creation time exceeds 10000 ns.
Optical photons already generated during that first step can survive, so this produces a corrupted light yield and missing parent information instead of a clean event-window cut.
Identical 2.2 MeV gammas launched at zero and 200 microseconds should differ only in absolute times because the medium and physical processes are time independent.
Thirty prompt events produced 6716 photons, versus 454 delayed photons, a 93.2400% deficit.
Deleting the cutoff in an isolated audit source copy restored exactly 6716 delayed photons, equal to the prompt result with identical seeds.
An additional 100-event 1 MeV neutron sample produced 1962 original photons and only three registered capture gammas, versus 21142 photons and 100 registered capture gammas after removing the cutoff.
This is a 90.7199% deficit in the original neutron sample.
The neutron comparison measures an ensemble effect because transporting additional tracks changes subsequent random consumption.
The paired gamma experiment is the controlled time-translation proof.
SK_WAND inherits this missing source light before its optics, digitizer or trigger can choose an acquisition window.

Minimal test, using `gamma_prompt.mac` and `gamma_delayed.mac`:

```python
prompt = uproot.open('gamma_prompt.root')['OpticalPhotons']['NOpticalPhotons'].array(library='np')
delayed = uproot.open('gamma_delayed.root')['OpticalPhotons']['NOpticalPhotons'].array(library='np')
assert delayed.sum() >= 0.95 * prompt.sum()
```

Minimal fix: remove the unconditional kill at `SteppingAction.cc:102-105` and apply an intentionally selected acquisition window after physical photon generation.
Evidence: `comparison_results.json`, `gamma_prompt.log`, `gamma_delayed.log`, `neutron.log` and their `_fixed` counterparts.

## Confirmed: pion replacement teleports the physical particle backwards

Affected source: [SteppingAction.cc:149](https://github.com/cesarjesusvalls/PhotonSim/blob/ff73224a669fdc3598dc2993a8763223b11a3e5b/src/SteppingAction.cc#L149).
After completing a scattering step, a pion is killed and replaced at the previous stored position and time, rather than its current post-step position and time.
The stored position/time are not updated on the first step, so some second-step replacements restart at the original vertex.
A replacement used to split truth labels must start at the actual endpoint of the parent segment, preserving physical continuity.
Twenty 1 GeV `pi+` events produced 61 replacements, with a maximum endpoint-to-birth gap of 3352.185871 mm and median gap of 0.433443 mm.
The minimal audit-only fix produced 59 replacements in a new sample, with exactly zero gap in every case.
The samples need not have identical replacement counts because repairing the real trajectory changes subsequent transport and random history.
This changes downstream photon emission locations, and applies to charged pions in SK_WAND data production.

Minimal regression:

```python
# For each TrackInfo_CreatorProcess starting with 'Deflection_':
parent = TrackInfo_ParentTrackID[replacement]
last = np.flatnonzero(Segment_TrackID == parent)[-1]
assert np.linalg.norm(replacement_vertex - segment_endpoint[last]) < 1e-6
```

Minimal fix:

```cpp
G4ThreeVector kinkPosition = track->GetPosition();
G4double kinkTime = track->GetGlobalTime();
```

The full executable test is `test_source_physics.py::test_pion_deflection_preserves_position`.
`pion_deflection.mac` reproduces the physical event sample.
`source_results.json` lists every observed original gap, and `comparison_results.json` summarizes the repaired sample.

## Confirmed: GENIE isotropic source has a strong positive-z bias

Affected source: [PrimaryGeneratorAction.cc:113](https://github.com/cesarjesusvalls/PhotonSim/blob/ff73224a669fdc3598dc2993a8763223b11a3e5b/src/PrimaryGeneratorAction.cc#L113), reached from `lucid/production/generate_macro.py:385`.
It draws a uniform axis and uniform rotation angle, then coherently rotates all final-state momenta.
The coherent rotation preserves physical correlations, but this axis-angle distribution is not uniform over rotations.
Rodrigues' formula predicts mean cos(theta) equal to 1/3 when it rotates an initial `+z` direction, whereas an isotropic source requires zero.
The real GENIE rooTracker path, with 20000 forward neutrino events, produced mean cos(theta) `0.3321303`, positive-z fraction `0.70395`, and second moment `0.4660071`.
The physical expectations are zero, one half and one third respectively.
The upper cos(theta) bin from 0.8 to 1 contained 6366 events, or 31.83%, instead of approximately 10%.
The fixed source produced mean cos(theta) `-0.0034010`, positive-z fraction `0.4971`, and second moment `0.3333844` in 20000 events.
LUCiD production defaults `apply_rotation` to false at `lucid/production/run_job.py:296-301`, so ordinary production retains this bias.
A separately requested correct isotropic Python rotation can remove the marginal orientation bias, but is not the default.
This applies to GENIE configurations requesting isotropic directions, rather than deliberate beam or physical-supernova directions with `genieIsotropic false`.

Minimal regression:

```python
z = np.array([v[0] for v in tree['TrackInfo_DirZ'].array(library='np')])
assert abs(z.mean()) < 0.02
assert abs((z > 0).mean() - 0.5) < 0.02
```

Minimal fix: retain the uniform axis and draw angle with Haar density proportional to `sin(angle / 2)**2`, or use a uniform unit quaternion.
The executed audit-only implementation uses rejection sampling:

```cpp
do {
  fGenieRotAngle = 2.0 * M_PI * G4UniformRand();
} while (G4UniformRand() > std::pow(std::sin(fGenieRotAngle / 2.0), 2));
```

`source_repro.py prepare` creates the minimal accepted rooTracker input and `genie_isotropic.mac`.
The complete regression is `test_source_physics.py::test_genie_isotropic_directions_are_uniform_on_sphere`.

## Reproduction and local fix validation

Get or share an authorized CPU allocation before running any command:

```bash
srun --partition=milano --account=neutrino:ml-dev@milano --qos=preemptable \
  --ntasks=1 --cpus-per-task=4 --mem=12G --time=00:30:00 --pty bash
cd /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics
source geant4_env.sh
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
uv_audit=/sdf/home/y/youngsam/.local/bin/uv
py_audit=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$uv_audit" run --offline --no-project --python "$py_audit" python source_repro.py prepare
for name in gamma_prompt gamma_delayed pion_deflection genie_isotropic; do
  ./run_photonsim.sh "$PWD/PhotonSim_audit" "$PWD/$name.mac" > "$name.log" 2>&1
done
"$uv_audit" run --offline --no-project --python "$py_audit" python -m pytest -q test_source_physics.py
PHOTONSIM_TEST_SUFFIX=_fixed "$uv_audit" run --offline --no-project --python "$py_audit" python -m pytest -q test_source_physics.py
```

`baseline-tests-39894010.log` shows all three physical-oracle tests failing against original outputs.
`fixed-tests-39894010.log` shows all three tests passing against outputs from the isolated changed source.
`prepare_fixes.py` constructs the audit-only source changes and macro variants, and `build_fixed.sh` records its compiler command.
`build_and_run.sbatch` records the original compiler command, and should use `run_photonsim.sh` around each executable invocation in the current CVMFS environment.
No production source was changed.

## Confirmed: missing thermal-neutron elastic physics distorts capture clock

Affected constructor: [PhysicsList.cc:62](https://github.com/cesarjesusvalls/PhotonSim/blob/ff73224a669fdc3598dc2993a8763223b11a3e5b/src/PhysicsList.cc#L62).
The standard hadron elastic constructor does not model room-temperature water's thermal neutron scattering.
This issue is distinct from the already demonstrated ten microsecond cutoff and becomes observable when that cutoff is removed for a delayed-neutron channel.
The independent pure-water capture-time oracle is approximately 204 microseconds, from hydrogen number density times the 2200 m/s capture cross section, consistent with [SK's measured pure-water capture lifetime](https://arxiv.org/abs/1311.3738).
Native Geant4 inspection confirmed `G4_WATER` is liquid, has density 1 g/cm3, temperature 293.15 K, and hydrogen number density `6.6856e28 /m3`.
The explicit rate calculation uses `tau = 1 / (n_H * 0.3326 barn * 2200 m/s) = 204.416 microseconds`.
One thousand initially thermal neutrons at 0.025 eV, using the original elastic physics with only the time cutoff removed, yielded 997 tagged captures with mean 289.940 microseconds, standard error 15.855 microseconds, and standard deviation 500.629 microseconds.
The median was 129.555 microseconds, demonstrating a long tail rather than a uniformly shifted clock.
Of these captures, 5.918% occurred after 1 ms, versus approximately 0.750% expected from a 204.416 microsecond exponential.
One other neutron beta-decayed after approximately 1159.86 seconds instead of being captured in water.
The remaining two events lacked an `nCapture` gamma tag but had neutron-induced reaction products, so the tagged-capture count should not be interpreted as total nuclear disappearance probability.
A diagnostic that inactivated only neutron elastic scattering returned an approximately exponential capture clock, mean 215.679 +/- 6.898 microseconds and standard deviation 218.132 microseconds.
Deleting elastic scattering is a diagnostic control, not a physical fix.

The executed minimal fix changes only `G4HadronElasticPhysics` to `G4HadronElasticPhysicsHPT` and retains the original `G4HadronPhysicsQGSP_BERT` capture/inelastic constructor.
The corresponding 1000-event sample had 1000 captures, mean 202.711 +/- 6.350 microseconds, median 137.360 microseconds, and standard deviation 200.816 microseconds, consistent with the independent clock and shape.
Geant4 11.3's thermal scattering names explicitly map `(G4_WATER, H)` to `h_water`, so the existing material selects the bound-water thermal dataset without a material rename.
The constructor installs the thermal scattering law below 4 eV, as implemented in [G4HadronElasticPhysicsHPT.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/hadron_elastic/src/G4HadronElasticPhysicsHPT.cc) and [G4ParticleHPThermalScatteringNames.cc](https://github.com/Geant4/geant4/blob/v11.3.0/source/processes/hadronic/models/particle_hp/src/G4ParticleHPThermalScatteringNames.cc).
The generic capture cross-section code clamps neutron energy below 1e-5 eV, which explains why unphysical repeated cooling can produce a slow capture tail when room-temperature scattering is absent.
This is supported by the executed no-elastic control and the restored thermal-elastic sample, rather than relying on that source observation alone.

Minimal fix:

```cpp
#include "G4HadronElasticPhysicsHPT.hh"
// In PhysicsList's registration list:
RegisterPhysics(new G4HadronElasticPhysicsHPT(0));
```

`test_neutron_capture.py` reads the actual ROOT products and checks the independent rate and the probability of a capture later than 1 ms.
The original uncut sample fails and the targeted thermal-elastic sample passes, with logs `thermal-tests-original-39895962.log` and `thermal-tests-fixed-39895962.log`.
The thermal simulations ran in milano allocation `39894010`, and the final regression and native material inspection ran after preemption in milano allocation `39895962`.
Evidence includes `thermal_results.json`, `thermal_thermalelastic_results.json`, `thermal_tail_summary.json` and `material-properties-39895962.log`.
The macro is `thermal_neutron_fixed.mac`; the changed constructor is isolated in `PhotonSim_thermalelastic/src/PhysicsList.cc`, with its build recipe in `build_thermalelastic.sh`.

Broad replacement with HP capture is not the tested fix.
The additional HP-capture variants produced incorrect clocks in this environment: HP elastic plus HP capture gave 383.876 +/- 12.257 microseconds, and HPT elastic plus HP capture gave 353.464 +/- 10.830 microseconds.
Those control results are preserved in `thermal_HP_results.json` and `thermal_HPT_results.json` and warrant separate investigation before selecting an expanded neutron physics list.
The successful narrow fix validates the pure-water thermal capture clock and its tail in this configuration, without claiming complete neutron physics agreement across all energies, isotopes or gadolinium mixtures.

## Scope

The 1000 m bulk-water source cube is a deliberate source abstraction, and is not itself classified here as a new bug.
The finite-detector treatment of near-boundary cascades, particles leaving water, and outside-born photons remains a separate assessment involving LUCiD's injection and optical tracing.
The water refractive-index defect and measured-material original-versus-fixed comparisons are owned by the `photon_sources` lane.
These proofs establish specific physical defects and local fixes, rather than establishing that every selected electromagnetic, hadronic or nuclear model exactly reproduces a real detector.

