"""Render the collaborator report from saved evidence, without simulations."""
from pathlib import Path
import ast
import html
import json
import fitz
import matplotlib

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
CSS = '''
@font-face {font-family:AuditSans; src:url(DejaVuSans.ttf);}
@font-face {font-family:AuditSans; font-weight:bold; src:url(DejaVuSans-Bold.ttf);}
@font-face {font-family:AuditMono; src:url(DejaVuSansMono.ttf);}
body {font-family:AuditSans; font-size:10pt; line-height:1.32; color:#172535;}
h1 {font-size:22pt; line-height:1.13; color:#16354b; margin:0 0 12pt;}
h2 {font-size:14pt; line-height:1.2; color:#16354b; margin:14pt 0 7pt;}
h3 {font-size:10.5pt; color:#16354b; margin:11pt 0 5pt;}
p {margin:0 0 7pt;}
.kicker {font-size:8.3pt; color:#39728b; margin:0 0 8pt;}
.lead {font-size:11pt; line-height:1.35;}
.note, .caption {font-size:8.4pt; line-height:1.3; color:#445768;}
.caption {margin:5pt 0 10pt;}
.callout {background:#edf4f7; border-left:3pt solid #39728b; padding:9pt 11pt; margin:10pt 0;}
pre {font-family:AuditMono; font-size:7.7pt; line-height:1.2; background:#f2f4f6; padding:8pt; margin:7pt 0 9pt; white-space:pre-wrap;}
code {font-family:AuditMono; font-size:8.4pt;}
table {width:100%; border-collapse:collapse; margin:8pt 0 10pt;}
th {background:#e6eef2; color:#16354b; font-size:8.2pt; text-align:left; padding:6pt; vertical-align:top;}
td {font-size:8.2pt; padding:6pt; border-bottom:0.5pt solid #d7e0e6; vertical-align:top;}
.compact td {padding:5pt;}
img {width:100%;}
a {color:#24627e; text-decoration:none;}
ul {margin:3pt 0 9pt; padding-left:17pt;}
li {margin:0 0 4pt;}
'''


def code(text):
    return '<pre>' + html.escape(text.strip('\n')) + '</pre>'


def fig(key, caption):
    return ""


def section(kicker, title, body):
    return f'<p class="kicker">{kicker}</p><h1>{title}</h1>' + body


def reference(path):
    return f'<p class="note">Evidence: <code>{html.escape(path)}</code>.</p>'


def main():
    sections = []
    sections.append(('Executive summary', section('SK_WAND / DATA MODE / 04 OCTOBER 2026', 'LUCiD physics audit', '''
<p class="lead">Eight substantial defects were reproduced across source generation, optical transport, and readout.
The handoff contains physical-contract tests and candidate corrections for collaborators to apply in the main repositories.</p>
<p>Reference: LUCiD <code>82f8d24</code>, <code>SK_WAND_geom_config.json</code> and <code>SK_WAND_physics_config.json</code>.
Upstream source: PhotonSim <code>ff73224</code>, Geant4 11.3.0.
Production code was not changed.</p>
<table class="compact"><tr><th>Finding</th><th>Measured consequence</th><th>Correction target</th></tr>
<tr><td>F1. QE / reflection</td><td>25.2% fewer direct PE</td><td>Conditional sensor acceptance</td></tr>
<tr><td>F2. First PMT</td><td>Wrong PMT; +77.62 ns in one ray</td><td>Complete segment search</td></tr>
<tr><td>F3. Exterior light</td><td>Real muon: +97 false PE, 37 PMTs</td><td>Transformed-origin bounds</td></tr>
<tr><td>F4. Late source tracks</td><td>Delayed gamma: 93.24% fewer photons</td><td>Remove 10 µs source kill</td></tr>
<tr><td>F5. Late readout</td><td>Two triggerable bursts become one</td><td>Explicit DAQ acceptance</td></tr>
<tr><td>F6. Thermal neutrons</td><td>&gt;1 ms capture tail: 5.92% vs 0.75%</td><td>Thermal elastic physics</td></tr>
<tr><td>F7. Pion continuity</td><td>Replacement jumps up to 3.35 m</td><td>Post-step position and time</td></tr>
<tr><td>F8. GENIE isotropy</td><td>Forward fraction 70.4% vs 50%</td><td>Uniform SO(3) rotation</td></tr>
</table>
<div class="callout"><b>Fix order.</b> Correct optical response and geometry first for prompt samples.
For delayed/neutron channels, address the source cutoff, thermal model and DAQ acceptance together.
Repair pion and GENIE generation before regenerating affected source samples.</div>
<p>The QE measurement concerns direct light, not every event's total charge.
Geometry percentages below are conditional on independently established first hits.
The two time cuts are separate stages; their losses cannot be added.</p>
<p><b>Additional findings:</b> charge-dependent timing, wavelength-QE clipping, supernova overlap, EventID joins, CLI configuration, external vertex truth, negative TTS, and source water-index calibration.</p>
<p class="note">All numerical tests used milano CPUs; zero GPUs.
The new regression suite exercises production functions directly.
Candidate patches are reviewed examples, not merged production changes.
Finite validation does not certify exact real-detector agreement.</p>
''')))
    sections.append(('F1 and F3: detector response', section('OPTICAL TRANSPORT', 'Sensor efficiency and birth bounds', '''
<h2>F1. Incident QE is treated as conditional QE</h2>
<p>Reflection is selected first, then incident QE is sampled on the nonreflected branch.
That gives detection probability <code>(1-R) × QE</code>.
Incident QE already includes reflection loss; the correct mutually exclusive outcomes are detection QE, reflection R, and terminal loss 1-QE-R.</p>
<p>The boundary test gives probability 0.16977 instead of incident QE 0.22619.
A selected 50,000-photon DATA beam gives 10,997.99 direct PE at R=0 and 8,225.99 at R=0.25.</p>
''' + fig('qe_reflection', 'Figure 1. Boundary acceptance and a real SK_WAND DATA beam at 400 nm. Boundary bars show binomial standard errors; the beam panel measures prompt direct PE, not total event loss.') + '''
<p><b>Target:</b> <code>photon_step.py</code>, <code>sensor_response.py</code>, and the DATA QE normalization in <code>simulator.py</code>.
Use the actual encounter reflectance and keep expected-value mode consistent.</p>
''' + code('''# Scalar boundary contract; validate parameters before JIT tracing.
if not (0 <= R <= 1 and 0 <= qe_incident <= 1 - R):
    raise ValueError("Incompatible incident QE and reflectance")
qe_conditional = qe_incident / (1 - R) if R < 1 else 0.0
# Bernoulli acceptance on the nonreflected branch uses qe_conditional.''') + '''
<p class="note">This is the probability contract, not a complete drop-in angular-surface patch.
Do not multiply detected PE weights by 4/3.
The absolute provenance of the supplied QE table remains unverified; the finding assumes incident-photon QE semantics.</p>
<h2>F3. Exterior photons enter the ID response</h2>
<p>The initial DATA mask handles padding but ignores transformed-origin bounds.
A 100,000-photon exterior beam creates 22,214 false PE.
A real translated 1 GeV muon gains 97 PE (+1.97%) and 37 lit PMTs; this is one affected event.</p>
''' + code('''# simulator.py, after rotation and translation of source origins:
mask = (jnp.arange(n_rays) < photon_data['N'])
mask &= get_inside_detector_flag(final_origins)''') + '''
<p>The isolated correction gives zero exterior response and leaves interior deposits identical.
This contract assumes a closed opaque ID; an exterior-light model requires explicit external boundaries.</p>
''' + reference('pmt_detection/findings.md; transport/findings.md; photon_sources/findings.md'))))

    sections.append(('F2: first-PMT selection', section('GEOMETRY', 'Search the whole photon segment', '''
<p>Candidate PMTs are chosen near the eventual cylinder endpoint.
Tangential rays can intersect a different sphere much earlier.
The valid example below starts inside water, outside every PMT sphere, and exits the cylinder after 18.5892 m.</p>
''' + fig('first_pmt_lookup', 'Figure 2. Lost and incorrectly selected first PMTs among oracle-confirmed hits. Each probe uses 20,000 isotropic rays and a float64 all-PMT oracle bounded by the first water exit. These are ray fractions, not event charge losses.') + '''
<p><b>Minimal failure:</b> expected PMT 4105 at 0.166513 m, returned PMT 458 at 17.662575 m.
The public DATA path adds 77.62 ns to the arrival time.</p>
''' + code('''g = DetectorGeometry.from_config(
    "config/SK_WAND_geom_config.json",
    temperature=0., deposit_leg_bound=True)
r = g.propagator(jnp.array([[16.8585866627, -.3529484099, -.4]]),
                 jnp.array([[0., 0., 1.]]))
assert 4105 in r['sensor_indices'][:, 0]''') + '''
<h2>Correction contract</h2>
<p><b>Target:</b> candidate selection in <code>lucid/propagation/shared.py</code>.
Find the smallest positive sphere entry along the complete travelled segment, before its water exit.
Use chunked all-sensor intersection as a correctness baseline; accelerate with grid traversal or a BVH that preserves candidate completeness.</p>
''' + code('''# Reference sphere entries for a normalized ray, not a drop-in grid patch.
v = centers - origin
b = v @ direction
disc = b*b - (v*v).sum(axis=1) + sensor_radius**2
entry = b - np.sqrt(np.maximum(disc, 0.))
valid = (disc >= 0.) & (entry > 0.) & (entry <= water_exit)
t = np.where(valid, entry, np.inf)
first = int(np.argmin(t)) if np.isfinite(t).any() else -1''') + '''
<p>The saved complete-selection prototype restores all 22 problematic rays.
Wrong or lost assignments among independently established hits total 1.07% over the standard production volume, 7.22% at an allowed top vertex, and 29.09% at 10 cm inside the barrel.
These are isotropic ray probes, not complete-event charge biases.
Increasing endpoint-neighbor count alone cannot guarantee the first hit.
The accepted spherical-PMT approximation does not explain missing intersections with those same spheres.</p>
''' + reference('geometry/findings.md; geometry/reproduce_selection_fix.py; end_to_end/tangent_data_reproducer.py'))))
    sections.append(('F4 and F5: delayed light', section('SOURCE AND READOUT', 'Two independent late-light cuts', '''
<p><b>F4. PhotonSim</b> kills newly created non-optical tracks whose global creation time exceeds 10 µs, after their first step.
Some first-step light remains, so the result is corrupted yield rather than a clean acquisition boundary.</p>
''' + fig('source_light', 'Figure 3. Left: the identical gamma sample shifted to 200 µs loses 93.24% of source photons; removing the cut restores the prompt counts exactly. Right: water-index calibration changes primary muon light near threshold. The 84.27% result at 54 MeV excludes secondary light and does not describe GeV campaigns. Error bars are event-level standard errors.') + '''
<p><b>Target:</b> PhotonSim <code>src/SteppingAction.cc:102-105</code>.</p>
''' + code('''// Delete this unconditional source-time veto:
if (time > 10000.0 * ns) {
    track->SetTrackStatus(fStopAndKill);
    return;
}''') + '''
<p class="note">The fix is deletion of the displayed block.
Apply an intentional acquisition window after physical photon generation.
The matched-seed 30-event gamma control changes 454 delayed photons to 6,716, exactly matching prompt output.</p>
<h2>F5. Detected deposits are deleted after 100 µs</h2>
<p><b>Target:</b> <code>lucid/simulation/digitizer.py:109,518</code>.
The digitizer removes late deposits before evaluating the configured trigger.
Two 80-PE triggerable bursts become one; the production-response control restores 58 delayed physics PE and a second gate when uncapped.</p>
''' + code('''# Required contract, not the complete DAQ implementation:
# 1. Retain finite signal deposits in explicit acquisition intervals.
# 2. Generate dark once over bounded interval unions.
# 3. Integrate, discriminate, and trigger those deposits.
# Diagnostic only: _MAX_DIGIT_TIME_NS = np.inf''') + '''
<p>Shipping the infinity change alone can generate excessive dark over radioactive tails.
A complete fix needs explicit bounded DAQ windows.
The SK-I preset's absence of SK-IV forced after-trigger readout is a separate detector-era limitation.</p>
''' + reference('geant4_physics/findings.md; electronics/findings.md; photon_sources/findings.md'))))

    sections.append(('F6: thermal neutron physics', section('GEANT4 PHYSICS', 'Restore the water capture clock', '''
<p>The default elastic model does not thermalize neutrons correctly in room-temperature water.
Repeated stationary-target scattering overcools neutrons; capture cross-section clamping can then suppress the rate at low velocity.
The source time cutoff is removed in both diagnostic samples so it cannot hide this effect.</p>
''' + fig('neutron_capture', 'Figure 4. Capture-time density and survival from actual ROOT output, normalized to tagged captures: 997 original and 1,000 corrected, out of 1,000 events each. The independent pure-water lifetime is 204.416 µs. Three original events lack a capture-gamma tag; the original longest capture, 6,814 µs, lies beyond the plot.') + '''
<p>The original mean is <b>289.94 ± 15.86 µs</b> and the tagged-capture tail beyond 1 ms is <b>5.918%</b>.
The narrow correction gives <b>202.71 ± 6.35 µs</b> and <b>0.700%</b>, versus 0.750% expected.
Uncertainties are standard errors.</p>
<h2>Minimal tested correction</h2>
''' + code('''// PhotonSim src/PhysicsList.cc:
#include "G4HadronElasticPhysicsHPT.hh"
// Replace G4HadronElasticPhysics; retain the original capture constructor.
RegisterPhysics(new G4HadronElasticPhysicsHPT(0));''') + '''
<p>Geant4 11.3 maps the existing <code>(G4_WATER, H)</code> material to bound-water thermal scattering.
The inspected material is liquid water at 293.15 K, density 1 g/cm³.
Full HP-capture replacements gave biased means of 383.88 and 353.46 µs in controls and are not the tested recommendation.</p>
''' + code('''# Minimal assertions after reading freshly regenerated capture times:
n_h = 2 * 6.02214076e23 * 1e6 / 18.01528
tau_us = 1e6 / (n_h * .3326e-28 * 2200)
assert abs(t.mean() - tau_us) < 4*t.std(ddof=1)/np.sqrt(len(t))
assert np.mean(t > 1000) < .02''') + '''
<p class="note">The original output fails; the corrected output passes.
This validates the thermal pure-water clock and tail, not all neutron energies, displacement, isotope fractions, or Gd mixtures.</p>
''' + reference('geant4_physics/test_neutron_capture.py; geant4_physics/thermal_tail_summary.json; astra_physics/findings.md'))))
    sections.append(('F7 and F8: source continuity and orientation', section('PHOTONSIM GENERATION', 'Preserve endpoints and isotropy', '''
<h2>F7. Pion truth splitting rewinds the trajectory</h2>
<p>A completed pion step is followed by a replacement at an earlier stored position and time, but with post-step momentum.
Twenty 1 GeV pion events contain 61 replacements, with a maximum endpoint-to-birth gap of 3.352 m.
Splitting labels must preserve physical position and time continuity.</p>
''' + code('''// src/SteppingAction.cc: replace the stored position/time reads.
G4ThreeVector kinkPosition = track->GetPosition();
G4double kinkTime = track->GetGlobalTime();''') + '''
<p>The isolated fixed sample has zero gap for all 59 replacements.
The changed histories need not produce the same replacement count.
Use <code>test_pion_replacement_preserves_endpoint</code> on freshly generated output.</p>
<h2>F8. Uniform axes and angles do not give isotropy</h2>
''' + fig('genie_isotropy', 'Figure 5. Actual GENIE neutrino directions before and after the coherent uniform-SO(3) correction; 20,000 events per sample. The uniform density is an independent expectation, not a fitted distribution.') + '''
<p><b>Target:</b> isotropic GENIE branch of <code>src/PrimaryGeneratorAction.cc</code>.
Retain the existing uniform axis and draw the angle with Haar density proportional to sin²(angle/2):</p>
''' + code('''do {
  fGenieRotAngle = 2.0 * M_PI * G4UniformRand();
} while (G4UniformRand() >
         std::pow(std::sin(fGenieRotAngle / 2.0), 2));''') + '''
<p>Apply the same accepted rotation to every momentum in the event.
The original forward fraction is 0.70395 and mean cos(θ) is 0.33213.
The fixed values are 0.4971 and -0.003401.
Apply this only when isotropy is requested; preserve deliberate beam or supernova anisotropy.</p>
''' + reference('geant4_physics/findings.md; typeset/patches_photonsim/02_pion_endpoint.patch; typeset/patches_photonsim/03_genie_haar_rotation.patch'))))

    sections.append(('Secondary readout and optical findings', section('SMALL LOCAL CORRECTIONS', 'Timing, spectrum and timestamp validity', '''
<h2>S1. Timing uses true count instead of sampled charge</h2>
<p><b>Target:</b> <code>digitizer.py::apply_readout_resolution</code>.
The selected model samples analogue charge, but computes jitter from true PE count.
In single-PE samples, the low-charge timing RMS is 3.483 ns instead of 4.802 ns, while the high-charge RMS is 3.503 ns instead of 2.421 ns.</p>
''' + code('''q_time = (np.maximum(pe_reco, 0.5)
          if model.get("time_model") in ("sk_gauss", "hk_emg")
          else pe_true)
t = _sample_time_jitter(digit_time, q_time, model, rng)''') + '''
<p>The correction restores the selected charge-conditioned timing curve.
It does not calibrate the complete measured SK timing distribution.</p>
<h2>S2. Water interpolation clipping changes PMT QE</h2>
<p><b>Target:</b> <code>optical_model.py::evaluate_optical_model</code>.
Medium-grid clamping changes the wavelength used for QE lookup.
Although configured QE is zero, 30,000 photons yield 128 PE at 275 nm and 49 PE at 674 nm.
The tested broadband charge bias is +0.38% to +0.68%.</p>
''' + code('''qe = qe_fn(jnp.asarray(wavelengths)) * _deviation(
    wl, control_lambda, rp.qe_dev)''') + '''
<p>Clamp wavelengths for water interpolation independently of the QE lookup.</p>
<h2>S3. Negative TTS timestamps erase all sensor charge</h2>
<p><b>Target:</b> <code>simulator.py::make_hits_data</code>.
An earliest TTS-smeared time below zero is treated as invalid, deleting the sensor's whole charge.
At 10 PE per sensor, the reproduced realistic-mode fixture loses 99.88%.
Shifting all source times by 100 ns restores that charge.</p>
''' + code('''# Use a finite-time validity condition after smearing.
valid_time = jnp.isfinite(detector_mins)
# Remove the additional detector_mins > 0 requirement.''') + '''
<p>This is dormant in the selected production path, which uses <code>per_segment</code> and <code>tts=0</code>.
It remains a severe failure for the affected DATA option.</p>
<h2>S4. Source phase-index calibration</h2>
<p>The water RINDEX table differs from measured water.
Changing only that table gives 149 versus 947 primary Cherenkov photons in 100 muon events at 54 MeV in the benchmark.
Choose and document the temperature/calibration curve before replacing <code>ConstructWater()</code> knots and regenerating sources.
This is a physical calibration defect near threshold, distinct from the declared constant photon-speed approximation.</p>
''' + reference('electronics/findings.md; water_optics/findings.md; pmt_detection/findings.md; photon_sources/findings.md'))))
    sections.append(('Conditional production failures', section('INPUT AND PILE-UP', 'Preserve identity, readout and truth', '''
<h2>S5. Supernova chunks ignore delayed-arrival overlap</h2>
<p><b>Target:</b> <code>event_generation.py::_group_interactions_by_gap</code> and the supernova merge loop.
Interactions at 0 and 1,000 ns with photon delays 1,100 and 100 ns arrive simultaneously.
Separate digitization writes two 1-PE digits instead of one 2-PE digit.
Overlapping dark spans give 93.98 PE rather than 46.48 PE, a 2.02× ratio in the fixture.</p>
<p>Merge actual arrival/readout spans before digitization and dark generation, including integration, deadtime and padding.
Generate dark once over the interval union.
Merging window labels after digitization cannot repair this defect.
The full shipped burst's corruption rate was not measured.</p>
<h2>S6. ROOT entry number is used as EventID</h2>
<p><b>Target:</b> <code>root_reader.py</code>, both public and production paths.
Entry zero storing EventID seven returns zero of nine photons.
A direct stored-ID join returns all nine.</p>
''' + code('''# Include EventID among the main-tree branches in both paths.
event_id = int(tree_data["EventID"][0])
photon_positions, photon_directions, photon_times, photon_wavelengths = \
    _read_photons_for_event(raw_tree, event_id)''') + '''
<p>Repeated IDs from multiple runs require a unique run/event key or explicit rejection.
Ordinary serial source production, where ID equals entry number, avoids this case.</p>
<h2>S7. Direct conversion CLI ignores selected readout</h2>
<p><b>Target:</b> <code>generate_events_with_particles.py:137-151</code>.
Explicit SK_WAND selection becomes basic digitization with no trigger.
A real zero-light electron event is retained; the ordinary <code>run_job</code> path correctly rejects it.</p>
''' + code('''# Reuse run_job's config readers and forward these generator kwargs:
digitizer=_read_digitizer_cfg({}, args.physics_config),
trigger=_read_trigger_cfg({}, args.physics_config),''') + '''
<p>Normal <code>run_job</code> forwards these blocks correctly.</p>
<h2>S8. Nonzero source vertex is overwritten in truth</h2>
<p><b>Target:</b> interaction metadata in both particle and pooled-stream generation.
A real shower rigidly translated to (5,-2,3) m writes vertex zero: a 6.164 m error.
Its photon and step coordinates are correct.</p>
''' + code('''# Coordinate contract; adapt the existing R/displacement variables.
vertex_detector = R @ vertex_input + displacement
# If sampling an absolute target vertex instead:
displacement = vertex_target - R @ vertex_input
vertex_detector = vertex_target''') + '''
<p>Require common primary vertices for one interaction and use the same transform for light, steps and metadata.
Current origin-zero production avoids the reproduced truth error.</p>
''' + reference('production_io/findings.md; g4_inputs/findings.md'))))

    sections.append(('Regression suite and patch handoff', section('COLLABORATOR WORKFLOW', 'Fail before, pass after', '''
<p><b>Runtime suite:</b> <code>regressions/test_runtime.py</code> calls production functions directly.
All seven tests failed on the reference checkout and passed with isolated corrections using unchanged assertions.
Fixtures are self-contained; no measurement-witness helpers are imported.</p>
<table class="compact"><tr><th>Physical contract</th><th>Current code</th><th>Isolated correction</th></tr>
<tr><td>Incident-QE acceptance</td><td>Fail</td><td>Pass</td></tr>
<tr><td>First PMT / path distance</td><td>Fail</td><td>Pass</td></tr>
<tr><td>Exterior-birth zero ID charge</td><td>Fail</td><td>Pass</td></tr>
<tr><td>Delayed deposits survive</td><td>Fail</td><td>Pass, diagnostic uncapping</td></tr>
<tr><td>Zero QE at spectral endpoints</td><td>Fail</td><td>Pass</td></tr>
<tr><td>Sampled-charge timing width</td><td>Fail</td><td>Pass</td></tr>
<tr><td>TTS time-translation invariance</td><td>Fail</td><td>Pass</td></tr>
</table>
<p><b>Source suite:</b> <code>regressions/test_photonsim_outputs.py</code> checks time-shifted gamma yield, pion continuity, GENIE isotropy, and thermal capture mean/tail.
All four fail on original products and pass on corrected products.
These read ROOT output: rebuild PhotonSim and regenerate the fixture files after a source fix.</p>
''' + code('''# Inside a CPU allocation, from the candidate LUCiD checkout:
export JAX_PLATFORMS=cpu
uv run pytest -q /path/to/handoff/regressions/test_runtime.py
# Apply/review one issue's candidate patch, then rerun the same test.
# The assertions are not edited to obtain a pass.''') + '''
<h2>What to apply</h2>
<p><code>patches_runtime/</code> contains per-issue LUCiD candidate diffs.
The complete first-PMT patch is a correctness reference, requiring an efficient production search.
The late-time infinity patch is a diagnostic, requiring bounded acquisition/dark windows before shipping.</p>
<p><code>patches_photonsim/</code> contains four separate diffs against PhotonSim <code>ff73224</code>.
Pion, late-source, GENIE and thermal-elastic changes belong upstream, then the source pin and generated data must be updated in LUCiD.</p>
<p>Read <code>HANDOFF.md</code> and the regression README for exact commands, dependencies, fixture regeneration and patch status.
The PDF and source reports distinguish tested local corrections from integration sketches and calibration work.</p>
<p class="note">Runtime: 7 failures in 32.03 s, then 7 passes in 31.56 s.
Source: four original-output failures and four corrected-output passes.
This validation used milano job 39896290 and changed no tracked production file.</p>
''')))
    sections.append(('Coverage and physical limits', section('VALIDATION AND REFERENCES', 'What the audit establishes', '''
<p>The audit followed source generation through ROOT conversion, source transforms, geometry, scattering/absorption, sensor response, digitization, triggering and HDF5 output.
Two real 100 MeV electron events were processed through the selected production path into four output files.</p>
<table class="compact"><tr><th>Stage</th><th>Independent checks</th></tr>
<tr><td>ROOT / genealogy</td><td>Genuine vector branches, mm-to-m conversion, shuffled chunks, daughters, electron/pi0 ancestry, empty input</td></tr>
<tr><td>Source handoff</td><td>Padded chunk conservation, times through 200 µs, coherent isotropic Python rotations</td></tr>
<tr><td>Geometry</td><td>Water-bounded float64 all-sphere oracle; central and translated-source timing</td></tr>
<tr><td>Transport</td><td>500,000 hazard trials, scattering mixture, normalized directions, reflection hemisphere</td></tr>
<tr><td>Sensor / readout</td><td>QE-once control, incident acceptance, sampled-charge timing, late bursts, discriminator and dark</td></tr>
<tr><td>Storage</td><td>Digit/trigger references, window inclusion, truth-charge joins, float64 long timestamps</td></tr>
</table>
<p>The previously repaired QE-squared loss is independently excluded.
The corrected input unit boundary passes displaced-source controls.
An earlier central-loss estimate used intersections beyond the water exit and was rejected.
Passing integration checks does not erase the physical defects in this report.</p>
<h2>Accepted approximations and remaining calibration</h2>
<p>Spherical PMTs are not classified as bugs.
Other explicit simplifications include scalar reflection, finite iterations, unpolarized Rayleigh scattering, constant photon speed and fitted electronics response.</p>
<p>Against a dispersive group-velocity reference, spectrum-weighted photons arrive about 3.14 ns early over 17 m, without about 0.91 ns chromatic spread.
PhotonSim polarization is discarded.
The fitted SPE spectrum has a 27% narrower RMS and a much smaller above-4-PE tail than its referenced table.
These are fidelity limits requiring calibration or a documented approximation budget.</p>
<p>Absolute QE provenance, angular/immersion and per-PMT calibration, red-water absorption provenance, water variation, and modern neutron-tagging DAQ remain unestablished.
The finite tests support specific corrections; they cannot prove 100% equivalence to a real detector.</p>
<h2>Primary references</h2>
<p class="note">[1] <a href="https://arxiv.org/pdf/physics/0408075">Motta and Schönert: photocathode QE, absorption and reflection, Eq. 1.</a><br>
[2] <a href="https://github.com/WCSim/WCSim/blob/develop/src/WCSimWCSD.cc">WCSim conditional QE correction.</a><br>
[3] <a href="https://arxiv.org/abs/1307.0162">Super-Kamiokande detector calibration, optics and charge-dependent timing.</a><br>
[4] <a href="https://arxiv.org/abs/1311.3738">Super-Kamiokande neutron capture lifetime and delayed readout.</a><br>
[5] <a href="https://github.com/Geant4/geant4/blob/v11.3.0/source/physics_lists/constructors/hadron_elastic/src/G4HadronElasticPhysicsHPT.cc">Geant4 11.3 thermal elastic constructor.</a><br>
[6] <a href="https://github.com/cesarjesusvalls/PhotonSim/tree/ff73224a669fdc3598dc2993a8763223b11a3e5b">Pinned PhotonSim source.</a><br>
[7] <a href="https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/H2O/nk/Daimon-20.0C.yml">Daimon 20 °C water phase-index coefficients.</a></p>
<p class="note">Detailed source provenance, experiment seeds, allocation IDs and logs are retained in the component reports.
The editable report is generated locally from these saved artifacts.</p>
''')))

    excerpts = json.loads((HERE / 'regressions/code_excerpts.json').read_text())
    tests_by_section = {
        1: ['test_incident_qe_is_preserved_after_reflection',
            'test_translated_photons_outside_id_make_zero_charge'],
        3: ['test_delayed_gamma_yield_is_time_translation_invariant',
            'test_two_readout_bursts_survive_100us'],
    }
    for index, names in tests_by_section.items():
        title, body = sections[index]
        body += '<h2>Minimal physical regressions</h2>'
        body += '<p class="note">Use the unchanged fixtures and imports in the bundled regression suite.</p>'
        for name in names:
            body += code(excerpts[name]['code'])
        sections[index] = title, body

    archive = fitz.Archive(str(HERE))
    archive.add(matplotlib.get_data_path() + '/fonts/ttf')
    paper = fitz.paper_rect('a4')
    body_rect = fitz.Rect(44, 62, paper.width-44, paper.height-47)
    out = fitz.open()
    toc, layout = [], []
    html_pages = []
    for title, content in sections:
        story = fitz.Story(html=content, user_css=CSS, archive=archive)
        def next_rect(number, filled):
            if number >= 3:
                raise RuntimeError(f'Layout did not fit: {title}; last rect: {filled}')
            return paper, body_rect, None
        print(f'Rendering: {title}', flush=True)
        doc = story.write_with_links(next_rect)
        first = len(out) + 1
        toc.append([1, title, first])
        layout.append({'title': title, 'first_page': first, 'page_count': len(doc)})
        out.insert_pdf(doc)
        html_pages.append('<section class="report-page">' + content + '</section>')
    for i, page in enumerate(out):
        page.draw_line((44, 42), (paper.width-44, 42), color=(.74, .82, .86), width=.6)
        page.insert_text((44, 33), 'LUCiD / SK_WAND DATA PHYSICS AUDIT', fontsize=8,
                         fontname='hebo', color=(.22, .36, .43))
        page.insert_text((44, paper.height-26), '04 Oct 2026  |  LUCiD 82f8d24 / PhotonSim ff73224',
                         fontsize=7.3, color=(.35, .42, .48))
        page.insert_text((paper.width-70, paper.height-26), f'{i+1} / {len(out)}',
                         fontsize=7.3, color=(.35, .42, .48))
    out.set_toc(toc)
    out.set_metadata({'title': 'LUCiD SK_WAND DATA physics audit',
                      'subject': 'Reproduced physical defects, minimal regression tests and candidate fixes',
                      'author': 'LUCiD audit', 'keywords': 'LUCiD, PhotonSim, water Cherenkov, SK_WAND, physics audit'})
    destination = HERE / 'LUCiD_SK_WAND_Physics_Audit.pdf'
    out.save(destination, garbage=4, deflate=True)
    html_css = CSS.replace('url(DejaVu', 'url(fonts/DejaVu')
    print_css = '@page {size:A4; margin:18mm 16mm;} .report-page {break-before:page;} .report-page:first-child {break-before:auto;} body{max-width:180mm;margin:auto;}'
    (HERE / 'LUCiD_SK_WAND_Physics_Audit.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>LUCiD physics audit</title><style>'
        + html_css + print_css + '</style></head><body>' + '\n'.join(html_pages) + '</body></html>')
    (HERE / 'report_layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    print(json.dumps({'pdf': str(destination), 'pages': len(out), 'sections': layout}, indent=2))


if __name__ == '__main__':
    main()
