"""Render audit measurements; run only inside an authorized CPU allocation."""
from pathlib import Path
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot

assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'typeset' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#64748B'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'grid.alpha': 0.22,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 220,
})
MANIFEST = {'slurm_job_id': os.environ['SLURM_JOB_ID'],
            'partition': os.environ['SLURM_JOB_PARTITION'],
            'method': 'Existing measurements only; no new simulation.', 'figures': []}

def read(rel):
    return json.loads((BASE / rel).read_text())

def save(fig, name, caption, sources, values):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight')
    MANIFEST['figures'].append({'name': name, 'pdf': 'figures/' + name + '.pdf',
        'png': 'figures/' + name + '.png', 'caption': caption,
        'sources': sources, 'numerical_values': values})
    plt.close(fig)

def panel(ax, label, title):
    ax.set_title(label + '  ' + title, loc='left', weight='bold', pad=9)
    ax.grid(axis='y', zorder=0)

def sem(a):
    a = np.asarray(a, float)
    return float(a.std(ddof=1) / np.sqrt(len(a)))

# Incident QE and an independent public DATA-mode beam benchmark.
pmt = read('pmt_detection/results.json')
bq = pmt['boundary_qe']
fix_log = (BASE / 'pmt_detection/fix-39894029.log').read_text().splitlines()
fix = next(json.loads(line) for line in fix_log if line.startswith('{'))
qe = bq['required_incident_probability']
probs = np.array([bq['measured_direct_detection_probability'],
                  fix['observed_incident_probability']])
ns = np.array([bq['n_incident'], 100000])
errs = np.sqrt(probs * (1-probs) / ns)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
a = axs[0]
a.bar([0, 1], 100*probs, color=[ORANGE, BLUE], width=.58, zorder=3)
a.errorbar([0, 1], 100*probs, yerr=100*errs, fmt='none', color='black', capsize=3, lw=.8, zorder=4)
a.axhline(100*qe, color='black', ls='--', lw=1, label='Incident QE = 22.62%')
a.set_xticks([0, 1], ['Current', 'Conditional-QE fix'])
a.set_ylabel('Direct detection probability (%)')
a.set_ylim(0, 27)
a.legend(loc='lower right', frameon=False)
for x, p in enumerate(probs): a.text(x, 100*p+.8, f'{100*p:.2f}%', ha='center', fontsize=8)
panel(a, '(a)', '400 nm sensor-boundary trial')
b = axs[1]
beam = [r for r in pmt['beam'] if r['K'] == 12]
vals = [r['direct_detected_pe'] for r in beam]
b.bar([0, 1], vals, color=[GRAY, ORANGE], width=.58, zorder=3)
b.set_xticks([0, 1], ['R = 0', 'R = 0.25'])
b.set_ylabel('Prompt direct PE / 50,000 photons')
b.set_ylim(0, 13500)
for x, v in enumerate(vals): b.text(x, v+250, f'{v:,.0f}', ha='center', fontsize=8)
loss = 1-vals[1]/vals[0]
b.text(.5, 12600, f'{100*loss:.2f}% direct-light deficit', ha='center', fontsize=8)
panel(b, '(b)', 'SK_WAND DATA beam, 5 m, K = 12')
save(fig, 'qe_reflection',
    'At 400 nm, reflection rejection followed by incident-QE rejection reduces direct detection by about 25%. '
    'The conditional-QE correction restores boundary acceptance within sampling uncertainty. '
    'Boundary error bars are one binomial standard error (300,000 current and 100,000 corrected incident photons). '
    'The independent 50,000-photon DATA beam measures prompt direct PE only; late reflected photons are excluded and total-event loss is not inferred.',
    ['pmt_detection/results.json', 'pmt_detection/fix-39894029.log', 'pmt_detection/verify_conditional_fix.py'],
    {'boundary': {'incident_qe': qe, 'observed': probs.tolist(), 'n': ns.tolist(), 'standard_errors': errs.tolist()},
     'data_beam': beam, 'data_beam_direct_loss_fraction': loss})
# Fractions refer to oracle-confirmed geometrical PMT hits before cylinder exit.
geo = read('geometry/results.json')
prod = read('geometry/production_population_results.json')
labels = ['isotropic_r0.000000', 'isotropic_r8.000000', 'isotropic_r14.962280',
          'isotropic_r16.500000', 'isotropic_r16.862280']
radial = [next(r for r in geo['metrics'] if r['label'] == label) for label in labels]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
a = axs[0]
xs = np.arange(len(radial))
lost = np.array([r['lost_fraction_true_hits'] for r in radial]) * 100
wrong = np.array([r['wrong_fraction_true_hits'] for r in radial]) * 100
a.bar(xs, lost, color=ORANGE, label='Lost hit', zorder=3, width=.65)
a.bar(xs, wrong, bottom=lost, color=BLUE, label='Wrong first PMT', zorder=3, width=.65)
a.set_xticks(xs, ['0', '8', 'R - 2', '16.5', 'R - 0.1'])
a.set_xlabel('Vertex radius at z = 0 (m); R = 16.962 m')
a.set_ylabel('Fraction of true PMT hits (%)')
a.set_ylim(0, 34)
a.legend(frameon=False, loc='upper left')
for x, v in zip(xs, lost+wrong): a.text(x, v+.7, f'{v:.2f}', ha='center', fontsize=8)
panel(a, '(a)', 'Radial scan: isotropic ray directions')
b = axs[1]
rows = prod['metrics']
p_lost = np.array([r['lost_fraction_true_hits'] for r in rows])*100
p_wrong = np.array([r['wrong_fraction_true_hits'] for r in rows])*100
xx = np.arange(2)
b.bar(xx, p_lost, color=ORANGE, width=.58, zorder=3)
b.bar(xx, p_wrong, bottom=p_lost, color=BLUE, width=.58, zorder=3)
b.set_xticks(xx, ['Uniform production\nvertex volume', 'Allowed top vertex\n(0, 0, 0.9H/2)'])
b.set_ylabel('Fraction of true PMT hits (%)')
b.set_ylim(0, 9)
for x, v in zip(xx, p_lost+p_wrong): b.text(x, v+.18, f'{v:.2f}%', ha='center', fontsize=8)
panel(b, '(b)', 'Production placement benchmarks')
save(fig, 'first_pmt_lookup',
    'Endpoint-local candidate lookup misses PMTs along the ray. Each condition uses 20,000 isotropic rays and a float64 oracle over all 11,096 PMTs, bounded by the first water-cylinder exit. '
    'Denominators include only oracle-confirmed PMT hits. Stacked bars distinguish missed hits from an incorrect first-PMT identity. '
    'Production-volume positions follow the default 0.9-radius/0.9-height placement. These fractions describe geometric rays, not integrated event PE loss.',
    ['geometry/results.json', 'geometry/production_population_results.json', 'geometry/reproduce_geometry.py'],
    {'radial_metrics': radial, 'production_metrics': rows,
     'minimal_tangent': geo['examples']['minimal_tangent_barrel'][0]})

# Read actual event-level capture tags from saved ROOT outputs.
def capture_times(rel):
    with uproot.open(BASE / rel) as f:
        a = f['OpticalPhotons'].arrays(['TrackInfo_PDG', 'TrackInfo_CreatorProcess', 'TrackInfo_Time'], library='np')
    times = []
    for pdg, process, time in zip(a['TrackInfo_PDG'], a['TrackInfo_CreatorProcess'], a['TrackInfo_Time']):
        mask = (pdg == 22) & (process == 'nCapture')
        if np.any(mask): times.append(float(np.min(time[mask])) / 1000)
    return np.asarray(times)

root_original = 'geant4_physics/thermal_neutron_fixed.root'
root_repaired = 'geant4_physics/thermal_neutron_thermalelastic.root'
original = capture_times(root_original)
repaired = capture_times(root_repaired)
thermal = read('geant4_physics/thermal_results.json')
thermal_fix = read('geant4_physics/thermal_thermalelastic_results.json')
for times, summary in [(original, thermal), (repaired, thermal_fix)]:
    assert len(times) == summary['captures']
    assert np.isclose(times.mean(), summary['mean_us'], rtol=1e-12)
tau = thermal['rate_oracle_us']
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), layout='constrained')
a = axs[0]
bins = np.arange(0, 1600, 100)
a.hist(original, bins=bins, weights=np.full(len(original), 1/len(original)), histtype='step', color=ORANGE, lw=1.6, label='Original elastic')
a.hist(repaired, bins=bins, weights=np.full(len(repaired), 1/len(repaired)), histtype='step', color=BLUE, lw=1.6, label='Thermal elastic fix')
expected = np.exp(-bins[:-1]/tau)-np.exp(-bins[1:]/tau)
a.plot((bins[:-1]+bins[1:])/2, expected, '--', color='black', lw=1.1, label='Independent 204.4 us clock')
a.set_xlabel('Tagged capture time (us)')
a.set_ylabel('Capture probability / 100 us bin')
a.set_xlim(0, 1500)
a.legend(frameon=False, loc='upper right', fontsize=7)
panel(a, '(a)', '0.025 eV neutrons in pure water')
b = axs[1]
for times, color, label in [(original, ORANGE, 'Original elastic'), (repaired, BLUE, 'Thermal elastic fix')]:
    st = np.sort(times)
    b.step(np.r_[0, st], np.r_[1, (len(st)-np.arange(1, len(st)+1))/len(st)], where='post', color=color, lw=1.5, label=label)
x = np.linspace(0, 2000, 300)
b.plot(x, np.exp(-x/tau), '--', color='black', lw=1.1)
b.axvline(1000, color=GRAY, lw=.7, ls=':')
b.set_yscale('log')
b.set_ylim(4e-4, 1.2)
b.set_xlim(0, 2000)
b.set_xlabel('Tagged capture time (us)')
b.set_ylabel('Survival probability P(T > t)')
b.text(1030, .075, 'P(T > 1 ms)\nOriginal: 5.92%\nFixed: 0.70%\nOracle: 0.75%', fontsize=7.5)
panel(b, '(b)', 'Excess late-capture tail')
save(fig, 'neutron_capture',
    'Actual saved ROOT capture tags for 1,000 initially thermal neutrons in pure water. The unrelated 10 us track cutoff is removed in both samples. '
    'Original elastic physics yields 997 tagged captures with mean 289.94 +/- 15.86 us; the isolated thermal-elastic replacement yields 1,000 captures with mean 202.71 +/- 6.35 us (standard errors). '
    'The independent clock is 204.416 us from hydrogen density and its 2200 m/s capture cross section. Histograms and survival curves normalize to tagged captures; three original events lack an nCapture gamma tag. '
    'The original maximum capture time is 6,814 us, beyond the plotted range. This validates a thermal pure-water benchmark, not all neutron physics.',
    [root_original, root_repaired, 'geant4_physics/thermal_results.json', 'geant4_physics/thermal_thermalelastic_results.json', 'geant4_physics/thermal_tail_summary.json'],
    {'original': thermal, 'thermal_elastic_fix': thermal_fix, 'tails': read('geant4_physics/thermal_tail_summary.json')})
# Source losses: time-shift proof and a separate near-threshold calibration A/B.
source = read('geant4_physics/source_results.json')
compare = read('geant4_physics/comparison_results.json')
index_ab = read('photon_sources/index_ab_results.json')
gamma = source['late_gamma']
with uproot.open(BASE / 'geant4_physics/gamma_delayed_fixed.root') as f:
    gamma_fixed = f['OpticalPhotons']['NOpticalPhotons'].array(library='np')
assert np.array_equal(gamma_fixed, gamma['prompt_per_event'])
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), layout='constrained')
a = axs[0]
gamma_arrays = [gamma['prompt_per_event'], gamma['delayed_per_event'], gamma_fixed]
means = np.array([np.mean(v) for v in gamma_arrays])
errors = np.array([sem(v) for v in gamma_arrays])
a.bar([0, 1, 2], means, color=[GRAY, ORANGE, BLUE], width=.6, zorder=3)
a.errorbar([0, 1, 2], means, yerr=errors, fmt='none', color='black', capsize=3, lw=.8, zorder=4)
a.set_xticks([0, 1, 2], ['Prompt', '200 us\ncurrent', '200 us\ncut removed'])
a.set_ylabel('Generated optical photons / event')
a.set_ylim(0, 285)
for x, v, e in zip(range(3), means, errors): a.text(x, v+e+7, f'{v:.1f}', ha='center', fontsize=8)
panel(a, '(a)', 'Identical 2.2 MeV gamma; 30 events')
b = axs[1]
x = np.arange(len(index_ab['rows']))
width = .3
for offset, key, color, label in [(-width/2, 'original', ORANGE, 'Original water index'), (width/2, 'measured_phase_index', BLUE, 'Measured phase-index table')]:
    arrays = [r[key]['primary_cherenkov_per_event'] for r in index_ab['rows']]
    ys = np.array([np.mean(v) for v in arrays])
    es = np.array([sem(v) for v in arrays])
    b.bar(x+offset, ys, width=width, color=color, label=label, zorder=3)
    b.errorbar(x+offset, ys, yerr=es, fmt='none', color='black', capsize=3, lw=.8, zorder=4)
    for xx, yy, ee in zip(x+offset, ys, es): b.text(xx, yy+ee+.2, f'{yy:.2f}', ha='center', fontsize=8)
b.set_xticks(x, [str(r['kinetic_MeV']) for r in index_ab['rows']])
b.set_xlabel('Muon initial kinetic energy (MeV)')
b.set_ylabel('Primary-track Cherenkov photons / event')
b.set_ylim(0, 12.8)
b.legend(loc='upper left', fontsize=7, frameon=False)
panel(b, '(b)', 'Near-threshold calibration; 100 events')
save(fig, 'source_light',
    'Two independent source-stage benchmarks. Left: an identical-seed 30-event 2.2 MeV gamma sample loses 93.24% of its generated photons when shifted to 200 us; removing the global-time cutoff exactly restores the prompt per-event counts. '
    'Right: changing only the water phase-index table changes primary-muon Cherenkov yield near threshold. At 54 MeV the original yields 1.49 versus 9.47 photons/event (84.27% deficit). '
    'The right panel excludes secondary-particle light and does not represent the effect on GeV campaigns. All error bars are the standard error across recorded event counts.',
    ['geant4_physics/source_results.json', 'geant4_physics/comparison_results.json', 'geant4_physics/gamma_delayed_fixed.root', 'photon_sources/index_ab_results.json'],
    {'gamma_means': means.tolist(), 'gamma_standard_errors': errors.tolist(),
     'gamma_total_counts': [int(np.sum(v)) for v in gamma_arrays], 'index_ab': index_ab})

# Recorded primary input direction, before optional LUCiD re-rotation.
def genie_z(rel):
    with uproot.open(BASE / rel) as f:
        return np.asarray([row[0] for row in f['OpticalPhotons']['TrackInfo_DirZ'].array(library='np')])

genie_original = genie_z('geant4_physics/genie_isotropic.root')
genie_fixed = genie_z('geant4_physics/genie_isotropic_fixed.root')
assert len(genie_original) == compare['original']['isotropy']['n']
assert np.isclose(genie_original.mean(), compare['original']['isotropy']['mean_cos_theta'])
assert np.isclose(genie_fixed.mean(), compare['_fixed']['isotropy']['mean_cos_theta'])
fig, a = plt.subplots(figsize=(7.1, 2.8), layout='constrained')
bins = np.linspace(-1, 1, 11)
a.hist(genie_original, bins=bins, weights=np.full(len(genie_original), 1/(len(genie_original)*.2)), histtype='step', lw=1.8, color=ORANGE, label='Original axis-angle sampling')
a.hist(genie_fixed, bins=bins, weights=np.full(len(genie_fixed), 1/(len(genie_fixed)*.2)), histtype='step', lw=1.8, color=BLUE, label='Uniform SO(3) correction')
a.axhline(.5, color='black', ls='--', lw=1.1, label='Uniform-direction density = 0.5')
a.set_xlabel('Primary neutrino direction cos(theta) relative to original +z')
a.set_ylabel('Probability density')
a.set_xlim(-1, 1)
a.set_ylim(0, 1.8)
a.legend(loc='upper left', frameon=False, fontsize=8)
a.text(.43, .84, 'Forward hemisphere\nOriginal: 70.40%\nCorrected: 49.71%\nExpected: 50%', transform=a.transAxes, va='top', fontsize=8)
a.grid(axis='y', zorder=0)
a.set_title('GENIE isotropic option: 20,000 forward-neutrino inputs per sample', loc='left', weight='bold', pad=9)
save(fig, 'genie_isotropy',
    'The GENIE isotropic option rotates identical initially forward neutrinos with a nonuniform axis-angle distribution. '
    'Saved ROOT directions show a +z hemisphere fraction of 70.395% and mean cos(theta) 0.33213. '
    'The isolated uniform-SO(3) correction gives 49.710% and -0.003401, consistent with isotropy. '
    'Each sample contains 20,000 recorded neutrino directions. The uniform density is an independent analytic reference, not a fit.',
    ['geant4_physics/genie_isotropic.root', 'geant4_physics/genie_isotropic_fixed.root', 'geant4_physics/source_results.json', 'geant4_physics/comparison_results.json'],
    {'original': compare['original']['isotropy'], 'corrected': compare['_fixed']['isotropy'],
     'original_histogram': np.histogram(genie_original, bins=bins)[0].tolist(),
     'corrected_histogram': np.histogram(genie_fixed, bins=bins)[0].tolist(), 'bin_edges': bins.tolist()})
(OUT / 'figure_manifest.json').write_text(json.dumps(MANIFEST, indent=2) + '\n')
print(json.dumps({'job_id': os.environ['SLURM_JOB_ID'], 'figures': [f['name'] for f in MANIFEST['figures']]}))
