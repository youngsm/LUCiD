# Independent SK_WAND physical calibration cross-check

Reviewed HEAD `20df3094bd563162d40c6ca34c5ad25e1e1d3656`; no production edits.
This lane found **no new demonstrated large event-yield bug**. It found an unresolved absolute-PMT-efficiency calibration question that could be large, plus a verified low-rate scattering-model mismatch.

All Python/numerical execution was on `milano`, CPU only: job `39967639` on `sdfmilan272` and job `39968668` on `sdfmilan271`. The second job reran the same checks with an added spectral impact estimate. Logs are `cpu-<job>.log`; final numbers are in `optical_reference_results.json`.

## Absolute PMT efficiency: consequential, unresolved calibration question

The supplied curve is used as total incident-photon probability to produce a counted PE. At 400 nm it is **0.2261907458**. The current correction for reflection restores this value after the nonreflection branch. `qe_corrections` in SK_WAND is exactly `1.0`. There is no separately applied electron collection efficiency in this path. See `config/SK_WAND_physics_config.json:4`, `lucid/wavelength/medium.py:72`, and `lucid/simulation/simulator.py:588`.

Physical sources distinguish photocathode QE from probability of a photoelectron reaching the amplification chain. [Okajima et al., Table 1](https://inspirehep.net/files/f043ea4dfbaf066f1e0ec243e6d87f15), with Hamamatsu coauthors, lists R3600 peak QE around 22%, collection efficiency 67% for the central 460 mm aperture or 61% over 498 mm, and total efficiencies around 15% or 13%. [Nishimura, Table 1 and Section 3.1](https://s3.cern.ch/inspire-prod-files-f/fd23eeff10e09b37f7ce3283940a852f) gives 67% in its table but 73% in its prose. Those references do not support selecting one universal 73% correction without defining illumination and aperture.

Conversely, [SK's own calibration paper, Section 3.1](https://arxiv.org/pdf/1307.0162) uses the term QE for the product of photocathode and collection efficiencies. Therefore the filename `SK_QE.json` and variable names cannot resolve whether another factor is needed.

Git establishes that commit `dbf5c87` introduced this exact JSON curve. It contains no measurement-source metadata, illumination medium, aperture, or CE convention. The removed `lucid/wavelength/data/sk_qe.csv` contained different values, so its comments cannot authenticate the replacement. The peak's similarity to published cathode QE is evidence to investigate, not proof of identical provenance.

**Conditional size:** if this is bare photocathode QE, a 73% CE gives `0.2261907458 * .73 = 0.1651192445`; treating it as final PDE overcounts direct detected PE by **36.99%**. A 67% CE gives `0.1515477997`, a **49.25%** relative overcount. These are conditional probability calculations, not established real-event errors. If this table is already an effective PDE calibration, applying CE again would itself be a bug.

Minimal numerical diagnostic (already executed in the supplied script):

```python
from lucid.wavelength.medium import load_qe_curve
q = float(load_qe_curve("config/pmt/SK_QE.json")(400.))
print(q, q * .73, 1 / .73 - 1)
# 0.2261907458, 0.1651192445, 0.3698630137
```

**Actionable resolution/minimal fix:** recover the curve's source and whether it is cathode QE or full PDE. If it is cathode QE, apply the independently calibrated CE through the existing `qe_corrections` field (or replace the table with measured incident PDE). The reflectance conditional normalization must use that final PDE. If already PDE, retain normalization and add explicit provenance. Validate absolute low-occupancy charge/occupancy with a known photon flux; a simulator matching its own configured QE does not resolve this.

An extra CE cannot be justified solely from WCSim parity: [WCSim's PMT20inch implementation](https://raw.githubusercontent.com/WCSim/WCSim/develop/src/WCSimPMTObject.cc) inherits a 100% collection-efficiency array, while its QE table is different (`0.211` at 400 nm). These conventions require calibration context too.

## Asymmetric scattering: verified model mismatch, low-rate effect

[The SK calibration fit, Section 3.2.1](https://arxiv.org/pdf/1307.0162), couples its asymmetric coefficient to a forward angular density `p(mu)=2*mu`, `0<=mu<=1`. LUCiD imports that coefficient but uses Henyey–Greenstein with `g=.95` in `lucid/wavelength/scattering.py:76` and `config/materials/water.json`.

One million midpoint quantiles give mean cosine `.95` rather than `2/3`, with **50.61%** rather than **0.7596%** inside a five-degree cone. HG also produces **1.089%** backward scatters in this component; the cited fit produces none. The coefficient itself agrees with the literal published formula to numerical tolerance.

```python
import jax.numpy as jnp
from lucid.wavelength.scattering import hg_sample_cos_theta
u = (jnp.arange(1_000_000) + .5) / 1_000_000
print(hg_sample_cos_theta(u, .95).mean(), jnp.sqrt(u).mean())
# .95, .6666666
```

At 400 nm the probability of at least one asymmetric scatter over 20 m is only **0.2021%**. Weighting `QE(lambda)/lambda**2` by absorption survival gives **0.1502%, 0.2978%, 0.5168%, 0.7345%** over **10, 20, 35, 50 m**, respectively. These are straight-path scale estimates, not detector-level systematic bounds. This is not evidence of a tens-of-percent total photon-yield error.

**Minimal consistent remedy:** if reproducing this SK fit, select a separate SK phase function with `mu=sqrt(U)` and uniform azimuth, retaining its fitted coefficient. Merely changing HG `g` to `2/3` matches one moment but not the angular law. If retaining HG as the model choice, fit the coefficient and `g` jointly to scattering calibration data and record that change of model.

## Checks and limitations

- Independent literal-equation evaluation confirms the shipped blue absorption and symmetric/asymmetric coefficient formulas. At 400 nm the implementation gives absorption length **400.4216 m**, symmetric length **175.6628 m**, and asymmetric length **9885.8188 m**. The paper's rounded narrative absorption number is about 402 m; this tiny difference is not a transcription bug. Red absorption knots are covered by the optical-transport lane.
- The default constant reflectances and specular fractions are not authenticated by merely citing the papers in config comments. [Motta and Schonert](https://arxiv.org/pdf/physics/0408075) analyze wavelength/angle-dependent multilayer PMT optics; that does not establish a universal 25% reflectance or 90% specular fraction for this SK_WAND surrogate. Real angular response and measured late-light distributions still need calibration.
- `_sample_spe_charge` is additive at any PE count; no high-occupancy PMT nonlinearity is present. [SK calibration Section 3.1.7/Fig.19](https://arxiv.org/pdf/1307.0162) reports over 10% nonlinearity above 1000 PE. Parent/electronics lanes assess whether actual WAND sources populate that region. This lane does not assign an event-level error.
- Uniform optical properties omit the SK calibration's measured vertical dependence. This is a calibration limitation, not a new reproduced large failure.
- Endpoint lookup, spherical PMTs, wavelength clipping in the supplied band, late-light cuts, constant light speed, unpolarized scattering, and SPE shape were treated as the user's accepted approximations.

Reproduction: `sbatch audit_wc_20261006/physics_crosscheck/run_cpu.sbatch`.
