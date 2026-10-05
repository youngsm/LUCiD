# Water optics and spectral response audit

Reference checkout is `82f8d24`, with `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json`.
No production code was edited.
Successful full reproduction: milano job `39894510`, node `sdfmilan269`, exit `0:0`, [log](slurm-39894510.log), [structured results](results.json), and [script](reproduce.py).
Minimal failing test and the isolated suggested fix were executed in milano allocation `39894028`, also on `sdfmilan269`.
All numerical work used `uv run`, `JAX_PLATFORMS=cpu`, and zero GPUs.
Initial inspection and creation of the first script preceded the tightened instruction requiring even nonnumerical work on compute nodes; every subsequent inspection and file operation used allocation `39894028`.
The environment reports an incompatible installed CUDA plugin, which JAX ignores; the executed backend was CPU and all independent algebra assertions passed.

## Confirmed defect: the medium-grid clamp changes PMT QE

Severity is low for the ordinary SK_WAND broadband yield, despite a completely false detection probability for photons outside PMT sensitivity.
The data path passes the physical photon wavelength through `simulator.py:823-825` to `_get_optical_arrays`, whose medium grid is limited to `[300, 648.22]` nm at `simulator.py:379-390`.
`lucid/wavelength/optical_model.py:119-120` clamps that wavelength and line 131 evaluates `qe_fn` on the clamped value.
`load_qe_curve` correctly returns zero outside its own support at `medium.py:75-79`, but the preceding clamp bypasses that support check.
The clamp also increases QE for real input wavelengths between the first QE knot, 294.22 nm, and 300 nm.
This affects SK_WAND production because PhotonSim generates wavelengths across approximately 275-674 nm and the production event builder supplies them to the data simulator.
The reviewed PhotonSim source table is at `geant4_physics/PhotonSim/src/DetectorConstruction.cc:89-107`; the separately configured production binary is an unresolved provenance question owned by the source lane.

An independent detector simulation also expects the chosen extreme wavelengths to be invisible: WCSim's 20-inch PMT has zero QE at 280 and 660 nm and limits its QE table to those endpoints. [WCSim PMT source](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimPMTObject.cc#L227-L233)
The stronger immediate contract is the exact SK_WAND QE curve supplied by the user, which itself evaluates to zero at 275 and 674 nm.

| Input wavelength | Configured QE | QE used by simulation | Detected PE from 30,000 incident photons |
|---|---:|---:|---:|
| 275 nm | 0 | 0.00550125 | 128 |
| 400 nm, positive control | 0.22619075 | 0.22619075 | 5080 |
| 500 nm, positive control | 0.13203444 | 0.13203444 | 3027 |
| 674 nm | 0 | 0.002312 | 49 |

The end-user simulator reproduction uses the exact SK_WAND geometry, `is_data=True`, the production `per_segment` hit mode, hard overlap, and a ray fired from 1 m behind the center of PMT 0 toward that PMT.
The initial attempt from the detector center had nearly complete red absorption over 23 m and therefore insufficient statistics at 674 nm; placing the source close to the sensor isolates the spectral error.
The measured mean-free-path settings, reflection, geometry, and sensor code are otherwise the production configuration.

For a bare `1/lambda^2` spectrum over 275-674 nm, comparing current QE with original-wavelength QE under the same current attenuation gives excess charge of 0.675% at zero propagation, 0.496% over 17 m, and 0.383% over 35 m.
These are controlled spectral integrations rather than a measured full-event energy bias, and include the configured direct-light scattering loss.
They should not be described as a large loss of detected light.

The smallest useful failing test is [minimal_qe.py](minimal_qe.py), and its executed output is [minimal_qe_job39894028.log](minimal_qe_job39894028.log).
The file has an allocation guard followed by these statements:

```python
import jax.numpy as jnp
from lucid.detector_params import load_physics_config
from lucid.wavelength.medium import load_qe_curve, make_medium
from lucid.wavelength.optical_model import evaluate_optical_model

dp, material, qe_path = load_physics_config('config/SK_WAND_physics_config.json')
medium = make_medium('water', jnp.linspace(300., 648.22, 200), material)
qe = load_qe_curve(qe_path)
wl = jnp.array([275., 674.])
assert jnp.all(qe(wl) == 0.)
actual = evaluate_optical_model(dp, wl, medium, 2, qe_fn=qe).qe
assert jnp.all(actual == 0.)
```

The actual result is `[0.00550125, 0.002312]`, so the final physical assertion fails.
The minimal suggested fix is to change only `qe_fn(wl)` at optical_model.py:131 to `qe_fn(jnp.asarray(wavelengths))`, while retaining the clamped wavelength for medium interpolation.
An isolated copy of that one-line fix is [optical_qe_fix.diff](optical_qe_fix.diff); its minimal test passes with `[0, 0]` in [minimal_qe_fixed_job39894028.log](minimal_qe_fixed_job39894028.log).
The isolated fixed copy is an audit artifact and was not installed in the simulator.

## Measured approximation: constant phase index used for optical travel time

This is an explicit existing approximation, not included in the confirmed implementation-defect count under the user's rule about deliberate simplifications.
Its measured error is nevertheless relevant to any requirement that the output reproduce physical SK timing.
`medium.py:163-165` sets speed to `c_vac / 1.33`; `medium.py:214` constructs a constant phase-index curve.
`geometry/detector_geometry.py:80,116` stores that scalar, `simulator.py:237` captures it, `simulator.py:664,678` uses it for all propagation and hit times, and `photon_step.py:127` uses it for every traveled leg.
The per-photon wavelength never determines a transport speed.

An optical wavepacket propagates with group speed, derived from the wavelength-dependent phase index, rather than `c/n_phase`.
WCSim provides the SK-based dispersion table and Geant4 calculates its group velocity from the index's derivative with logarithmic photon energy. [WCSim water table](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimConstructMaterials.cc#L363-L401), [Geant4 GROUPVEL implementation](https://github.com/Geant4/geant4/blob/master/source/materials/src/G4MaterialPropertiesTable.cc#L558-L659)
The reproducer transcribes the actual Geant4 finite-difference prescription and interpolates the resulting velocity table, rather than treating the phase index as a group index.
The downloaded WCSim source was compared byte-for-byte with commit `7a15837bd8be5cf26a1989083e50d1c4400e13e5`; the contents are identical.

| Wavelength | Reference phase index | Reference group index | LUCiD early arrival over 17 m | LUCiD early arrival over 35 m |
|---|---:|---:|---:|---:|
| 300 nm | 1.35942 | 1.43481 | 5.943 ns | 12.236 ns |
| 337 nm | 1.35193 | 1.40704 | 4.369 ns | 8.994 ns |
| 375 nm | 1.34675 | 1.38910 | 3.351 ns | 6.900 ns |
| 400 nm | 1.34419 | 1.38093 | 2.888 ns | 5.946 ns |
| 445 nm | 1.34066 | 1.37046 | 2.294 ns | 4.723 ns |
| 500 nm | 1.33753 | 1.36201 | 1.815 ns | 3.737 ns |

QE and absorption weighting of the same 275-674 nm spectrum over 17 m gives a 3.141 ns early bias in its mean and removes a physical dispersion sigma of 0.909 ns.
These values use a WCSim-based SK reference, not an assertion that the particular detector campaign's measured optical group velocity is known exactly.
The end-user data-mode reproduction gives exactly the same first-arrival time, 3.3095603 ns, for 400 and 500 nm along the same ray.
The shortest physical arithmetic check is `17 / make_medium('water').speed_of_light = 75.418957 ns`, while the independent 400 nm group speed gives `78.306845 ns`.
This arithmetic was executed in the full reproducer.

A minimal physical improvement is to add a measured phase-index table and a distinct group-speed curve, evaluate group speed at each original photon wavelength, and thread that per-photon array through both `photon_iteration_sample` and `times_ns`.
For a smooth phase-index curve the required relation is `n_group(lambda) = n_phase(lambda) - lambda * dn_phase/dlambda` and `v_group = c_vac / n_group`.
Changing the scalar phase index to 1.38 would improve one timing reference but would still remove dispersion and conflate different physical quantities.
The source lane separately owns incorrect dispersion in the upstream PhotonSim emission model.

## Checked material and scattering domains

All nine P0-P8 coefficients in water.json match the independently read SK Table 3 values, and the actual formulas reproduce SK Eqs. 14-17 in inverse meters with wavelength in nanometers. [SK calibration paper, section 3.2.1](https://arxiv.org/pdf/1307.0162)
The independently transcribed symmetric/asymmetric formulas and blue absorption formula pass relative-tolerance `1e-6` assertions in the reproducer.
At 400 nm the executed lengths are 400.42 m absorption, 175.66 m symmetric scattering, and 9885.82 m asymmetric scattering.
At the sampled in-range blue wavelengths, the 200-point medium-grid interpolation changes absorption lengths by at most 0.0101% and symmetric-scattering lengths by 0.00513%.
The original diagnostic included wavelengths outside the grid and was corrected to report actual interpolation error; the corrected successful log is job `39894510`.
No large coefficient-unit or grid-interpolation defect was found.

The actual Rayleigh sampler passed independent unpolarized phase-function moments using 200,000 draws: mean cosine -0.00102 and mean squared cosine 0.40044, compared with the physical expectations 0 and 0.4.
The asymmetric angular model deliberately uses HG with g=0.95, yielding mean cosine 0.94989 and 1.084% backward scattering.
SK's empirical asymmetric angular function instead has density `2*cos(theta)` on forward cosine [0,1], so its mean is 2/3 and it has no backward component. [SK calibration paper, section 3.2.1](https://arxiv.org/pdf/1307.0162)
Those angular functions differ materially conditional on an asymmetric scatter, but the default asymmetric length at 400 nm is almost 10 km, so that channel affects only about 0.35% of photons over a 35 m straight path before other processes.
This is a model/calibration mismatch requiring an explicit choice of physical reference, and is not counted automatically as an implementation bug.
If matching the cited SK empirical tune is intended, the minimal angular change is to sample `cos(theta) = sqrt(u)` for that channel and use its corresponding density wherever scores are needed; using the HG g=0.95 density with coefficients fit to a different angle law cannot itself establish physical equivalence.
The reported probability bound is approximate and no complete detector-response impact for this model change was measured.

## Unresolved physical calibration and applicability limits

The Pope-Fry red absorption knots in water.json have correct scale for inverse meters and the implemented red formula was read, but their individual values were not independently checked against the original measurement dataset.
The external-source restriction permits only GitHub and arXiv; the authoritative raw Pope-Fry URL named by the config is elsewhere, so the config comment alone was not accepted as a proof.
The 452-476 nm absorption blend is a deliberate smoothing choice and was not classified automatically as a bug.
The SK calibration paper explicitly describes its optical coefficients as empirically fitted detector-model parameters, with absorption varying over time and with height. [SK calibration paper](https://arxiv.org/pdf/1307.0162)
The fixed April 2009 tune therefore cannot certify a time-specific real detector without appropriate calibration inputs.

SK_QE.json has no machine-readable source, sensor phase, calibration date, or uncertainty, and its peak is 22.68%, compared with 21.1% in the WCSim PMT20inch table. [WCSim PMT source](https://github.com/WCSim/WCSim/blob/7a15837bd8be5cf26a1989083e50d1c4400e13e5/src/WCSimPMTObject.cc#L227-L237)
The curves differ by wavelength, and the difference does not by itself establish that the supplied configuration is wrong.
An authoritative detector QE/collection-efficiency calibration is needed to certify its absolute spectral normalization.
The source lane owns the actual PhotonSim input spectrum and missing/incorrect emission physics, while the PMT and transport lanes own reflection/detection competition and optical boundary behavior.
QEWeighted and PowerLaw spectrum implementations are not exercised by a healthy data-mode event carrying mandatory PhotonSim wavelength arrays; the built-in QE-weighted mode explicitly rejects `is_data=True` at simulator.py:433-438.
Polarization-dependent Rayleigh scattering and detector-specific water-quality evolution remain unverified physical modeling domains in this lane.
No claim of exact physical equivalence or 100% correctness follows from these passing algebra and sampling checks.

## Reproduction commands

The full script can be submitted from an authorized compute allocation with `sbatch audit/wc_data_review_20261004/water_optics/run.sbatch`.
The sbatch file hardcodes milano and the authorized account, has no GPU request, and sets the working directory and output path explicitly.
For the minimal failing or isolated fixed test, execute the following only inside milano or roma:

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python audit/wc_data_review_20261004/water_optics/minimal_qe.py
```

Use `minimal_qe_fixed.py` for the isolated suggested fix, which passes.
Earlier job `39894015` stopped on a harness statistic for an empty red sample, and `39894201` stopped on JSON serialization of a NumPy float32 after already reproducing the spectral issue.
Jobs `39894277` and `39894510` completed with all reproduction assertions passing; the latter contains the corrected in-range interpolation diagnostic.
