# Residual stopped-pion radiative capture

Research/static review only, 2026-10-06. No simulations or production changes by this agent. Parent runtime result: every scored hydrogen π− capture produced π0n (880/880); no direct capture photon above10MeV appeared in the baseline or water-target-corrected 3000-stop samples. Target correction and nuclear final-state correction are separate requirements.

## Hydrogen: primary measurement and exact convention

[Spuller et al., “A remeasurement of the Panofsky ratio”, Physics Letters67B,479–482(1977)](https://muon.npl.washington.edu/exp/WildIdeas/DarkLiterature/piHe/spuller77.pdf), [DOI](https://doi.org/10.1016/0370-2693(77)90449-X), gives **P=1.546±0.009**. Printedp479 explicitly defines the numerator to include both π0→γγ and π0→e+e−γ, and the denominator to include both π−p→γn and π−p→e+e−n. Thus `1/(1+P)≈39.28%` is the **radiative capture family**, not precisely the real-photon channel alone. The paper measures the real-photon peak near129MeV and the π0-decay photons between55 and83MeV. Printedp481 discusses the internal-conversion correction to the measured ratio; its factor0.999 does not mean that each individual conversion channel has probability0.1%.

The parent observation of100%π0n is therefore decisively incorrect. A two-channel correction with probability `1/(1+P)` for γn and `P/(1+P)` for π0n repairs the dominant branching failure, while approximating the small internal-conversion contribution. Preserve ordinary π0 decays and record that approximation explicitly; do not describe that two-channel model as complete pion-capture physics.

For the direct γn branch, use exact two-body recoil: in the capture-system rest frame, with available invariant mass W, `E_gamma=(W*W-m_n*m_n)/(2*W)` and neutron momentum opposite the photon. For π0n, use the standard two-body momentum from W,m_pi0,m_n. Account for atomic binding/cascade energy once in W. Use nuclear masses consistently, not a mixture of nuclear and atomic masses. A dedicated π− stopping model can delegate non-H targets to the existing cascade; see [component notes](pion_capture_components.md).

## Oxygen: measured spectrum versus extrapolated total

The independently accessible **primary full text** is [Bistirlich et al., “Photon Spectra from Radiative Absorption of Pions in Nuclei”, PRC5,1867(1972)](https://doi.org/10.1103/PhysRevC.5.1867), with [LBL374 author manuscript](https://escholarship.org/content/qt5835t658/qt5835t658.pdf). Locations use printed manuscript pages:

- pp12–13 (PDF pages16–17): the detector measures photons above50MeV; the total rate includes a model extrapolation below50MeV. The authors state that this missing tail is typically less than15% of the total spectrum.
- p24 (PDF page28): total oxygen radiative branching is **(2.24±0.48)%**. Reported components are continuum/pole term1.84±0.38%, ground-state transition0.15±0.03%, and giant-resonance contribution0.25±0.06%.
- p23 (PDF page27), Fig.6: measured oxygen spectrum has continuum and high-energy peaks, rather than one monochromatic photon. The continuum fit has appreciable model dependence.

Therefore **2.24% is not the branching fraction above50MeV**. Its existence and observed high-energy spectrum nevertheless falsify zero direct capture photons above10MeV. This test does not require assuming that the extrapolated total equals the cut-specific acceptance.

[Berridge et al., PRA75,034501(2007)](https://doi.org/10.1103/PhysRevA.75.034501), [author-uploaded full text](https://www.researchgate.net/publication/242310160_Study_ofNuclear_Capture_Ratio_on_Hydrogen_and_Oxygen_in_Water), printedp3, fits photon energies **above50MeV**, with a separate **above80MeV** cross-check. Its oxygen model uses **nine photon lines and two continuum distributions** from [Strassner et al., PRC20,248(1979)](https://doi.org/10.1103/PhysRevC.20.248), and imports **2.27±0.24%** from that paper. The Strassner abstract was read, but full-text retrieval was unsuccessful (APS fulltext endpoint returned401). Consequently this audit does **not** independently establish the extrapolation convention or exact spectral coefficients behind2.27%. It must not be relabeled as a measured above50MeV branching fraction. Berridge's fit threshold alone cannot establish that convention.

## Smallest physically defensible oxygen correction

This is a proposed model boundary, not an implemented or validated final-state model:

1. Add an explicit oxygen radiative-capture branch normalized to a measured total with its uncertainty; specify its measured photon-energy domain and any extrapolation. Sample a validated oxygen spectrum with discrete transitions and continuum. A fixed129MeV photon is a hydrogen model, not an oxygen spectrum.
2. Replace the ordinary absorption final state for that branch. Do not append a photon to a full Bertini absorption event, which would double count energy.
3. Construct the recoil residual with the correct charge: π−+16O→γ+16N*. If the initial capture system is `(W,0)`, sample photon four-vector `k=(E_gamma,E_gamma*n)` and set `P_res=(W,0)-k`. Its invariant excitation is `sqrt(W*W-2*W*E_gamma)-M(16N)`. Restrict photon energies to allowed residual masses/states and apply atomic-energy accounting consistently.
4. De-excite the residual with a supported nuclear de-excitation component, conserving four-momentum, baryon number and charge across all products. For continuum states this can include neutron emission and further photons. A valid recoil nucleus plus generic de-excitation is only a starting model: it does not prove measured state populations, particle correlations, or subsequent gamma branches are reproduced.

The photon spectrum alone does not determine every exclusive residual-nucleus final state. A minimal patch may restore missing prompt electromagnetic events while leaving nuclear-model uncertainty; keep those claims separate.

## Focused verification criteria

- **Hydrogen alone:** score direct nuclear-capture daughters before transport/stacking. Verify nonzero γn near129.4MeV and the Panofsky family ratio, with the conversion convention explicit. Assert per-event energy/momentum and charge/baryon conservation.
- **Oxygen alone:** score direct photons separately from atomic-cascade, de-excitation, secondary-interaction and π0-decay photons. Compare the photon spectrum and branching fraction using the same energy cut as the experimental oracle. Inspect residual A,Z and excitation, then conservation after de-excitation.
- **Water mixture:** combine the independently corrected capture-target probabilities and nuclear branches. Do not use successful target selection as evidence that final-state branches are correct.

All simulation/build/import work for these checks must remain on Slurm milano/roma (CPU) or turing (GPU), with the user's global limit of10 turing GPUs.
