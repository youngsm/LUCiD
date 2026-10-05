"""Compare PhotonSim's registered water index with measured water dispersion."""
import json
import os
from pathlib import Path
import re

import numpy as np

BASE = Path(__file__).resolve().parent
PHOTON_SOURCE = BASE.parent.parent / "wc_data_20261004/sources/PhotonSim-v1.0.0/src/DetectorConstruction.cc"


def cpp_array(text, name, dimension):
    body = re.search(rf"G4double\s+{name}\[{dimension}\]\s*=\s*\{{([^}}]+)\}}", text).group(1)
    return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)])


def measured_n(wavelength_nm):
    # Daimon & Masumura (2007), published four-term Sellmeier fit, lambda in um.
    text = (BASE / "water_Daimon_20C.yml").read_text()
    coefficients = np.array([float(x) for x in re.search(r"coefficients: (.+)", text).group(1).split()])
    lam2 = (np.asarray(wavelength_nm) / 1000.0)**2
    n2 = 1.0 + coefficients[0]
    for strength, pole in coefficients[1:].reshape(-1, 2):
        n2 = n2 + strength * lam2 / (lam2 - pole)
    return np.sqrt(n2)


def main():
    assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
    text = PHOTON_SOURCE.read_text().split("G4Material* DetectorConstruction::ConstructWater()")[1].split("G4Material* DetectorConstruction::ConstructLiquidArgon()")[0]
    source_e = cpp_array(text, "photonEnergy", "nEntries")
    source_n = cpp_array(text, "refractiveIndex", "nEntries")
    e = np.linspace(source_e.min(), source_e.max(), 50_001)
    wl = 1239.8419843320026 / e
    old_n = np.interp(e, source_e, source_n)
    physical_n = measured_n(wl)
    wcsim = re.sub(r"/\*.*?\*/", "", (BASE / "WCSimConstructMaterials.cc").read_text(), flags=re.S)
    wcsim_e = cpp_array(wcsim, "ENERGY_water", "NUMENTRIES_water") * 1e9
    wcsim_n = cpp_array(wcsim, "RINDEX1", "NUMENTRIES_water")
    sk_n = np.interp(e, wcsim_e, wcsim_n)
    curve = json.loads(Path("config/pmt/SK_QE.json").read_text())
    qe = np.interp(wl, curve["wavelengths_nm"], np.array(curve["qe_percent"]) / 100, left=0, right=0)
    muon_mass_mev = 105.6583755
    alpha = 1 / 137.035999084
    hbarc_ev_m = 197.3269804e-9
    rows = []
    for kinetic in (50, 52, 54, 55, 56, 58, 60, 70, 100, 1000):
        beta = np.sqrt(1 - (muon_mass_mev / (muon_mass_mev + kinetic))**2)
        densities = [alpha / hbarc_ev_m * np.maximum(1 - 1 / (beta * n)**2, 0) for n in (old_n, physical_n, sk_n)]
        yields = [float(np.trapezoid(y, e)) for y in densities]
        qe_yields = [float(np.trapezoid(y * qe, e)) for y in densities]
        rows.append({"kinetic_MeV": kinetic, "beta": float(beta), "source_photons_per_m": yields[0], "measured_photons_per_m": yields[1], "SK_WCSim_photons_per_m": yields[2], "source_QE_weighted_per_m": qe_yields[0], "measured_QE_weighted_per_m": qe_yields[1], "SK_WCSim_QE_weighted_per_m": qe_yields[2], "QE_weighted_loss_fraction": 1 - qe_yields[0] / qe_yields[1] if qe_yields[1] else None})
    wavelength_rows = []
    for wavelength in (300., 350., 400., 500., 600.):
        ns = np.interp(1239.8419843320026 / wavelength, source_e, source_n)
        nm = measured_n(wavelength)
        nsk = np.interp(1239.8419843320026 / wavelength, wcsim_e, wcsim_n)
        thresholds = [muon_mass_mev * (1 / np.sqrt(1 - 1 / n**2) - 1) for n in (ns, nm, nsk)]
        angles = [np.degrees(np.arccos(1 / n)) for n in (ns, nm, nsk)]
        wavelength_rows.append({"wavelength_nm": wavelength, "source_n": float(ns), "measured_n": float(nm), "SK_WCSim_n": float(nsk), "source_threshold_MeV": float(thresholds[0]), "measured_threshold_MeV": float(thresholds[1]), "source_beta1_angle_deg": float(angles[0]), "measured_beta1_angle_deg": float(angles[1])})
    summary = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ["SLURM_JOB_PARTITION"], "emission_band_nm": [float(wl.min()), float(wl.max())], "wavelength_rows": wavelength_rows, "yield_rows": rows}
    assert next(row for row in rows if row["kinetic_MeV"] == 54)["QE_weighted_loss_fraction"] > .5
    print(json.dumps(summary, indent=2), flush=True)
    (BASE / "cherenkov_index_results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
