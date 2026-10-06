"""Independent positive-part Frank--Tamm integral for actual pinned material."""
import json, os
from pathlib import Path
assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
data=np.genfromtxt(BASE/'mean_yield.csv',delimiter=',',names=True)
knots=[line.split(',')[1:] for line in (BASE/'material.txt').read_text().splitlines() if line.startswith('Knot,')]
energy,n,ref=np.array(knots,dtype=float).T
beta=data['beta']
# Exact integral for piecewise-linear n(E): integral(1/n(E)^2)dE = deltaE/(na*nb).
exact=np.zeros(len(beta))
for a,b,na,nb in zip(energy[:-1],energy[1:],n[:-1],n[1:]):
    threshold=1/beta
    lower=np.maximum(a,a+(threshold-na)*(b-a)/(nb-na))
    lower=np.minimum(lower,b)
    nl=na+(nb-na)*(lower-a)/(b-a)
    exact+=(b-lower)*np.maximum(0,1-1/(beta**2*nl*nb))
exact*=369.81/10  # photons / (mm eV), z=1
coarse=data['coarse_per_mm']
dense=data['dense_linear_per_mm']
np.testing.assert_allclose(dense[exact>.01],exact[exact>.01],rtol=.003,atol=5e-6)
mask=(exact>0)&(coarse<=0)
q=json.loads((ROOT/'config/pmt/SK_QE.json').read_text())
grid=np.linspace(energy[0],energy[-1],20001)
gridn=np.interp(grid,energy,n)
gridq=np.interp(np.clip(1239.841984332/grid,300,q['wavelengths_nm'][-1]),q['wavelengths_nm'],np.array(q['qe_percent'])/100)
rows=[]
for target in (.7318,.732,.733,.734,.735,.736,.737,.738,.740,.745,.750,.8,.9,.99):
    i=int(np.argmin(abs(beta-target)))
    spectrum=np.maximum(0,1-1/(beta[i]*gridn)**2)
    eq=np.trapezoid(spectrum*gridq,grid)*369.81/10
    rows.append({'beta':float(beta[i]),'electron_KE_MeV':float(.51099895*(1/np.sqrt(1-beta[i]**2)-1)),
                 'muon_KE_MeV':float(105.6583755*(1/np.sqrt(1-beta[i]**2)-1)),
                 'proton_KE_MeV':float(938.27208816*(1/np.sqrt(1-beta[i]**2)-1)),
                 'coarse_photons_per_mm':float(coarse[i]),'exact_photons_per_mm':float(exact[i]),
                 'ratio_coarse_to_exact':float(max(coarse[i],0)/exact[i]) if exact[i]>0 else None,
                 'exact_QEweighted_photons_per_mm':float(eq),
                 'lost_QEweighted_photons_per_mm':float(eq*(1-max(coarse[i],0)/exact[i])) if exact[i]>0 else 0})
result={'job_id':os.environ['SLURM_JOB_ID'],'partition':os.environ['SLURM_JOB_PARTITION'],
        'node':os.uname().nodename,'max_abs_n_knot_minus_Sellmeier20C':float(np.max(abs(n-ref))),
        'beta_true_threshold':float(1/n[-1]),
        'max_dense20C_vs_piecewise_linear_fraction_for_beta_ge_08':float(np.max(np.abs(data['dense_physical_per_mm'][beta>=.8]/exact[beta>=.8]-1))),
        'allowed_but_no_G4_emission_beta_range': [float(beta[mask].min()),float(beta[mask].max())],
        'rows':rows,
        'dense_exact_comparison_pass':True,
        'QE_convention':'Actual production medium clips wavelengths to [300,648.22]nm before QE, including UV tail.',
        'interpretation':'No total-event bias implied; GEANT near-threshold rate uses negative/coarse integral; dense_linear preserves exact same n(E).'}
(BASE/'mean_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
