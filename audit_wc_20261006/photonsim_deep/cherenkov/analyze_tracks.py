"""True step lengths/endpoints; no reliance on production segment labels."""
import json, os
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') in {'milano','roma'}
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
energy=np.array([1.84,2.07,2.48,2.76,3.10,3.31,3.54,3.81,4.13,4.51])
n=np.array([1.33110,1.33306,1.33680,1.33966,1.34356,1.34624,1.34944,1.35360,1.35915,1.36679])
factor=369.81/10
ca=np.r_[0,np.cumsum(np.diff(energy)*.5*(1/n[:-1]**2+1/n[1:]**2))]
def exact_rate(beta):
    beta=np.maximum(beta,1e-12)
    result=np.zeros_like(beta)
    for a,b,na,nb in zip(energy[:-1],energy[1:],n[:-1],n[1:]):
        lo=np.clip(a+(1/beta-na)*(b-a)/(nb-na),a,b)
        nl=na+(nb-na)*(lo-a)/(b-a)
        result+=(b-lo)*np.maximum(0,1-1/(beta**2*nl*nb))
    return result*factor
def coarse_rate(beta):
    beta=np.maximum(beta,1e-12)
    lo=np.interp(1/beta,n,energy)
    return np.maximum(0,factor*(energy[-1]-lo-(ca[-1]-np.interp(lo,energy,ca))/beta**2))
q=json.loads((ROOT/'config/pmt/SK_QE.json').read_text())
grid=np.linspace(energy[0],energy[-1],50001)
gridn=np.interp(grid,energy,n)
gridq=np.interp(np.clip(1239.841984332/grid,300,q['wavelengths_nm'][-1]),q['wavelengths_nm'],np.array(q['qe_percent'])/100)
def cumulative(x,y):
    return np.r_[0,np.cumsum(np.diff(x)*(y[:-1]+y[1:])/2)]
cq=cumulative(grid,gridq); cqn=cumulative(grid,gridq/gridn**2)
def qe_rate(beta):
    beta=np.maximum(beta,1e-12)
    lo=np.interp(1/beta,n,energy)
    return factor*np.maximum(0,cq[-1]-np.interp(lo,grid,cq)-(cqn[-1]-np.interp(lo,grid,cqn))/beta**2)
betagrid=np.linspace(.65,1,200001)
cb=cumulative(betagrid,exact_rate(betagrid))
cbq=cumulative(betagrid,qe_rate(betagrid))
def linear_beta_rate(a,b,qweighted=False):
    integral=cbq if qweighted else cb
    rate=qe_rate if qweighted else exact_rate
    delta=a-b
    out=rate((a+b)/2)
    valid=np.abs(delta)>1e-8
    out[valid]=(np.interp(a[valid],betagrid,integral)-np.interp(b[valid],betagrid,integral))/delta[valid]
    return out
rows=[]
for path in sorted(BASE.glob('*_steps.csv')):
    variant,scenario=path.stem.rsplit('_steps',1)[0].split('_',1)
    data=np.loadtxt(path,delimiter=',',ndmin=2)
    b1,b2,length,charge,count=data[:,3:].T
    beta=(b1+b2)/2
    weighted=length*charge**2
    exact=exact_rate(beta); coarse=coarse_rate(beta); qe=qe_rate(beta)
    keep=np.divide(coarse,exact,out=np.ones_like(exact),where=exact>0)
    expected=exact*weighted
    cexpected=coarse*weighted
    qexpected=qe*weighted
    qcoarse=qexpected*keep
    events=int(np.max(data[:,0]))+1
    event_expected=np.bincount(data[:,0].astype(int),weights=expected,minlength=events)
    event_qexpected=np.bincount(data[:,0].astype(int),weights=qexpected,minlength=events)
    event_observed=np.bincount(data[:,0].astype(int),weights=count,minlength=events)
    row={'variant':variant,'scenario':scenario,'events':events,'steps':len(data),
      'photons_observed_per_event':float(count.sum()/events),
      'exact_expected_photons_per_event':float(expected.sum()/events),
      'exact_expected_photons_per_event_standard_error':float(event_expected.std(ddof=1)/np.sqrt(events)),
      'observed_photons_per_event_standard_error':float(event_observed.std(ddof=1)/np.sqrt(events)),
      'coarse_expected_photons_per_event_on_same_paths':float(cexpected.sum()/events),
      'sparse_table_expected_fraction_lost':float(1-cexpected.sum()/expected.sum()),
      'exact_QEweighted_per_event':float(qexpected.sum()/events),
      'exact_QEweighted_per_event_standard_error':float(event_qexpected.std(ddof=1)/np.sqrt(events)),
      'sparse_table_QEweighted_fraction_lost':float(1-qcoarse.sum()/qexpected.sum()),
      'linear_beta_path_vs_midpoint_fraction':float((linear_beta_rate(b1,b2)*weighted).sum()/expected.sum()-1),
      'linear_beta_path_vs_midpoint_QEweighted_fraction':float((linear_beta_rate(b1,b2,True)*weighted).sum()/qexpected.sum()-1),
      'emission_weighted_mean_abs_beta_change':float((expected*np.abs(b1-b2)).sum()/expected.sum()),
      'actual_aggregate_count_minus_expected_in_sqrt_expected':float((count.sum()-(cexpected.sum() if variant=='coarse' else expected.sum()))/np.sqrt(cexpected.sum() if variant=='coarse' else expected.sum()))}
    primary=data[:,1]==1
    primary_expected=expected[primary].sum()
    primary_qexpected=qexpected[primary].sum()
    primary_event_expected=np.bincount(data[primary,0].astype(int),weights=expected[primary],minlength=events)
    row['primary_track']={'exact_expected_photons_per_event':float(primary_expected/events),
      'exact_expected_photons_per_event_standard_error':float(primary_event_expected.std(ddof=1)/np.sqrt(events)),
      'observed_photons_per_event':float(count[primary].sum()/events),
      'exact_QEweighted_per_event':float(primary_qexpected/events),
      'sparse_table_expected_fraction_lost':float(1-cexpected[primary].sum()/primary_expected) if primary_expected else None,
      'sparse_table_QEweighted_fraction_lost':float(1-qcoarse[primary].sum()/primary_qexpected) if primary_qexpected else None,
      'linear_beta_path_vs_midpoint_fraction':float((linear_beta_rate(b1,b2)*weighted)[primary].sum()/primary_expected-1) if primary_expected else None}
    hard_candidate=(b1>1/n[-1])&(b2<.9*b1)
    row['large_endpoint_beta_drop']={'step_count':int(hard_candidate.sum()),
      'max_missing_photons_per_event_assuming_entire_step_at_prebeta':float(((exact_rate(b1)-exact)*weighted)[hard_candidate].sum()/events),
      'max_missing_fraction_assuming_entire_step_at_prebeta':float(((exact_rate(b1)-exact)*weighted)[hard_candidate].sum()/expected.sum()),
      'max_missing_QEweighted_fraction_assuming_entire_step_at_prebeta':float(((qe_rate(b1)-qe)*weighted)[hard_candidate].sum()/qexpected.sum()),
      'caveat':'Upper-bound diagnostic for selected >10% beta drops only; endpoint process and continuous loss not separated.'}
    pp=BASE/f'{variant}_{scenario}_photons.csv'
    if pp.stat().st_size:
        phot=np.loadtxt(pp,delimiter=',',ndmin=2)
        p1,p2,e,cos,fraction,time,slength=phot.T
        ni=np.interp(e,energy,n); pb=(p1+p2)/2
        cos_expected=1/(pb*ni)
        residual=cos-cos_expected
        delta_time=slength*fraction/(299.792458*(p1+fraction*(p2-p1)*.5))
        localbeta=p1+fraction*(p2-p1)
        localcos=1/(localbeta*ni)
        row.update({'photon_rows_sampled':len(phot),
                    'max_cosine_residual_actual_vs_G4_mean_beta':float(np.max(np.abs(residual))),
                    'max_time_residual_ns_actual_vs_G4_formula':float(np.max(np.abs(time-delta_time))),
                    'photons_below_local_linear_beta_threshold_fraction':float(np.mean(localcos>1)),
                    'rms_cosine_residual_vs_local_linear_beta':float(np.sqrt(np.mean((cos-localcos)**2)))})
        assert np.max(np.abs(residual))<2e-10
        assert np.max(np.abs(time-delta_time))<1e-7
    rows.append(row)
result={'job_id':os.environ['SLURM_JOB_ID'],'partition':os.environ['SLURM_JOB_PARTITION'],'node':os.uname().nodename,
  'QE_convention':'Production medium clips wavelength to [300,648.22]nm before SK_QE; incident detection weighting, no geometry/water transport.',
  'linear_beta_caveat':'Diagnostic only: linear beta between endpoints is not established physical trajectory; fine-step convergence is stronger.',
  'rows':rows}
(BASE/'track_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
