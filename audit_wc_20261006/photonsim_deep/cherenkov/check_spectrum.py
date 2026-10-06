"""Conditional CDF oracle for the energy-angle correlation of emitted photons."""
import json, os
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION') in {'milano','roma'}
import numpy as np
from scipy.stats import kstest
base=Path(__file__).resolve().parent
knots=np.array([1.84,2.07,2.48,2.76,3.10,3.31,3.54,3.81,4.13,4.51])
indices=np.array([1.33110,1.33306,1.33680,1.33966,1.34356,1.34624,1.34944,1.35360,1.35915,1.36679])
def integral(beta,upper):
    total=np.zeros_like(beta)
    for a,b,na,nb in zip(knots[:-1],knots[1:],indices[:-1],indices[1:]):
        lo=np.clip(a+(1/beta-na)*(b-a)/(nb-na),a,b)
        hi=np.maximum(lo,np.minimum(b,upper))
        nl=na+(nb-na)*(lo-a)/(b-a)
        nh=na+(nb-na)*(hi-a)/(b-a)
        total+=(hi-lo)*np.maximum(0,1-1/(beta**2*nl*nh))
    return total
rows=[]
for sample in ['conv_default','conv_small','conv_smaller','coarse_e100','coarse_mu1000']:
    data=np.loadtxt(base/f'{sample}_photons.csv',delimiter=',',ndmin=2)
    beta=(data[:,0]+data[:,1])/2
    pit=integral(beta,data[:,2])/integral(beta,np.full_like(beta,knots[-1]))
    test=kstest(pit,'uniform')
    rows.append({'sample':sample,'photon_rows':len(pit),'conditional_CDF_uniform_KS_statistic':float(test.statistic),'pvalue':float(test.pvalue)})
    assert test.pvalue>1e-4, (sample,test)
result={'job_id':os.environ['SLURM_JOB_ID'],'partition':os.environ['SLURM_JOB_PARTITION'],'node':os.uname().nodename,
        'interpretation':'Energy conditional on G4 step-mean beta follows the exact dispersive Frank-Tamm spectrum; this does not validate its within-step beta averaging.',
        'rows':rows}
(base/'spectrum_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
