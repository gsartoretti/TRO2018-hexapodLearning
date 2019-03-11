import numpy as np

def jacobianAngleCorrection(cpg):
    ## Calculate Angle Correction

    J = cpg['smk'].getLegJacobians(cpg['legs'])
    
    theta = np.zeros((3,6))

    for leg in range(6):
        theta[:,leg] = np.linalg.lstsq(J[leg,:3,:], cpg['dx'][:,leg])[0]
    
    return theta
