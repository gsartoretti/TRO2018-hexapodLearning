import numpy as np
from SMCF import decomposeSO3
from SMCF import SMComplementaryFilter as SMCF

def rotz(theta):
    return SMCF.rotZ(None, theta)

def XYrot(pose):

    _,_,gamma = decomposeSO3(pose)
    newPose = rotz(gamma)
    newPose = newPose[:3][:3]

    y1 = np.dot(pose , np.array([[0],[1],[0]])).T
    y2 = np.dot(newPose , np.array([[0],[1],[0]])).T

    rotM = rotz( np.arctan2(y1[0,1], y1[0,0]) - np.arctan2(y2[0,1], y2[0,0]) )
    newPose = np.dot(rotM[0:3][0:3], newPose)
    T = np.linalg.lstsq(pose.T, newPose.T)[0].T

    return (T, newPose)
