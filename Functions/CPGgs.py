import numpy as np


def CPGgs(cpg, t, dt):
    if t <= cpg['initLength']:
        cpg['legs'] = np.zeros((1,18))

    # Read dTheta from workers
    #theta = cpg['dTheta2']

    #Move back to uncorrected stance
    alpha = 0.1
    theta = cpg['dTheta2'] - alpha * cpg['theta2']

    ## Applies CPG to the first two joints and IK to the last joint. Only apply stabilization to second joint though

    shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
    shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
    elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

    shoulders1Corr      = np.array([1,-1,1,-1,1,-1]) * cpg['direction'] # correction factor for left/right legs
    shoulder1Offsets    = np.array([-1,-1,0,0,1,1]) * cpg['s1Off'] * cpg['direction'] # offset so that legs are more spread out
    shoulder2Offsets    = [cpg['s2Off']] * np.ones((1,6)) #tilt

    # Robot Dimensions
    endWidth = 0.075 # dimensions of the end section of the leg
    endHeight = 0.185
    endTheta = np.arctan(endHeight/endWidth)
    L1 = 0.125 # all distances in m
    L2 = np.sqrt(endWidth**2 + endHeight**2)
    moduleLen = .097

    xKy = cpg['r'] # distance of the central leg from the shoulder
    GSoffset = 0.07

    radCentral = L1*np.cos(shoulder2Offsets[0,0]) + .063-.0122
    r0 = moduleLen + np.array([xKy,xKy,radCentral,radCentral,xKy,xKy]) - GSoffset

    K = [[ 0,-1,-1, 1, 1,-1],
         [-1, 0, 1,-1,-1, 1],
         [-1, 1, 0,-1,-1, 1],
         [ 1,-1,-1, 0, 1,-1],
         [ 1,-1,-1, 1, 0,-1],
         [-1, 1, 1,-1,-1, 0]]

    # CPG Equations
    gamma = 20.0
    lambd = 6.0

    y0 = cpg['theta2']

    ## Normal ellipse
    dx = -2 * cpg['a'] * (cpg['y'][t] - y0)  * cpg['w'] + gamma * (cpg['mu']**2 - (cpg['b'] * cpg['x'][t]**2 + cpg['a'] * (cpg['y'][t] - y0)**2)) * 2 * cpg['b'] *  cpg['x'][t]
    dy =  2 * cpg['b'] *  cpg['x'][t]        * cpg['w'] + gamma * (cpg['mu']**2 - (cpg['b'] * cpg['x'][t]**2 + cpg['a'] * (cpg['y'][t] - y0)**2)) * 2 * cpg['a'] * (cpg['y'][t] - y0) + (np.dot(K,(cpg['y'][t] - y0).T)).T/lambd + theta
    
    cpg['theta2'] += theta * dt
    # Cap theta2 (easier for learning)
    cpg['theta2'] = np.maximum(-shoulder2Offsets-np.pi/2, np.minimum(np.pi/2-shoulder2Offsets, cpg['theta2']))[0]

    if not cpg['move']:
        dx = 0
        dy = theta.copy()
    
    cpg['x'][t+1] = cpg['x'][t] + dx * dt
    cpg['y'][t+1] = cpg['y'][t] + dy * dt

    ## CPG
    r0s = r0 * xKy / r0[2]

    cpg['legs'][0,shoulders1] = (shoulder1Offsets + cpg['x'][t+1]) * shoulders1Corr #CPG Controlled
    cpg['legs'][0,shoulders2] = (shoulder2Offsets + np.maximum(cpg['theta2'], cpg['y'][t+1]))
    
    sinVal = np.maximum(-np.ones((1,6)), np.minimum(np.ones((1,6)), (r0s/np.cos(cpg['legs'][0,shoulders1]) - L1*np.cos(cpg['legs'][0,shoulders2]))/L2))
    cpg['legs'][0,elbows] = np.arcsin( sinVal ) - cpg['legs'][0,shoulders2] - np.pi/2 + endTheta


    return cpg
