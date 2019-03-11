from math import sin, cos, atan2, sqrt
import numpy as np
from hebiapi import kinematics
from hebiapi.base import flatten
from copy import copy
import seatools.hexapod as hp
import scipy.optimize

import time
import hebiapi
#import setupfunctions as setup
import tools
import SMCF
from SMCF.SMComplementaryFilter import feedbackStructure,decomposeSO3
import seatools.hexapod as hp
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs
from setupfunctions import setupSnakeMonsterShoulderData
from Functions.stabilizationPID import stabilizationPID

pi = np.pi

class NewConfig(object):

    def __init__(self):
        # self.smk = HexapodKinematicsForIK()
        self.smk = hp.HexapodKinematics()
        self.baseAngles = np.array(([np.random.uniform(0,-pi/3), np.random.uniform(0,pi/3), 
                                np.random.uniform(-pi/8,pi/8), np.random.uniform(-pi/8,pi/8),
                                np.random.uniform(0,pi/3), np.random.uniform(0,-pi/3)])).reshape(1,6)
        self.q0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.expectedPoints = np.array(([0.25, -0.25, 0.25, -0.25, 0.25, -0.25],[0., 0., 0., 0., 0., 0.]))

        xyzrpy = np.array(([0, 0, 0.21, 0.2, 0, 0]))
        self.baseFrame = self.smk.trans(xyzrpy)

    def findConfig(self):
        # Numerical IK
        return self.inverseKinematics()

    def inverseKinematics(self):
        def FK(q): # where q is a list of 12 angles [leg1_joint2, leg1_joint3, leg2_joint2, leg2_joint3, and so on]
            qReshaped = []
            for i in range(6):
                qReshaped.append(self.baseAngles[0,i])
                qReshaped.append(q[2*i])
                qReshaped.append(q[2*i+1])

            # frames of all 6 foots as seen from body frame
            endPositionsBaseFrame = self.smk.getLegPositions(qReshaped)

            # frames of all 6 foots as seen from world frame
            endPositionsWorldFrame = np.matmul(self.baseFrame, np.vstack((endPositionsBaseFrame.T, np.ones((1,6)))))
            
            # [x,z] components of position of all 6 foots as seen from world frame
            endFramesCoordinates = np.vstack((endPositionsWorldFrame[0,:], endPositionsWorldFrame[2,:]))
            
            positionDifference = endFramesCoordinates - self.expectedPoints
            
            # more weight should be given to z components as we want all foot to be on ground
            cost = np.matmul(np.array(([1, 1])),np.sum(np.power(positionDifference,2),axis=1))
            
            # way of putting constraints on joint angles. 
            # baseAngles are always in range by default (randomly generated)
            if min(q)<-pi/2 or max(q)>pi/2:
                cost += 1000
            return cost

        # angles = scipy.optimize.fmin_slsqp( func=FK, x0=self.q0, acc=1e-4, iter=1000) # iprint=0 suppresses output
        angles = scipy.optimize.fmin(func=FK, x0 = self.q0, disp=1, maxfun=10000)
        # angles = scipy.optimize.minimize(fun=FK, x0 = self.q0, tol=1.)

        # # concatenate angles from IK with baseAngles
        # # we want a list of 18 angles: [leg1_joint1, leg1_joint2, leg1_joint3, leg2_joint1, leg2_joint2, leg2_joint3, and so on]
        jointAngles = np.zeros((1,18))
        for i in range(6):
            jointAngles[3*i] = kin.baseAngles[0,i]
            jointAngles[3*i + 1] = angles[2*i]
            jointAngles[3*i +2] = angles[2*i+1]

        return jointAngles

    def cpgConfig(self,cpg,randHeight=0.):

        
        cpg['legs'] = np.zeros((1,18))

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

        ## CPG
        r0s = r0 * xKy / r0[2]

        if np.random.rand() > randHeight:
            flag = True
            while flag:
                cpg['legs'][0,6] = np.random.uniform(-pi/4,pi/4)
                cpg['legs'][0,9] = np.random.uniform(-pi/4,pi/4)

                delta = pi/12
                angleMin, angleMax = cpg['s1Off']-delta, cpg['s1Off']+delta
                cpg['legs'][0,0] = np.random.uniform(min(cpg['legs'][0,6], -angleMin),  -angleMax)
                cpg['legs'][0,3] = np.random.uniform(max(cpg['legs'][0,9],  angleMin),   angleMax)
                cpg['legs'][0,12] = np.random.uniform(max(cpg['legs'][0,6], angleMin),   angleMax)
                cpg['legs'][0,15] = np.random.uniform(min(cpg['legs'][0,9], -angleMin), -angleMax)

                delta2 = pi/3
                if np.random.rand() < 0.5:
                    cpg['legs'][0,[1,10,13]] = np.random.uniform(0, delta2, size=[1,3])
                else:
                    cpg['legs'][0,[1,10,13]] = np.random.uniform(-delta2, 0, size=[1,3])

                if np.random.rand() < 0.5:
                    cpg['legs'][0,[4,7,16]] = np.random.uniform(-delta2, 0, size=[1,3])
                else:
                    cpg['legs'][0,[4,7,16]] = np.random.uniform(0, delta2, size=[1,3])

                sinVal = np.maximum(-np.ones((1,6)), np.minimum(np.ones((1,6)), (r0s/np.cos(cpg['legs'][0,shoulders1]) - L1*np.cos(cpg['legs'][0,shoulders2]))/L2))
                cpg['legs'][0,elbows] = np.arcsin(sinVal) - cpg['legs'][0,shoulders2] - np.pi/2 + endTheta
                print(cpg['legs'])
                if (cpg['legs'][0,elbows] > 0).any():
                    flag = True
                else:
                    flag = False
        else:
            cpg['legs'][0,6] = np.random.uniform(-pi/4,pi/4)
            cpg['legs'][0,9] = np.random.uniform(-pi/4,pi/4)

            delta = pi/12
            angleMin, angleMax = cpg['s1Off']-delta, cpg['s1Off']+delta
            cpg['legs'][0,0] = np.random.uniform(min(cpg['legs'][0,6], -angleMin),  -angleMax)
            cpg['legs'][0,3] = np.random.uniform(max(cpg['legs'][0,9],  angleMin),   angleMax)
            cpg['legs'][0,12] = np.random.uniform(max(cpg['legs'][0,6], angleMin),   angleMax)
            cpg['legs'][0,15] = np.random.uniform(min(cpg['legs'][0,9], -angleMin), -angleMax)

            cpg['legs'][0,[1,4,7,10,13,16]] = np.random.uniform(-0.25*pi, -0.1*pi, size=[1,6])

            sinVal = np.maximum(-np.ones((1,6)), np.minimum(np.ones((1,6)), (r0s/np.cos(cpg['legs'][0,shoulders1]) - L1*np.cos(cpg['legs'][0,shoulders2]))/L2))
            cpg['legs'][0,elbows] = np.arcsin(sinVal) - cpg['legs'][0,shoulders2] - np.pi/2 + endTheta
            print(cpg['legs'])

        # Store other values in cpg structure
        cpg['x'][0] = cpg['legs'][0,shoulders1] * shoulders1Corr - shoulder1Offsets
        cpg['theta2'] = (cpg['legs'][0,shoulders2] - shoulder2Offsets)[0]
        cpg['y'][0] = np.minimum( np.zeros((1,6))[0] , cpg['theta2'] )
        
        return cpg

if __name__ == '__main__':
    kin = NewConfig()
    # angles = kin.findConfig()

    TIME_FACTOR = 1
    ## tensorboard --logdir=worker_0:'./train_W_0',worker_1:'./train_W_1',worker_2:'./train_W_2',worker_3:'./train_W_3',worker_4:'./train_W_4',worker_5:'./train_W_5'

    ## A3C parameters
    OUTPUT_GRAPH            = True
    LOG_DIR                 = './log'
    GLOBAL_NET_SCOPE        = 'Global_Net'
    UPDATE_GLOBAL_ITER      = round(50 / TIME_FACTOR)
    #GAMMA                   = 0.999 ** TIME_FACTOR
    GAMMA                   = .995
    ENTROPY_BETA            = 1e-3
    LR_A                    = 1e-3    # learning rate for actor
    LR_C                    = 1e-3    # learning rate for critic
    GLOBAL_REWARD           = []
    GLOBAL_EP               = 0
    N_WORKERS               = 6 #should = multiprocessing.cpu_count()
    model_path              = './model_online'
    s_size                  = 7
    a_size                  = 9
    actionList              = [-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1]
    #a_size                  = 3
    #actionList              = [-.5, 0, .5]
    load_model              = False
    continue_EP             = False



    print('Setting up Snake Monster...')

    names = SMCF.NAMES

    HebiLookup = tools.HebiLookup
    shoulders = names[::3]
    imu = HebiLookup.getGroupFromNames(shoulders)
    snakeMonster = HebiLookup.getGroupFromNames(names)
    while imu.getNumModules() != 6 or snakeMonster.getNumModules() != 18:
        print('Found {} modules in shoulder group, {} in robot.'.format(imu.getNumModules(), snakeMonster.getNumModules()), end='  \n')
        imu = HebiLookup.getGroupFromNames(shoulders)
        snakeMonster = HebiLookup.getGroupFromNames(names)
    print('Found {} modules in shoulder group, {} in robot.'.format(imu.getNumModules(), snakeMonster.getNumModules()))
    snakeData = setupSnakeMonsterShoulderData()
    smk = hp.HexapodKinematics()

    print('Setup complete!')

    ## Initialize Variables

    T = 1e5
    dt = 0.02
    nIter = round(T/dt)
    prevReward = [0,0,0,0,0,0]

    forward = np.ones((1,6))
    backward = -1 * np.ones((1,6))
    leftturn = np.array([[1, -1, 1, -1, 1, -1]])
    rightturn = np.array([[-1, 1, -1, 1, -1, 1]])

    shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
    shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
    elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

    cpg = {
        'initLength': 250,
        'bodyHeight': 0.18,
        'r': 0.10,
        'direction': forward,
        ## SUPERELLIPSE
        #'w': 60,
        #'a': 45 * np.ones((1,6)),
        #'b': 3.75 * np.ones((1,6)),
        #'x': 5.0 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
        #'y': 20.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
        ## /SUPERELLIPSE
        'w': 7,
        'a': 0.03 * np.ones((1,6)),
        'b': 1.00 * np.ones((1,6)),
        'mu': np.array([0.0412, 0.0412, 0.0882, 0.0882, 0.0412, 0.0412]),
        'x': 0.0 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
        'y': 0.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
        's1Off': np.pi/3,
        's2Off': np.pi/16,
        't3Str': 0.0,
        'stabilize': True,
        #'isStance': np.zeros((1,6)),
        'dx': np.zeros((3,6)),
        'legs': np.zeros((1,18)),
        'move': True,
        'smk': smk,
        'pose': np.identity(3),
        'zErr': 0.0,
        'zHistory': np.ones((1,10)),
        'zHistoryCnt': 0,
        'theta2': np.array([0.,0.,0.,0.,0.,0.]),
        'dTheta2': np.array([0.,0.,0.,0.,0.,0.]),
        }


    cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']

    ## Walk the Snake Monster

    joy = Controller()
    cpgJoy = True

    print('Finding initial stance...')

    cpg = kin.cpgConfig(cpg)
    # cpg['legs'] = jointAngles
    print(cpg['legs'])

    while True:
        tStart = time.perf_counter()
        snakeMonster.setAngles(cpg['legs'][0])
        loopTime = time.perf_counter() - tStart
        time.sleep(max(0,dt-loopTime))
