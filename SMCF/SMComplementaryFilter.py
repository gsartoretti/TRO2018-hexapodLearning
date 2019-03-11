import argparse
import numpy as np
import math
from numpy import linalg as LA
import SMCF.spinCalcQBottom as psc # Python version of SpinCalc
import sys
import time
from copy import copy
import hebiapi
import tools

# Global Variables
NAMES = ['SA085', 'SA028', 'SA030',
        'SA056', 'SA001', 'SA027',
        'SA035', 'SA032', 'SA026',
        'SA038', 'SA031', 'SA018',
        'SA046', 'SA040', 'SA073',
        'SA041', 'SA047', 'SA077']

HEBI = tools.HebiLookup

class SMComplementaryFilter(object):
        def __init__(self, accelOffset=[], gyroOffset=[], gyrosTrust=[]):

                self.robotDataCreation(robotType="Snake Monster", numModules=6)

                try:
                        if (accelOffset):
                                self.accelOffset = accelOffset
                        else:
                                self.accelOffset = np.zeros(3, self.robotData['numModules'])
                except:
                        self.accelOffset = accelOffset

                try:
                        if (gyroOffset):
                                self.gyroOffset = gyroOffset
                        else:
                                self.gyroOffset = np.zeros(3, self.robotData['numModules'])
                except:
                        self.gyroOffset = gyroOffset

                self.firstRun = True
                self.everUpdated = False

                if(gyrosTrust):
                        # self.gyrosTrustability = np.absolute(anglesSEAtoU(self.robotData, gyrosTrust))
                        self.gyrosTrustability = np.ones(self.robotData['numModules'])
                else:
                        self.gyrosTrustability = np.ones(self.robotData['numModules'])

                self.R = None

        def robotDataCreation(self, robotType, numModules):
                """
                Initializes the data needed to run the control code on one of our robot.
                The hope is that, using this function will allow us to design code that
                can be used on future robot with different configurations,
                with a minimal amount of re-work.

                [IN]:   robot type
                [IN]:   number of modules
                [OUT]:  robot data dictionary containing:
                                -> robotType
                                -> modules
                                -> numModules
                                -> moduleLength
                                -> moduleDiameter
                                -> axisPerm
                                -> robotShape
                """
                self.robotData = {}
                self.robotData['numModules'] = copy(numModules)
                self.robotData['robotType'] = copy(robotType)
                self.robotData['modules'] = np.linspace(1, numModules, num=numModules)
                self.robotData['moduleLength'] = 0.0639         #[m]
                self.robotData['moduleDiameter'] = 0.0508       #[m]
                #####################################################
        # Snake Monster module degrees of freedom #
        #####################################################
        # 1 3 5 -> same direction, 2 4 6 -> turned pi around Y axis
        # Leg Numbering / Chassis Coordinate convention:

                # 2 ----- 1     +y
                #    |          ^
                # 4 ----- 3     |
                #    |          o--> +x
                # 6 ----- 5   +z

                robotShape = np.zeros((4,4,7))
                robotShape[0:3,0:3,1] = np.matmul(self.rotY(math.pi/2), self.rotZ(-math.pi/2)) # should joint 1
                robotShape[0:3,0:3,2] = np.matmul(np.matmul(self.rotY(math.pi/2), self.rotZ(-math.pi/2)), self.rotY(-math.pi)) #should joint 2
                robotShape[0:3,0:3,3] = copy(robotShape[0:3,0:3,1]) # should joint 3
                robotShape[0:3,0:3,4] = copy(robotShape[0:3,0:3,2]) # should joint 4
                robotShape[0:3,0:3,5] = copy(robotShape[0:3,0:3,1]) # should joint 5
                robotShape[0:3,0:3,6] = copy(robotShape[0:3,0:3,2]) # should joint 6
                self.robotData['robotShape'] = copy(robotShape)

                ##################################################
                ###### Permutation Axis for Virtual Chassis ######
                ##################################################

                # Defaults to X-Y-Z being 1st-2nd-3rd Principal Moments
                self.robotData['axisPerm'] = np.identity(3)

        def rotX(self, angle):
                return np.array(([1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                        [0, math.sin(angle), math.cos(angle)]))

        def rotY(self, angle):
                return np.array(([math.cos(angle), 0, math.sin(angle)],
                        [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]))

        def rotZ(self, angle):
                return np.array(([math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle), math.cos(angle), 0], [0, 0, 1]))

        def update(self, fbk):
                #try:
                        self.removeGyroOffset(fbk)
                        self.removeAccelOffset(fbk)
                        if self.firstRun:
                                self.previousTime = fbk.time
                                self.firstRun = False
                        if (fbk.time-self.previousTime)>0.01:
                                self.updateFilter(fbk)
                                self.everUpdated = True
                #except:
                        #return

        def updateFilter(self, fbk):
                # Weight on accelerometer correction term
                # Empirically set value
                accelWeight = .5

                dt = fbk.time - self.previousTime
                dt = max(min(dt, 1), 0.01)

                #########################
                # ACCELEROMETERS UPDATE #
                #########################

                accelVecModule = np.vstack((copy(fbk.accelX), copy(fbk.accelY), copy(fbk.accelZ)))

                # Rotate accelerometer vectors into the body frame
                accelVecBody = np.zeros(accelVecModule.shape)

                for i in range(0, self.robotData['numModules']):
                        accelVecBody[:,i] = np.matmul(self.robotData['robotShape'][0:3,0:3,i+1],accelVecModule[:,i])

                # Average accelerometers
                accelVecAvg = np.mean(accelVecBody, axis=1)

                ###############
                # GYROS UPDATE#
                ###############

                gyroVecModule = np.vstack((copy(fbk.gyroX), copy(fbk.gyroY), copy(fbk.gyroZ)))

                # Rotate gyros into the body frame, taking into account joint angle velocities.
                gyroVecBody = np.zeros(gyroVecModule.shape);
                for i in range(0, self.robotData['numModules']):
                        gyroVecBody[:,i] = np.matmul(self.robotData['robotShape'][0:3,0:3,i+1], gyroVecModule[:,i])

                # Average gyros
                gyroVecAvg = np.mean(gyroVecBody[:,self.gyrosTrustability > 0], axis=1)

                # TODO REMOVE HACK

                #############################
                # CALCULATE THE ORIENTATION #
                #############################

                # ACCELEROMETER GRAVITY VECTOR

                gravityVec = 1.0*accelVecAvg/LA.norm(accelVecAvg)

                upVec = np.array([[0, 0, 1]])

                if not self.everUpdated:
                        accelAxis = np.cross(upVec, gravityVec)
                        print(accelAxis)
                        accelAxis = 1.0*accelAxis/LA.norm(accelAxis)

                        accelAngle = math.acos(upVec.dot(gravityVec.T)) # radians
                        self.q = np.real(psc.EV2Q(np.append(accelAxis, accelAngle), tol=1E-6, ichk=False))

                # ESTIMATE NEW ORIENTATION BY FORWARD INTEGRATING GYROS
                w_x = gyroVecAvg[0]
                w_y = gyroVecAvg[1]
                w_z = gyroVecAvg[2]
                q_from_gyros = self.quatRotate(w_x, w_y, w_z, self.q, dt)

                orientDCM = psc.Q2DCM(q_from_gyros, tol=1E-6, ichk=False)
                orientDCM = orientDCM.transpose()

                # gravityVec
                self.accelGrav = np.matmul(orientDCM.T,gravityVec)

                accelAxis = np.cross(upVec, self.accelGrav)
                if not LA.norm(accelAxis) == 0:
                        accelAxis = 1.0*accelAxis/LA.norm(accelAxis)

                        accelAngle = math.acos(upVec.dot(self.accelGrav.T))

                        # MESS W/ THE ACCEL WEIGHT

                        # Scale down if gyro readings are large
                        gyroMag = LA.norm(gyroVecAvg)
                        gyroScale = 1

                        accelWeight = 1.0*accelWeight/(1+gyroScale*gyroMag)

                        # Scale down if accelerometers deviate from 1g.
                        accelMag = LA.norm(accelVecAvg)
                        accelThresh = 1 # 0.1

                        accelDev = math.fabs(accelMag - 9.81) > accelThresh

                        if accelDev:
                                accelWeight = 0.
                        else:
                                accelWeight = accelWeight * (1.0 - accelDev/accelThresh)

                        R_error = np.real(psc.EV2DCM(np.append(-accelAxis, (accelWeight*accelAngle)), tol=1E-6, ichk=False))
                        updatedDCM = np.matmul(R_error.T, orientDCM.T)

                else:
                    updatedDCM = orientDCM.T


                self.R = updatedDCM
                q = np.real(psc.DCM2Q(updatedDCM, tol=1E-6, ichk=False))
                self.q = psc.Qnormalize(q)
                self.previousTime = copy(fbk.time)

        def resetCoordinates(self):
        # resets the frame of reference of the pose
                self.R = np.identity(3)

        def removeGyroOffset(self, fbk):

                fbkFixed = copy(fbk)
                fbkFixed.gyroX = fbk.gyroX - self.gyroOffset[0,:]
                fbkFixed.gyroY = fbk.gyroY - self.gyroOffset[1,:]
                fbkFixed.gyroZ = fbk.gyroZ - self.gyroOffset[2,:]

                fbkFixed.accelX = fbk.accelX
                fbkFixed.accelY = fbk.accelY
                fbkFixed.accelZ = fbk.accelZ

                fbk = copy(fbkFixed)

        def removeAccelOffset(self, fbk):
                fbkFixed = copy(fbk)

                fbkFixed.gyroX = fbk.gyroX
                fbkFixed.gyroY = fbk.gyroY
                fbkFixed.gyroZ = fbk.gyroZ

                fbkFixed.accelX = fbk.accelX - self.accelOffset[0,:]
                fbkFixed.accelY = fbk.accelY - self.accelOffset[1,:]
                fbkFixed.accelZ = fbk.accelZ - self.accelOffset[2,:]

                fbk = copy(fbkFixed)

        def quatRotate(self, wX, wY, wZ, q, dt=0.001):
                if type(q) is list:
                        q=np.array(q);
                elif type(q) is tuple:
                        q=np.array(q);
                if len(q.shape)==1:
                        if q.shape[0] % 4 == 0:
                                q.shape=[int(q.size/4),4]
                        else:
                                print ("Wrong number of elements")
                                sys.exit(1)
                if q.shape[1] != 4:
                        q.shape=[int(q.size/4),4]

        # Convert q into scalar top fnormormat
                qModified = np.c_[q[0,3], -q[0,0], -q[0,1], -q[0,2]]
                qNew = np.empty_like(q)
                qNew = qModified + dt*0.5*self.quatMultiply(qModified, np.c_[0, wX, wY, wZ])
                psc.Qnormalize(qNew)

                # return qNew after converting back to original format: scalar at bottom
                return np.c_[-qNew[0,1], -qNew[0,2], -qNew[0,3], qNew[0,0]]
                # return qNew

        def quatMultiply(self,q1,q2):
                if type(q1) is list:
                        q1=np.array(q1);
                elif type(q1) is tuple:
                        q1=np.array(q1);
                if len(q1.shape)==1:
                        if q1.shape[0] % 4 == 0:
                                q1.shape=[int(q1.size/4),4]
                        else:
                                print ("Wrong number of elements")
                                sys.exit(1)
                if q1.shape[1] != 4:
                        q1.shape=[int(q1.size/4),4]

                if type(q2) is list:
                        q2=np.array(q2);
                elif type(q2) is tuple:
                        q2=np.array(q2);
                if len(q2.shape)==1:
                        if q2.shape[0] % 4 == 0:
                                q2.shape=[int(q1.size/4),4]
                        else:
                                print ("Wrong number of elements")
                                sys.exit(1)
                if q2.shape[1] != 4:
                        q2.shape=[int(q1.size/4),4]

                qProduct = np.empty_like(q1)
                qProduct[0,0] =  q1[0,0]*q2[0,0] - q1[0,1]*q2[0,1] - q1[0,2]*q2[0,2] - q1[0,3]*q2[0,3]
                qProduct[0,1] =  q1[0,0]*q2[0,1] + q1[0,1]*q2[0,0] + q1[0,2]*q2[0,3] - q1[0,3]*q2[0,2]
                qProduct[0,2] =  q1[0,0]*q2[0,2] + q1[0,2]*q2[0,0] + q1[0,3]*q2[0,1] - q1[0,1]*q2[0,3]
                qProduct[0,3] =  q1[0,0]*q2[0,3] + q1[0,3]*q2[0,0] + q1[0,1]*q2[0,2] - q1[0,2]*q2[0,1]
                # qProduct[0,0] =  q1[0,3]*q2[0,0] - q1[0,2]*q2[0,1] + q1[0,1]*q2[0,2] + q1[0,0]*q2[0,3]
                # qProduct[0,1] =  q1[0,2]*q2[0,0] + q1[0,3]*q2[0,1] - q1[0,0]*q2[0,2] + q1[0,1]*q2[0,3]
                # qProduct[0,2] =  q1[0,1]*q2[0,0] + q1[0,0]*q2[0,1] + q1[0,3]*q2[0,2] + q1[0,2]*q2[0,3]
                # qProduct[0,3] =  q1[0,0]*q2[0,0] - q1[0,1]*q2[0,1] - q1[0,2]*q2[0,2] + q1[0,3]*q2[0,3]

                return qProduct

def decomposeSO3(rotationMatrix):
        thetaX = math.atan2(rotationMatrix[2,1],rotationMatrix[2,2])
        thetaY = math.atan2(-rotationMatrix[2,0],math.hypot(rotationMatrix[2,1],rotationMatrix[2,2]))
        thetaZ = math.atan2(rotationMatrix[1,0],rotationMatrix[0,0])
        return np.array((thetaX, thetaY, thetaZ))

class feedbackStructure(object):
        def __init__(self, hebigroup):
            self.initialized = False
            while not self.initialized:
                try:
                    self.hebigroup = hebigroup
                    data = hebigroup.getFeedback()
                    while data.getAccelerometers() == []:
                        data = hebigroup.getFeedback()
                    self.accelX = data.getAccelerometers()[0,:,0]
                    self.accelY = data.getAccelerometers()[0,:,1]
                    self.accelZ = data.getAccelerometers()[0,:,2]
                    self.gyroX = data.getGyros()[0,:,0]
                    self.gyroY = data.getGyros()[0,:,1]
                    self.gyroZ = data.getGyros()[0,:,2]
                    self.position = self.hebigroup.getAngles()
                    self.torques = data.getTorques()
                    self.time = time.clock()
                    self.initialized = True
                except:
                    self.initialized = False

        def getNextFeedback(self):
                data = self.hebigroup.getFeedback()
                while data.getAccelerometers() == []:
                    data = hebigroup.getFeedback()
                self.accelX = data.getAccelerometers()[0,:,0]
                self.accelY = data.getAccelerometers()[0,:,1]
                self.accelZ = data.getAccelerometers()[0,:,2]
                self.gyroX = data.getGyros()[0,:,0]
                self.gyroY = data.getGyros()[0,:,1]
                self.gyroZ = data.getGyros()[0,:,2]
                self.position = self.hebigroup.getAngles()
                self.torques = data.getTorques()
                self.time = time.clock()
                        

def dataLogging(fbk):
        fbk.getNextFeedback()
        gyroOffsetX = np.asarray(fbk.gyroX)
        gyroOffsetY = np.asarray(fbk.gyroY)
        gyroOffsetZ = np.asarray(fbk.gyroZ)

        frequency = 100
        numSteps = 5*frequency
        for i in range (0,numSteps):
                fbk.getNextFeedback()
                gyroOffsetX = np.vstack((gyroOffsetX,fbk.gyroX))
                gyroOffsetY = np.vstack((gyroOffsetY,fbk.gyroY))
                gyroOffsetZ = np.vstack((gyroOffsetZ,fbk.gyroZ))
                time.sleep(1/frequency)

        gyroOffset = [np.mean(gyroOffsetX,axis=0), np.mean(gyroOffsetY,axis=0), np.mean(gyroOffsetZ,axis=0)]
        return gyroOffset

def calibrateOffsets(fbk):
        # Snake Monster Torque Modules Definition (intermediate modules of each leg)
        fbk.getNextFeedback()
        gyroOffsetX = np.asarray(fbk.gyroX)
        gyroOffsetY = np.asarray(fbk.gyroY)
        gyroOffsetZ = np.asarray(fbk.gyroZ)
        accelOffsetX = np.asarray(fbk.accelX)
        accelOffsetY = np.asarray(fbk.accelY)
        accelOffsetZ = np.asarray(fbk.accelZ)

        frequency = 100
        numSteps = 5*frequency
        for i in range (0,numSteps):
                fbk.getNextFeedback()
                gyroOffsetX = np.vstack((gyroOffsetX,fbk.gyroX))
                gyroOffsetY = np.vstack((gyroOffsetY,fbk.gyroY))
                gyroOffsetZ = np.vstack((gyroOffsetZ,fbk.gyroZ))
                accelOffsetX = np.vstack((accelOffsetX,fbk.accelX))
                accelOffsetY = np.vstack((accelOffsetY,fbk.accelY))
                accelOffsetZ = np.vstack((accelOffsetZ,fbk.accelZ))
                time.sleep(1/frequency)

        gyroOffset = np.array([np.mean(gyroOffsetX,axis=0), np.mean(gyroOffsetY,axis=0), np.mean(gyroOffsetZ,axis=0)])
        accelOffset = np.array([np.mean(accelOffsetX,axis=0), np.mean(accelOffsetY,axis=0), np.mean(accelOffsetZ,axis=0)])
        print("Gyros and Accelerometers calibrated!")
        np.savetxt("accelOffset.txt", accelOffset)
        np.savetxt("gyroOffset.txt", gyroOffset)
        return (gyroOffset, accelOffset)

def anglesSEAtoU(snakeData, jointAngles):
        # ANGLESSEATOU Changes SEAsnake joint angles as if it were a uSnake
        # Switches the signs of all joint angles and flips the module order to
        # be head to tail.
        # for n in range (0,len(jointAngles)):
        #       jointAngles[n,:] = fliplr(jointAngles[n,:])
        #       jointAngles[n,:] = snakeData.reversals.*jointAngles[n,:]
        return jointAngles

if __name__ == '__main__':
        # Snake Monster Shoulders Definition
        shoulderModules = np.array(NAMES)
        shoulderModules.shape = (6,3)
        shoulderModules = shoulderModules[0:6,0]
        shoulderHebiGroup = HEBI.getGroupFromNames(shoulderModules.tolist())
        fbk = feedbackStructure(shoulderHebiGroup)

        # Calibration of the gyros/accelerometers, snake monster should lay flat on its "belly"
        try:
            gyroOffset = np.loadtxt("gyroOffset.txt")
            accelOffset = np.loadtxt("accelOffset.txt")
        except:
            print('calibrating gyros+accel')
            gyroOffset, accelOffset = calibrateOffsets(fbk)

        # Instantiation of the CF class
        CF = SMComplementaryFilter(accelOffset=accelOffset, gyroOffset=gyroOffset)
        fbk.getNextFeedback()
        CF.update(fbk)
        time.sleep(0.1)
        fbk.getNextFeedback()
        CF.update(fbk)
        time.sleep(0.1)
        fbk.getNextFeedback()
        CF.update(fbk)
        time.sleep(1)

        print('Initial pose:', CF.R)
        cnt = 0

        print("Control loop running...")
        while True:
                # cmd.torque = nan(1,18)
                # Get the current pose
                fbk.getNextFeedback()
                CF.update(fbk)
                currentPose = copy(CF.R)
                if currentPose is not None:
                    currentOrientation = decomposeSO3(currentPose)
                    if cnt == 0:
                        print(currentOrientation)
                time.sleep(0.01)
                cnt = (cnt + 1) % 50
