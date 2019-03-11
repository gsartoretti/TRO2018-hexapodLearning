''' -------------------------------------------------
| Computes forward kinematics for the Snake monster.|
|                                                   |
| example call:                                     |
|                                                   |
| thetas = [0]*18                                   |
| getFootPositions(thetas)                          |
|                                                   |
| Author: Garrison Johnston, Aug. 18. 2017          |
--------------------------------------------------'''

import numpy as np
from math import sin,cos

def createRotation(theta, axis):
	'''
	This function returns a rotation only homogeneous transformation

	INPUTS:
	-------
	Theta = rotation angle
	axis = rotation axis e.g. 'x'

	OUTPUTS:
	--------
	Transformation np matrix
	'''

	if axis == 'x':
		rot = np.matrix([[1, 0,           0,          0],
						 [0, cos(theta), -sin(theta), 0],
						 [0, sin(theta),  cos(theta), 0],
						 [0, 0,           0,          1]])
	elif axis == 'z':
		rot = np.matrix([[cos(theta), -sin(theta), 0, 0],
						 [sin(theta),  cos(theta), 0, 0],
						 [0,           0,          1, 0],
						 [0,           0,          0, 1]])
	return rot

def createTranslation(x, y, z):
	'''
	This function returns a translation only homogeneous transformation

	INPUTS:
	-------
	x, y, z = components of translation

	OUTPUTS:
	--------
	Transformation np matrix
	'''

	transl = np.matrix([[1, 0, 0, x],
					    [0, 1, 0, y],
					    [0, 0, 1, z],
					    [0, 0, 0, 1]])
	return transl

def createTransform(d, theta, r, a):
	'''
	This function returns the DH matrix of a link

	INPUTS:
	-------
	Denavit-Hartenberg Parameters

	d =  offset along previous z to the common normal
	r = length of the common normal
	a = angle about the common normal
	theta = joint angle

	OUTPUTS:
	--------
	Transformation np matrix
	'''
	T = np.matrix([[cos(theta), -sin(theta)*cos(a),  sin(theta)*sin(a), r*cos(theta)],
					[sin(theta), cos(theta)*cos(a), -cos(theta)*sin(a), r*sin(theta)],
					[0,          sin(a),             cos(a),             d],
					[0,          0,                  0,                  1]])
	return T

def legFK(theta):
	'''
	This function returns the workspace position of the snake monster legs.

	INPUTS:
	-------
	theta: 3 vector of joint angles

	OUTPUTS:
	--------
	T: Transformation from endeff to leg base frame
	'''

	## local variables
	module_in = 0.043 #length of a module input
	module_out = 0.021 #length of a module output
	end_eff = [0.05, 0.188, 0] #vector to end effector
	link = 0.064 #first nonmodule link
	d = 0 #the DH parameter d for the module

	##transformations
	T1 = createTransform(0, theta[0], module_out+module_in, -np.pi/2)
	T2 = createTransform(0, theta[1], module_out+link+module_in, 0)
	T3 = createTransform(0, theta[2], module_out+end_eff[0], 0)
	T4 = createTransform(0, -np.pi/2, end_eff[1], 0)
	T = T1*T2*T3*T4
	return T

def getFootPositions(thetas):
	'''
	This function calculates the xyz positions of each leg

	INPUTS:
	-------
	thetas: np array 1x18 of joint angles

	OUTPUTS:
	--------
	foot_pos: 3x6 list of xyz positions of the snake monster feet
	'''

	## translation vectors from body frame to base of legs
	v = [[0.1350, -0.1350,   0.1350,  -0.1350,   0.1350,  -0.1350],
    	 [0.0970,  0.0970,   0.0000,  -0.0000,  -0.0970,   -0.0970],
    	 [0.0000,  0.0000,   0.0000,   0.0000,   0.0000,   0.0000]]

	## foot position
	foot_pos = [0]*6

	## Creating transformations
	i = 0
	while i < 6:
		if (i%2) == 0: # if legs 1, 3, or 5
			rot = createRotation(np.pi, 'x')
			transl = createTranslation(v[0][i], v[1][i], v[2][i])
			T_body2leg = transl*rot
		else:
			rotz = createRotation(np.pi, 'z')
			rotx = createRotation(np.pi, 'x')
			transl = createTranslation(v[0][i], v[1][i], v[2][i])
			T_body2leg = transl*rotz*rotx
		T_leg2foot = legFK(thetas[i*3:i*3+3])
		T = T_body2leg*T_leg2foot
		foot_pos[i] = T[0:3,3].T.tolist()[0]
		i += 1
	return foot_pos




