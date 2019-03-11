'''
Snake Monster stabilization with CPG controls
 Created 24 May 2017
 requires the HEBI API and the Snake Monster HEBI API

 Setting up the Snake Monster

 NOTE: If the modules have been changed, first correct the names and run
calibrateSM.m
'''
from copy import copy
import time
import scipy.signal as signal
import numpy as np
import hebiapi
#import setupfunctions as setup
import tools
import SMCF
from SMCF.SMComplementaryFilter import feedbackStructure, decomposeSO3, calibrateOffsets
import seatools.hexapod as hp
from Functions.CPGgs import CPGgs
from setupfunctions import setupSnakeMonsterShoulderData
from Functions.stabilizationPID import stabilizationPID
from Functions.jacobianAngleCorrection import jacobianAngleCorrection
import configGenerator
import math

import sys
import os
import multiprocessing
import threading
import shutil
import GroupLock
import tensorflow as tf
import tensorflow.contrib.layers as layers


TIME_FACTOR = 1
dt = 1 / 25.  # TIME_FACTOR=2, dt=1/50 and legsAlpha=0.3 gave good results
# tensorboard --logdir=worker_SM:'./train_SM'

# A3C parameters
WALKING = False
#WALKING                 = True
OUTPUT_GRAPH = True
LOG_DIR = './log'
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 256 * (1 + int(WALKING))
GAMMA = 0.88
LR_AC = 1.e-3  # learning rate for actor
GLOBAL_REWARD = []
GLOBAL_EP = 397
START_EP = GLOBAL_EP
N_WORKERS = 1  # should = multiprocessing.cpu_count()
model_path = './model_online'
NsavedModels = 30  # keep ALL models (otherwise, change to 5 or 10)
s_size = 7*6
actionList = [-1., 0., 1.]
a_size = len(actionList)**6
ACTION_COST = -0.02
ZERROR_THRESHOLD = 5.e-3
TH_DEADZONE = 0.015
FILTER_WIN = 2
TORQUE_THRES = 5.0
TRAIN_ON_GROUND = 100
STOP_ON_GROUND = 300
NUM_BUFFERS = N_WORKERS
EPISODE_SAMPLES = 128 * (1 + int(WALKING))
load_model = (GLOBAL_EP != 0)
RESET_TRAINER = False
RNN_USE = True
#RNN_USE                 = False

if WALKING:
    from Functions.Controller import Controller

print('(LR_AC=%.4f) Setting up Snake Monster...' % (LR_AC))

names = SMCF.NAMES

HebiLookup = tools.HebiLookup
shoulders = names[::3]
imu = HebiLookup.getGroupFromNames(shoulders)
snakeMonster = HebiLookup.getGroupFromNames(names)
while imu.getNumModules() != 6 or snakeMonster.getNumModules() != 18:
    print('Found {} modules in shoulder group, {} in robot.'.format(
        imu.getNumModules(), snakeMonster.getNumModules()), end='  \n')
    imu = HebiLookup.getGroupFromNames(shoulders)
    snakeMonster = HebiLookup.getGroupFromNames(names)
print('Found {} modules in shoulder group, {} in robot.'.format(
    imu.getNumModules(), snakeMonster.getNumModules()))
snakeData = setupSnakeMonsterShoulderData()
smk = hp.HexapodKinematics()

fbk_imu = feedbackStructure(imu)
while not fbk_imu.initialized:
    fbk_imu = feedbackStructure(imu)
print('fbk_imu structure created')

fbk_sm = feedbackStructure(snakeMonster)
while not fbk_sm.initialized:
    fbk_sm = feedbackStructure(imu)
print('fbk_sm structure created')

# Reading calibration files (or calibrating)
try:
    gyroOffset = np.loadtxt("gyroOffset.txt")
    accelOffset = np.loadtxt("accelOffset.txt")
    print('Calibration files found and loaded')
except:
    print('calibrating gyros+accel')
    gyroOffset, accelOffset = calibrateOffsets(fbk_imu)

cmd = tools.CommandStruct()
#CF = SMCF.SMComplementaryFilter(accelOffset=accelOffset, gyroOffset=gyroOffset)
# fbk_imu.getNextFeedback()
# CF.update(fbk_imu)
# time.sleep(0.02)
#pose = []
# while pose is None or not list(pose):
# fbk_imu.getNextFeedback()
# CF.update(fbk_imu)
#pose = copy(CF.R)


print('Setup complete!')

# Initialize Variables

T = 1e4
nIter = round(T / dt)
#prevReward = [0,0,0,0,0,0]

forward = np.ones((1, 6))
backward = -1 * np.ones((1, 6))
leftturn = np.array([[1, -1, 1, -1, 1, -1]])
rightturn = np.array([[-1, 1, -1, 1, -1, 1]])

shoulders1 = list(range(0, 18, 3))  # joint IDs of the shoulders
shoulders2 = list(range(1, 18, 3))  # joint IDs of the second shoulder joints
elbows = list(range(2, 18, 3))  # joint IDs of the elbow joints

shoulderVec = np.matrix([[0.1350, -0.1350, 0.1350, -0.1350, 0.1350, -0.1350], [0.0970, 0.0970,
                                                                               0.0000, -0.0000, -0.0970, -0.0970], [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

cpg0 = {
    'initLength': int(WALKING) * 251 - 1,
    'bodyHeight': 0.16,
    'r': 0.10,
    'direction': forward,
    # SUPERELLIPSE
    #'w': 60,
    #'a': 45 * np.ones((1,6)),
    #'b': 3.75 * np.ones((1,6)),
    #'x': 5.0 * np.array([[1, -1, 1, -1, 1, -1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
    #'y': 20.0 * np.array([[1, 1, 1, 1, 1, 1]]+[[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
    # /SUPERELLIPSE
    'w': 7,
    'a': 0.03 * np.ones((1, 6)),
    'b': 1.00 * np.ones((1, 6)),
    'mu': np.array([0.0412, 0.0412, 0.0882, 0.0882, 0.0412, 0.0412]),
    'x': float(WALKING) * np.array([[1, -1, 1, -1, 1, -1]] + [[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
    'y': float(WALKING) * np.array([[1, 1, 1, 1, 1, 1]] + [[0, 0, 0, 0, 0, 0] for i in range(nIter)]),
    's1Off': np.pi / 3,
    's2Off': np.pi / 16,
    't3Str': 0.0,
    'stabilize': True,
    #'isStance': np.zeros((1,6)),
    'dx': np.zeros((3, 6)),
    'legs': np.zeros((1, 18)),
    'legsAlpha': 0.3,
    'move': WALKING,
    'smk': smk,
    'pose': np.identity(3),
    'zErr': 0.0,
    'zHistory': np.ones((1, 10)),
    'zHistoryCnt': 0,
    'dTheta2': np.array([0., 0., 0., 0., 0., 0.]),
    'dTheta2GS': np.array([0., 0., 0., 0., 0., 0.]),
    'theta2': np.array([0., 0., 0., 0., 0., 0.]),
    'shoulderZError': np.array([0., 0., 0., 0., 0., 0.]),
    'groundD': 0,
    'groundNorm': 0,
    'torques': np.zeros((3, 6)),
    'on_ground': np.array([0., 0., 0., 0., 0., 0.]),
    'jacobianTorques': np.zeros((3, 6)),
    'done': False,
}
cpg0['zHistory'] = cpg0['zHistory'] * cpg0['bodyHeight']
cpg0['theta_min'] = -np.pi / 2 - cpg0['s2Off']
cpg0['theta_max'] = np.pi / 2 - cpg0['s2Off']

# Walk the Snake Monster

if WALKING:
    joy = Controller()
    cpgJoy = True
else:
    configCreator = configGenerator.NewConfig()

print('Finding initial stance...')

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x):
    return signal.lfilter([1], [1, -GAMMA], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# Class Actor and Critic network
class ACNet(object):
    def __init__(self, scope):
        with tf.variable_scope(str(scope) + '/qvalues'):
            self.is_Train = True
            # The input size may require more work to fit the interface.
            self.inputs = tf.placeholder(
                shape=[None, s_size], dtype=tf.float32)
            if RNN_USE:
                self.policy, self.value, self.state_out, self.state_in, self.state_init, self.too_high, self.on_ground = self._build_net(
                    self.inputs)
            else:
                self.policy, self.value, self.too_high, self.on_ground = self._build_net(
                    self.inputs)
        if(scope != GLOBAL_NET_SCOPE):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, a_size, dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.target_too_high = tf.placeholder(
                tf.float32, [None], 'target_too_high')
            self.target_on_ground = tf.placeholder(
                tf.float32, [None,6], 'target_on_ground')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])
            self.decay_on_ground = tf.placeholder(tf.float32)
            self.train_value = tf.placeholder(tf.float32, [None])

            # Loss Functions
            self.value_loss = 0.5 * tf.reduce_sum(self.train_value * tf.square(
                self.target_v - tf.reshape(self.value, shape=[-1])))
            self.too_high_loss = - tf.reduce_sum(self.target_too_high * tf.log(tf.clip_by_value(tf.reshape(self.too_high, shape=[-1]), 1e-10, 1.0)) + (
                1 - self.target_too_high) * tf.log(tf.clip_by_value(1 - tf.reshape(self.too_high, shape=[-1]), 1e-10, 1.0)))
            self.on_ground_loss = - self.decay_on_ground * tf.reduce_sum(tf.log(tf.clip_by_value(self.on_ground,1e-10,1.0)) * \
                self.target_on_ground+tf.log(tf.clip_by_value(1-self.on_ground,1e-10,1.0)) * (1-self.target_on_ground))

            # something to encourage exploration
            self.entropy = - \
                tf.reduce_sum(
                    self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.)))

            self.policy_loss = - \
                tf.reduce_sum(tf.log(tf.clip_by_value(
                    self.responsible_outputs, 1e-10, 1.)) * self.advantages)
            self.loss = 0.5 * self.value_loss + 0.5 * self.too_high_loss + 0.5 * self.on_ground_loss + \
                0.5 * self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope + '/qvalues')
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE + '/qvalues')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
        print("Hello World... From LEG " + str(scope))     # :)

    def _build_net(self, inputs):

        h0 = layers.fully_connected(
            inputs=inputs, num_outputs=128, activation_fn=None)
        conv1 = tf.layers.conv1d(inputs=tf.reshape(
            h0, shape=(-1, 128, 1)), filters=1, kernel_size=1, strides=1, use_bias=False)

        h1 = layers.fully_connected(inputs=inputs, num_outputs=128)
        h2 = layers.fully_connected(inputs=h1,     num_outputs=128)
        h3 = layers.fully_connected(inputs=h2,     num_outputs=128)
        h4 = layers.fully_connected(
            inputs=h3,     num_outputs=128, activation_fn=None)

        res1 = tf.nn.relu(layers.flatten(conv1) + h4)
        h5 = layers.fully_connected(inputs=res1,  num_outputs=128)

        if RNN_USE:
            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(h5, [0])
            step_size = tf.shape(inputs)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 128])
        else:
            rnn_out = h5

        policy = layers.fully_connected(inputs=rnn_out, num_outputs=a_size, biases_initializer=None,
                                        activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(1.0))
        value = layers.fully_connected(inputs=rnn_out, num_outputs=1, biases_initializer=None,
                                       activation_fn=None, weights_initializer=normalized_columns_initializer(1.0))
        too_high = layers.fully_connected(inputs=rnn_out, num_outputs=1, biases_initializer=None,
                                          activation_fn=tf.sigmoid, weights_initializer=normalized_columns_initializer(1.0))
        on_ground = layers.fully_connected(inputs=rnn_out, num_outputs=6, biases_initializer=None,
                                           activation_fn=tf.sigmoid, weights_initializer=normalized_columns_initializer(1.0))

        if RNN_USE:
            return policy, value, state_out, state_in, state_init, too_high, on_ground
        else:
            return policy, value, too_high, on_ground

# Worker class


class Worker(object):
    def __init__(self, name, globalAC, workerID, groupNumber, groupLock):
        self.groupNumber = groupNumber
        self.groupLock = groupLock
        self.name = name
        self.local_AC = ACNet(name)
        self.model_path = model_path
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.name))
        self.shouldDie = False
        self.workerID = workerID
        self.pull_global = update_target_graph(
            GLOBAL_NET_SCOPE + '/qvalues', self.name + '/qvalues')

    def work(self):
        global GLOBAL_REWARD, GLOBAL_EP, EPISODE_FINISHED
        episodeStep = 0 #episodeReward = 0
        i_buf, episode_buffers = -1, [[] for i in range(NUM_BUFFERS)]
        s1_values = [0 for i in range(NUM_BUFFERS)]
        actorLoss, criticLoss = [], []

        while True:
            self.groupLock.acquire(self.groupNumber, self.name)

            if self.shouldDie:
                # release lock and stop thread
                print('Thread %s exiting now...' % (self.name))
                self.groupLock.release(self.groupNumber, self.name)
                return

            if episodeStep == 0:
                SESS.run(self.pull_global)
                if RNN_USE:
                    rnn_s = self.local_AC.state_init
                else:
                    rnn_s = 0

                i_buf = (i_buf + 1) % NUM_BUFFERS
                episode_buffers[i_buf] = []
                oldZErr = too_high_wrongs = on_ground_wrongs = 0
                train_value = 1

                s, a, v, rnn_s, pt_h, t_h, po_g, o_g = self.getStateAction(rnn_s)
#                if int(pt_h > 0.5) != t_h and t_h != 0.5:
#                    a = np.random.choice(range(a_size))
#                    too_high_wrongs += 1
#                    train_value = 0
#                elif (np.logical_and(np.asarray(po_g > 0.5, dtype=int) != o_g, o_g != 0.5)).any():
#                    a = np.random.choice(range(a_size))
#                    on_ground_wrongs += 1
#                    train_value = 0

                self.performAction(a)

                Zerr = self.getZerror()
#                doneTag = False
                episodetmp = [s, a, v, t_h, *o_g, train_value, Zerr]
                episodeStep += 1
            else:
                if not cpg['done']:
                    # s1: current state, s: previous state
                    s1, a, v, rnn_s, pt_h, t_h, po_g, o_g = self.getStateAction(rnn_s)
                    #r, oldZErr                            = self.getReward(oldZErr, dt*TIME_FACTOR)
                    Zerr = self.getZerror()
                    train_value = 1

                    # If state quality is good enough, stop episode
#                    if int(pt_h > 0.5) != t_h and t_h != 0.5:
#                        a = np.random.choice(range(a_size))
#                        too_high_wrongs += 1
#                        train_value = 0
#                    elif (np.logical_and(np.asarray(po_g > 0.5, dtype=int) != o_g, o_g != 0.5)).any():
#                        a = np.random.choice(range(a_size))
#                        on_ground_wrongs += 1
#                        train_value = 0

                    # episodetmp.append(r)
                    episode_buffers[i_buf].append(episodetmp)
                    episodetmp = [s, a, v, t_h, *o_g, train_value, Zerr]

                    self.performAction(a)
                    #episodeReward              += r
                    episodeStep += 1

                # update global and assign to local net
                if episodeStep == UPDATE_GLOBAL_ITER or cpg['done']:
                    print('({})Worker {} updating global net'.format(
                        episodeStep, self.workerID))

                    # TD Stuffs
                    if RNN_USE:
                        s1_values[i_buf] = SESS.run(self.local_AC.value, {self.local_AC.inputs: np.reshape(np.array(s1), [1, s_size]),
                                                                          self.local_AC.state_in[0]: rnn_s[0],
                                                                          self.local_AC.state_in[1]: rnn_s[1]})[0, 0]
                    else:
                        s1_values[i_buf] = SESS.run(self.local_AC.value, {
                                                    self.local_AC.inputs: np.reshape(np.array(s1), [1, s_size])})[0, 0]

                    i_rand = np.random.randint(
                        min(int((GLOBAL_EP - START_EP) / N_WORKERS) + 1, NUM_BUFFERS))
#                   episodeReward, v_l, p_l, e_l, g_n, v_n, th_l, og_l = self.train(
#                       episode_buffers[i_rand], s1_values[i_rand], Zerr)
#                   SESS.run(self.pull_global)

#                   actorLoss.append(p_l)
#                   criticLoss.append(v_l)

                    # Reset episode
#                    GLOBAL_REWARD.append(episodeReward)
                    episodeStep, GLOBAL_EP = 0, GLOBAL_EP + 1
#                    EPISODE_FINISHED = True

                    # Tensorboard stuffs
#                    self.tensorboardPlot(
#                        criticLoss, actorLoss, th_l, too_high_wrongs, og_l, on_ground_wrongs)
#                    if GLOBAL_EP % 1 == 0:
#                        # last thread saves model
#                        print("Saving Model (GLOBAL_EP={})...".format(
#                            GLOBAL_EP), end='\n')
#                        saver.save(SESS, model_path + '/model-' +
#                                   str(GLOBAL_EP) + '.cptk')

#            if episodeStep >= cpg['zHistory'].shape[1]:
#                cpg['done'] = (abs(Zerr) < ZERROR_THRESHOLD)

            self.groupLock.release(self.groupNumber, self.name)

    def performAction(self, action):
        start = action

        for i in range(5,-1,-1):
            a = math.floor(start / (len(actionList)**i))
            start = start % (len(actionList)**i)
            cpg['dTheta2'][i] = actionList[a]

    def medfilter(self, array, win_size):
        filtered_array = array.copy()

        for i in range(len(array)):
            filtered_array[i] = np.mean(
                array[max(0, i - win_size):min(len(array) - 1, i + win_size + 1)])

        return filtered_array

    def train(self, rollout, bootstrap_value, Zerr1):
        global GLOBAL_EP
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        values = rollout[:, 2]
        too_highs = rollout[:, 3]
        on_grounds = rollout[:, 4:-2]
        train_values = rollout[:, -2]
        #rewards         = rollout[:,-1]
        Zerrors = np.array(list(rollout[:, -1]) + [Zerr1])

        # Reward filtering
        filtered_err = self.medfilter(Zerrors, FILTER_WIN)
        rewards = ACTION_COST + \
            np.clip((filtered_err[:-1] - filtered_err[1:]) / dt, -5.0, 5.0)
        # cancel rewards for legs in the air (no responsibility)
        episodeReward = np.sum(rewards)

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + GAMMA * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages)
        decay_on_ground = 1.0
        if GLOBAL_EP > TRAIN_ON_GROUND:
            decay_on_ground = np.exp(- (GLOBAL_EP - TRAIN_ON_GROUND) / 10.)
        elif GLOBAL_EP > STOP_ON_GROUND:
            decay_on_ground = 0.0

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(
            advantages.shape[0], size=(num_samples,), replace=False))

        if RNN_USE:
            rnn_state = self.local_AC.state_init
            feed_dict = {self.local_AC.target_v: np.stack(discounted_rewards[sampleInd]),
                         self.local_AC.inputs: np.reshape(np.stack(observations[sampleInd]), [-1,s_size]),
                         self.local_AC.actions: actions[sampleInd],
                         self.local_AC.target_too_high: too_highs[sampleInd],
                         self.local_AC.target_on_ground: on_grounds[sampleInd],
                         self.local_AC.train_value: train_values[sampleInd],
                         self.local_AC.advantages: advantages[sampleInd],
                         self.local_AC.decay_on_ground: decay_on_ground,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {self.local_AC.target_v: np.stack(discounted_rewards[sampleInd]),
                         self.local_AC.inputs: np.reshape(np.stack(observations[sampleInd]), [-1,s_size]),
                         self.local_AC.actions: actions[sampleInd],
                         self.local_AC.target_too_high: too_highs[sampleInd],
                         self.local_AC.target_on_ground: on_grounds[sampleInd],
                         self.local_AC.train_value: train_values[sampleInd],
                         self.local_AC.advantages: advantages[sampleInd],
                         self.local_AC.decay_on_ground: decay_on_ground}

        v_l, p_l, e_l, g_n, v_n, th_l, og_l, _ = SESS.run([self.local_AC.value_loss,
                                                           self.local_AC.policy_loss,
                                                           self.local_AC.entropy,
                                                           self.local_AC.grad_norms,
                                                           self.local_AC.var_norms,
                                                           self.local_AC.too_high_loss,
                                                           self.local_AC.on_ground_loss,
                                                           self.local_AC.apply_grads],
                                                          feed_dict=feed_dict)

        SESS.run(self.pull_global)
        return episodeReward, v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, th_l, og_l

    #def getReward(self, oldZErr, dt): # unused now that we filter out reward at training time
        #zError = getZerror()
        #reward = np.clip((oldZErr - zError) / dt, -
                        #5.0, 5.0)  # minus sign needed

        ## if not self.workerID:
        ##print(verticalError, dZErr, on_ground, reward)
        #return reward, zError

    def getZerror(self):
        # Leg-specific vertical error
        heightFac, verticalError = 10.0, sum([abs(err - cpg['zErr']) for err in cpg['shoulderZError']]) / 6.
        return (heightFac * verticalError)**2

    def tensorboardPlot(self, criticLoss, actorLoss, too_high_loss, too_high_wrongs, on_ground_loss, on_ground_wrongs):
        global GLOBAL_REWARD, GLOBAL_EP
        if GLOBAL_EP >= 2:
            meanReward = np.mean(GLOBAL_REWARD[-2:])
            meanCriticLoss = np.max(np.mean(criticLoss[-2:]))
            meanActorLoss = np.max(np.mean(actorLoss[-2:]))
            too_high_accuracy = (UPDATE_GLOBAL_ITER -
                                 too_high_wrongs) / float(UPDATE_GLOBAL_ITER)
            on_ground_accuracy = (UPDATE_GLOBAL_ITER-on_ground_wrongs)/float(UPDATE_GLOBAL_ITER)
            summary = tf.Summary()
            summary.value.add(tag='Perf/Reward',
                              simple_value=float(meanReward))
            summary.value.add(tag='Losses/Value Loss',
                              simple_value=float(meanCriticLoss))
            summary.value.add(tag='Losses/Policy Loss',
                              simple_value=float(meanActorLoss))
            summary.value.add(tag='Losses/Too_High Body Loss',
                              simple_value=float(too_high_loss))
            summary.value.add(tag='Losses/Too_High Body Accuracy',
                              simple_value=float(too_high_accuracy))
            summary.value.add(tag='Losses/On_Ground Loss', simple_value=float(on_ground_loss))
            summary.value.add(tag='Losses/On_Ground Accuracy', simple_value=float(on_ground_accuracy))
            self.summary_writer.add_summary(summary, GLOBAL_EP)
            self.summary_writer.flush()

    def getStateAction(self, init_state):
        # State definition
        state = [
            2 / np.pi * cpg['legs'][0, shoulders1],
            2 / np.pi * cpg['legs'][0, shoulders2],
            2 / np.pi * cpg['legs'][0, elbows],
            10 * (cpg['shoulderZError'] - cpg['zErr']),
            0.2 * cpg['torques'][0],
            0.2 * cpg['torques'][1],
            0.2 * cpg['torques'][2]
        ]
        # Action selection
        if RNN_USE:
            a_dist, value, rnn_state, pred_too_high, pred_on_ground = SESS.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out, self.local_AC.too_high, self.local_AC.on_ground], feed_dict={
                self.local_AC.inputs: np.reshape(np.array(state), [1, s_size]),
                self.local_AC.state_in[0]: init_state[0],
                self.local_AC.state_in[1]: init_state[1]})
        else:
            a_dist, value, pred_too_high = SESS.run([self.local_AC.policy, self.local_AC.value, self.local_AC.too_high, self.local_AC.on_ground], feed_dict={
                                                    self.local_AC.inputs: np.reshape(np.array(state), [1, s_size])})
            rnn_state = 0
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        verticalError = cpg['zErr'] # note: cpg['zErr'] = cpg['bodyHeight'] - np.median(cpg['zHistory'])
        too_high = int(verticalError > 0)
        if abs(verticalError) < TH_DEADZONE:
            too_high = 0.5

        on_ground = np.asarray(cpg['jacobianTorques'][2,:] > 0., dtype=int) #or cpg['on_ground'][self.workerID]
        on_ground[cpg['jacobianTorques'][2,:] < TORQUE_THRES] = 0.5

        # if not self.workerID:
        # , np.var(state)) The state may be way too big to print, but we may be able to do something about this
        #print(a_dist, a, pred_too_high, too_high)
        # 
        return state, a, value, rnn_state, pred_too_high, too_high, pred_on_ground, on_ground


if __name__ == "__main__":
    # begin Tensorflow initialization:

    # initialize groupLock
    groups = [['main'],['SM']]
    groupLock = GroupLock.GroupLock(groups)

    groupLock.acquire(0,'main')#lock here
    SESS = tf.Session()

    # initilize optimizers and workers
    with tf.device("/cpu:0"):  # GPU is faster...
        trainer = tf.contrib.opt.NadamOptimizer(LR_AC, use_locking=True)
        #trainer = tf.train.AdadeltaOptimizer(LR_AC)
        # Global net: we only need its parameters
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []  # creat workers
        for i in range(N_WORKERS):
            workerName = 'SM'
            workers.append(Worker(workerName, GLOBAL_AC, i, 1, groupLock))
        saver = tf.train.Saver(max_to_keep=NsavedModels)

    # load old model or Initialize new vairables
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(SESS, ckpt.model_checkpoint_path)
        if RESET_TRAINER:
            trainer = tf.contrib.opt.NadamOptimizer(LR_AC, use_locking=True)
    else:
        SESS.run(tf.global_variables_initializer())

    # output tensorflow graph
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    # start threads
    worker_threads = []
    for worker in workers:
        th = threading.Thread(target=worker.work)
        th.start()
        worker_threads.append(th)
    # end Tensorflow initialization

    cpg = {}
    while True:
        # Reset snakemonster cpg sutrcture
        cpg = {}
        for key in cpg0.keys():
            cpg[key] = copy(cpg0[key])

        # Find and command initial pose
        if not WALKING:
            cpg = configCreator.cpgConfig(cpg, randHeight=0.)
            snakeMonster.setAngles(cpg['legs'][0])
        snakeMonster.setCommandLifetime(0)

        # Wait for this to be initialized
        # time.sleep(2)
        input('Ready for next Episode ?  ')

        # Reset Complementary Filter
        CF = SMCF.SMComplementaryFilter(
            accelOffset=accelOffset, gyroOffset=gyroOffset)
        fbk_imu.getNextFeedback()
        CF.update(fbk_imu)
        time.sleep(0.02)
        pose = []
        while pose is None or not list(pose):
            fbk_imu.getNextFeedback()
            CF.update(fbk_imu)
            pose = copy(CF.R)

        print('Starting new episode')
        EPISODE_FINISHED = False

        for t in range(nIter):
            if ((t - 1) % TIME_FACTOR) == 0:
                tStart = time.perf_counter()

            if t == cpg['initLength']:
                print('Snake Monster is ready!')
                print('Begin walking')

            # Get pose/gravity vector
            fbk_imu.getNextFeedback()
            CF.update(fbk_imu)

            if t >= cpg['initLength']:
                cpg['pose'] = copy(CF.R)
                shoulderWorldVec = np.matmul(cpg['pose'], shoulderVec)
                cpg['shoulderZError'] = np.asarray(shoulderWorldVec[2, :])[0]

                # Get Snake Monster feedback
                fbk_sm.getNextFeedback()

                # Correct leg angles based on fbk_sm structure
                cpg['legs'] = cpg['legsAlpha'] * fbk_sm.position + \
                    (1 - cpg['legsAlpha']) * cpg['legs']
                cpg['torques'] = np.reshape(fbk_sm.torques, [3, 6])

                # Average torques using Jacobian
                J = cpg['smk'].getLegJacobians(cpg['legs'])
                for leg in range(6):
                    cpg['jacobianTorques'][:, leg] = np.matmul(
                        cpg['pose'].T, np.linalg.lstsq(J[leg, :3, :].T, cpg['torques'][:, leg])[0])
                # print(cpg['jacobianTorques'][2,:])

                cpg = stabilizationPID(cpg)
#                cpg['dTheta2GS'] = jacobianAngleCorrection(cpg)[1]
                if (t % TIME_FACTOR) == 0:
                    groupLock.release(0, 'main')  # unlock here
                    # Threads do their business...
                    groupLock.acquire(0, 'main')  # lock here

            # Check if stable snake monster, stop episode if so
            if EPISODE_FINISHED:
                break
            else:
                # Update CPG
                cpg = CPGgs(cpg, t, dt)

                # Command
                cmd.position = cpg['legs']
                snakeMonster.setAngles(cmd.position[0])

                if (t % TIME_FACTOR) == 0 and t > 0:
                    loopTime = time.perf_counter() - tStart
                    #print(loopTime, TIME_FACTOR*dt, max(0.0,(TIME_FACTOR*dt)-loopTime))
                    time.sleep(max(0.0, (TIME_FACTOR * dt) - loopTime))

    print("Saving Model (GLOBAL_EP={})...".format(GLOBAL_EP), end='\n')
    saver.save(SESS, model_path + '/model-' + str(GLOBAL_EP) + '.cptk')
    print('OK\nTrying to exit code cleanly...')
    if WALKING:
        joy.running = False
    for worker in workers:
        print('Sending kill signal to thread {}...'.format(worker.name))
        worker.shouldDie = True
    time.sleep(0.1)
    print('Letting Threads stop...')
    groupLock.release(0, 'main')  # unlock here and let threads stop cleanly
    sleep(1)
    print('Threads stopped, you should Ctrl-C...')
