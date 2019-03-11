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
from SMCF.SMComplementaryFilter import feedbackStructure,decomposeSO3,calibrateOffsets
import seatools.hexapod as hp
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs
from setupfunctions import setupSnakeMonsterShoulderData
from Functions.stabilizationPID import stabilizationPID
from Functions.jacobianAngleCorrection import jacobianAngleCorrection
import configGenerator

import sys, os
import multiprocessing
import threading
import shutil
import GroupLock
import tensorflow as tf
import tensorflow.contrib.layers as layers


TIME_FACTOR = 1
## tensorboard --logdir=worker_0:'./train_W_0',worker_1:'./train_W_1',worker_2:'./train_W_2',worker_3:'./train_W_3',worker_4:'./train_W_4',worker_5:'./train_W_5'

## A3C parameters
OUTPUT_GRAPH            = True
LOG_DIR                 = './log'
GLOBAL_NET_SCOPE        = 'Global_Net'
UPDATE_GLOBAL_ITER      = round(100 / TIME_FACTOR)
EPISODE_LEARNING        = 5
GAMMA                   = 0.95 ** TIME_FACTOR
LR_AC                   = 1/6.e3
GLOBAL_REWARD           = []
GLOBAL_EP               = 0
N_WORKERS               = 6 #should = multiprocessing.cpu_count()
model_path              = './model_online'
NsavedModels            = 0 # keep ALL models (otherwise, change to 5 or 10)
s_size                  = 33
#actionList              = [-1., 0.5, 0.2, 0., 0.2, 0.5, 1.]
actionList              = [-1., 0., 1.]
a_size                  = len(actionList)
EPS_FACT                = 2
load_model              = True
#load_model              = False
l_size                  = max(s_size, a_size) # layers' size

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
    gyroOffset, accelOffset = calibrateOffsets(fbk_imu)#setup.setupSnakeMonster()

cmd = tools.CommandStruct()
CF = SMCF.SMComplementaryFilter(accelOffset=accelOffset, gyroOffset=gyroOffset)
fbk_imu.getNextFeedback()
CF.update(fbk_imu)
time.sleep(0.02)
pose = []
while pose is None or not list(pose):
    fbk_imu.getNextFeedback()
    CF.update(fbk_imu)
    pose = copy(CF.R)


print('Setup complete!')

## Initialize Variables

T = 1e5
dt = 0.02
nIter = round(T / dt)
prevReward = [0,0,0,0,0,0]

forward = np.ones((1,6))
backward = -1 * np.ones((1,6))
leftturn = np.array([[1, -1, 1, -1, 1, -1]])
rightturn = np.array([[-1, 1, -1, 1, -1, 1]])

shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

cpg0 = {
    'initLength': -1, # was 250, -1 needed for random initial poses given by cpgConfig code
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
    'legsAlpha': 0.75,
    'move': False,
    'smk': smk,
    'pose': np.identity(3),
    'zErr': 0.0,
    'zHistory': np.ones((1,10)),
    'zHistoryCnt': 0,
    'dTheta2': np.array([0.,0.,0.,0.,0.,0.]),
    'theta2': np.array([0.,0.,0.,0.,0.,0.]),
    'groundD': 0,
    'groundNorm': 0,
    'torques': np.zeros((3,6)),
    'jacobianTorques': np.zeros((3,6)),
}
cpg0['zHistory'] = cpg0['zHistory'] * cpg0['bodyHeight']
cpg0['theta_min'] = -np.pi/2 - cpg0['s2Off']
cpg0['theta_max'] = np.pi/2 - cpg0['s2Off']

## Walk the Snake Monster

joy = Controller()
cpgJoy = True

print('Finding initial stance...')

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x):
    return signal.lfilter([1], [1, -GAMMA], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer




## Class Actor and Critic network
class ACNet(object):
    def __init__(self, scope):
        with tf.variable_scope(str(scope)+'/qvalues'):
            self.is_Train = True
            #The input size may require more work to fit the interface.
            self.inputs = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
            self.policy, self.value = self._build_net(self.inputs)
        if(scope!=GLOBAL_NET_SCOPE):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

            # Loss Functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))

            # something to encourage exploration
            self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.)))

            self.policy_loss = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs, 1e-10, 1.)) * self.advantages)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvalues')
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
        print("Hello World... From LEG "+str(scope))     # :)

    def _build_net(self,inputs):

        h1 = layers.fully_connected(inputs=inputs,  num_outputs=48)
        h2 = layers.fully_connected(inputs=h1,  num_outputs=48)
        h3 = layers.fully_connected(inputs=h2,  num_outputs=48)
        h4 = layers.fully_connected(inputs=h3,  num_outputs=48)

        policy = layers.fully_connected(inputs=h4, num_outputs=a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        value = layers.fully_connected(inputs=h4, num_outputs=1, biases_initializer=None, activation_fn=None)

        return policy, value

##Worker class
class Worker(object):
    def __init__(self, name, globalAC, workerID, cpg, groupNumber, groupLock):
        self.groupNumber    = groupNumber
        self.groupLock      = groupLock
        self.name           = name
        self.local_AC       = ACNet(name)
        self.model_path     = model_path
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.name))
        self.shouldDie      = False
        self.workerID       = workerID
        self.cpg            = cpg
        self.episode_buffer = episodeBuffer
        self.lastUpdate     = 0
        self.actorLoss      = []
        self.criticLoss     = []
        self.pull_global    = update_target_graph(GLOBAL_NET_SCOPE+'/qvalues', self.name+'/qvalues')

    def work(self):
        global GLOBAL_REWARD, GLOBAL_EP
        episodeReward = episodeStep = 0
        self.episode_buffer['worker%d'%self.workerID] = []

        while True:
            self.groupLock.acquire(self.groupNumber,self.name)

            if self.shouldDie:
                # release lock and stop thread
                print('Thread %s exiting now...' % (self.name))
                self.groupLock.release(self.groupNumber,self.name)
                return

            if episodeStep == 0:
                SESS.run(self.pull_global)
                self.explorQ   = 0
                s, a, v        = self.getStateAction()
                self.performAction(a)
                episodetmp     = [s,a,v]
                episodeStep   += 1
            else:
                #s1: current state, s: previous state
                a_old                       = a
                s1, a, v             = self.getStateAction()
                r                           = self.getReward(a_old)
                episodetmp.append(r)
                self.episode_buffer['worker%d'%self.workerID].append(episodetmp)
                episodetmp=[s,a,v]

                self.performAction(a)
                episodeReward              += r
                episodeStep                += 1

                # update global and assign to local net
                if episodeStep % UPDATE_GLOBAL_ITER == 0:
                    GLOBAL_EP              += 1
                    GLOBAL_REWARD.append(episodeReward)
                    episodeReward           = 0

                    if int((GLOBAL_EP-1)/N_WORKERS) % EPISODE_LEARNING == 0 and GLOBAL_EP >= EPISODE_LEARNING * N_WORKERS:
                        numEpisode     = EPISODE_LEARNING # number of training episodes to sample from experience buffer
                        end            = len(self.episode_buffer['worker%d'%self.workerID])
                        start          = self.lastUpdate
                        print(start, end)
                        updatingThread = threading.Thread(target = self.updateTraining,
                                                        args=(GLOBAL_EP, numEpisode, start, end))
                        updatingThread.start()



            self.groupLock.release(self.groupNumber,self.name)

    def performAction(self, action):
        self.cpg['dTheta2'][self.workerID] = actionList[action]

    def getReward(self, a_old):
        ## New idea ? 1 - penalties
        poseFac, heightFac = 5, 20.0
        dirVec = np.array([[0,0,1]])
        nBody = np.matmul(self.cpg['pose'], dirVec.T).T

        zError    = (heightFac * cpg['zErr'])**2
        poseError = poseFac * (1.0 - np.dot(nBody[0], dirVec[0]))
        alpha = 0.5

        dxFact, dxNorm = 10.0, abs(self.cpg['dx'][0,self.workerID])
        dxPenalty = 1 + dxFact * dxNorm

        ## actions costs
        #a2Fact, action = 0.2, actionList[a_old]
        #actionsCost = a2Fact * abs(action)

        penalty = dxPenalty * (alpha * poseError + (1.0-alpha) * zError) #+ actionsCost

        reward = 1.0 - 2 * penalty
        if not self.workerID:
            #print(dxPenalty, alpha, poseError, zError, dxPenalty, actionsCost, reward)
            print(poseError, zError, dxPenalty, reward)
        return reward

    def tensorboardPlot(self):
        global GLOBAL_REWARD, GLOBAL_EP
        if GLOBAL_EP >= 12:
            meanReward = np.mean(GLOBAL_REWARD[-12:])
            meanCriticLoss = np.max(np.mean(self.criticLoss[-12:]))
            meanActorLoss = np.max(np.mean(self.actorLoss[-12:]))
            summary=tf.Summary()
            summary.value.add(tag='Perf/Reward', simple_value=float(meanReward))
            summary.value.add(tag='Losses/Value Loss', simple_value=float(meanCriticLoss))
            summary.value.add(tag='Losses/Policy Loss', simple_value=float(meanActorLoss))
            self.summary_writer.add_summary(summary, GLOBAL_EP)
            self.summary_writer.flush()

    def getStateAction(self):
        ## State definition
        state           = [
                            self.cpg['legs'][0,shoulders1][self.workerID],
                            self.cpg['legs'][0,shoulders2][self.workerID],
                            self.cpg['legs'][0,elbows][self.workerID],
                            self.cpg['dx'][0,self.workerID],
                            self.cpg['dx'][1,self.workerID],
                            self.cpg['dx'][2,self.workerID],
                            np.sin(self.cpg['legs'][0,shoulders2][self.workerID]),
                            np.sin(self.cpg['legs'][0,shoulders2][self.workerID]),
                            np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            np.cos(self.cpg['legs'][0,shoulders2][self.workerID]),
                            np.cos(self.cpg['legs'][0,shoulders2][self.workerID]),
                            np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            self.cpg['theta2'][self.workerID],
                            self.cpg['torques'][0,self.workerID],
                            self.cpg['torques'][1,self.workerID],
                            self.cpg['torques'][2,self.workerID]
                          ]
        # Action selection
        a_dist, value = SESS.run([self.local_AC.policy,self.local_AC.value], feed_dict={self.local_AC.inputs:np.reshape(np.array(state),[1,s_size])})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist==a)

        if not self.workerID:
            print(a_dist)
        return state, a, value

    def train(self, rollout, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        values = rollout[:,2]
        rewards = rollout[:,3]
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + GAMMA * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages)
        feed_dict = {self.local_AC.target_v:np.stack(discounted_rewards),
        self.local_AC.inputs:np.stack(observations),
        self.local_AC.actions:actions,
        self.local_AC.advantages:advantages}

        v_l,p_l,e_l,g_n,v_n,_ = SESS.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def updateTraining(self, GLOBAL_EP, numEpisode, start, end):
        print('\n\nWorker {} updating global net\n\n'.format(self.workerID))
        # TD Stuffs
        for sample in range(numEpisode):
            startpoint         = np.random.randint(start, end - UPDATE_GLOBAL_ITER - 1)
            s1                 = self.episode_buffer['worker%d'%self.workerID][startpoint+UPDATE_GLOBAL_ITER+1][0]
            episode_buffer        = self.episode_buffer['worker%d'%self.workerID][startpoint:startpoint+UPDATE_GLOBAL_ITER+1]

            s1Value = SESS.run(self.local_AC.value, {self.local_AC.inputs: np.reshape(np.array(s1),[1,s_size])})[0, 0]
            v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,s1Value)

            self.actorLoss.append(p_l)
            self.criticLoss.append(v_l)

        # Pull from global net
        SESS.run(self.pull_global)
        self.lastUpdate = len(self.episode_buffer['worker%d'%self.workerID])+1

        self.tensorboardPlot()
        # last thread saves model
        print("Saving Model (GLOBAL_EP={})...".format(GLOBAL_EP), end='\n')
        saver.save(SESS, model_path+'/model-'+str(GLOBAL_EP)+'.cptk')





if __name__ == "__main__":
    ## begin Tensorflow initialization:

    # initialize groupLock
    groups = [['main'],[]]
    for i in range(N_WORKERS):
        workerName = 'W_%i' % i   # worker name
        groups[1].append(workerName)
    groupLock = GroupLock.GroupLock(groups)

    groupLock.acquire(0,'main')#lock here

    SESS = tf.Session()

    # Copy cpg0 as initial cpg value
    cpg = copy(cpg0)

    # Initiralize a global dictionary to store episode buffers
    episodeBuffer = {}
    for i in range(N_WORKERS):
        episodeBuffer['worker%d'%i] = []

    # initilize optimizers and workers
    with tf.device("/cpu:0"):
        trainer = tf.train.AdamOptimizer(LR_AC)
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # Global net: we only need its parameters
        workers = [] #creat workers
        for i in range(N_WORKERS):
            workerName = 'W_%i' % i
            workers.append(Worker(workerName, GLOBAL_AC, i, cpg, 1, groupLock))
        saver = tf.train.Saver(max_to_keep=NsavedModels)

    # load old model or Initialize new vairables
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(SESS,ckpt.model_checkpoint_path)
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
        th = threading.Thread(target = worker.work)
        th.start()
        worker_threads.append(th)
    ## end Tensorflow initialiization

    for t in range(nIter):
        tStart = time.perf_counter()

        if t == cpg['initLength']:
            print('Snake Monster is ready!')
            print('Begin walking')

        ## Joystick stuffs - Exit Button
        if joy.pressed[2]:
            #resetSnakeMonster()
            print('Reset.\n')
            print('Exiting at t = {}\n'.format(t))
            break

        if joy.pressed[1]:
            print('Exiting at t = {}\n'.format(t))
            break

        ### Jostick stuffs - controls
        #if t >= cpg['initLength'] and cpgJoy:
            #if any([abs(c) > 20 for c in joy.ch]):
                #cpg['move']= True
                #if joy.channel(1) > 0:
                    #cpg['direction']= forward
                #elif joy.channel(1) < 0:
                    #cpg['direction']= backward
                ##elif joy.channel(0) > 0:
                    ##cpg['direction']= rightturn
                ##elif joy.channel(0) < 0:
                    ##cpg['direction']= leftturn
            #else:
                #cpg['move']= False

        ## Get pose/gravity vector
        fbk_imu.getNextFeedback()
        CF.update(fbk_imu)
        cpg['pose']= copy(CF.R)

        ## Get pose/gravity vector
        fbk_imu.getNextFeedback()
        CF.update(fbk_imu)
        cpg['pose']= copy(CF.R)

        ## Get Snake Monster feedback
        fbk_sm.getNextFeedback()

        ## Correct leg angles based on fbk_sm structure
        cpg['legs'] = cpg['legsAlpha'] * fbk_sm.position + (1 - cpg['legsAlpha']) * cpg['legs']
        cpg['torques'] = np.reshape(fbk_sm.torques, [3,6])

        if t > cpg['initLength']:
            cpg = stabilizationPID(cpg)
            groupLock.release(0,'main')#unlock here
            # Threads do their business...
            groupLock.acquire(0,'main')#lock here

        # Update cpg
        cpg = CPGgs(cpg, t, dt)

        #print(cpg['dTheta2'], cpg['theta2'])
        # Command
        cmd.position = cpg['legs']

        snakeMonster.setAngles(cmd.position[0])

        loopTime = time.perf_counter() - tStart
        time.sleep(max(0,dt-loopTime))

    print("Saving Model (GLOBAL_EP={})...".format(GLOBAL_EP), end='\r')
    saver.save(SESS, model_path+'/model-'+str(GLOBAL_EP)+'.cptk')
    print('OK\nTrying to exit code cleanly...')
    joy.running = False
    for worker in workers:
        print('Sending kill signal to thread {}...'.format(worker.name))
        worker.shouldDie = True
    time.sleep(0.1)
    print('Letting Threads stop...')
    groupLock.release(0,'main') #unlock here and let threads stop cleanly
    sleep(1)
    print('Threads stopped, you should Ctrl-C...')



                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.sin(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.sin(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.sin(self.cpg['legs'][0,elbows][self.workerID]),
                            #np.cos(self.cpg['legs'][0,shoulders1][self.workerID]) * np.cos(self.cpg['legs'][0,shoulders2][self.workerID]) * np.cos(self.cpg['legs'][0,elbows][self.workerID]),
