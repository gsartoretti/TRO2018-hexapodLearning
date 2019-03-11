'''
%%%%%%%%%%%%%%%%%%%%%%%
% ONE GREAT BIG GROUP %
%%%%%%%%%%%%%%%%%%%%%%%

%Leg Numbering / Chassis Coordinate convention:
%
% 2 ----- 1 +y
% |^
% 4 ----- 3|
% |o--> +x
% 6 ----- 5+z
%


% HebiLookup.clearGroups();
% HebiLookup.clearModuleList();
'''
import pickle
import tools
import hebiapi

from SMCF.SMComplementaryFilter import NAMES

#names = tools.load('names.mat', objarray=True)

feet = ('FOOT_1', 'FOOT_2', 'FOOT_3', 'FOOT_4', 'FOOT_5', 'FOOT_6')

groupName = '*'

snakeMonster = tools.HebiLookup.getGroupFromNames(NAMES)

# Initialize all the gains
def constants(c): return [c for x in range(snakeMonster.getNumModules())]
gains = snakeMonster.groupGains
try:
    gains['ControlStrategy'] = constants(4)
    gains['PositionKp'] = constants(6)
    gains['PositionKi'] = constants(0.01)
    gains['PositionKd'] = constants(1)
    gains['TorqueKp'] = constants(1)
    gains['TorqueKi'] = constants(0)
    gains['TorqueKd'] = constants(0.1)
    gains['TorqueMaxOutput'] = constants(8.0)
    gains['TorqueMinOutput'] = [-x for x in gains['TorqueMaxOutput']]
    gains['PositionIClamp'] = constants(1)
    gains['VelocityKp'] = constants(1)
    gains['PositionMaxOutput'] = constants(10)
    gains['PositionMinOutput'] = constants(-10)
    gains['TorqueOutputLowpass'] = constants(.5)
    gains['TorqueFeedForward'] = constants(0.15)
    snakeMonster.setFeedbackFrequency(200)
    snakeMonster.setCommandLifetime(0)
    snakeMonster.sendCommand()
    print('Gains setup properly')
except hebiapi.base.HebiAccessError:
    print('Could not setup gains properly')
'''
%footGroup = HebiApi.newGroupFromNames(groupName, feet);

% % Setup matlabSnakeControl Tools
% mainDir = pwd;
% cd('matlabSnakeControl');
% SETUP_SCRIPT;
% cd(mainDir);

% add paths to working directory
% addpath(genpath([pwd '/tools'])); % genpath() includes all subfolders
%
% fprintf('Created SnakeMonster group and added paths.\n\n');
'''
