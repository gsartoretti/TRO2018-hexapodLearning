%%%%%%%%%%%%%%%%%%%%%%%
% ONE GREAT BIG GROUP %
%%%%%%%%%%%%%%%%%%%%%%%

%  Leg Numbering / Chassis Coordinate convention:
%  
%   2 ----- 1     +y
%       |          ^
%   4 ----- 3      |
%       |          o--> +x
%   6 ----- 5    +z
%


% HebiLookup.clearGroups();
% HebiLookup.clearModuleList();

load('names.mat');
      
feet = {'FOOT_1', 'FOOT_2', 'FOOT_3', 'FOOT_4', 'FOOT_5', 'FOOT_6'};

groupName = '*';

snakeMonster = HebiLookup.newGroupFromNames(groupName, names);

% Initialize all the gains
gains                           = snakeMonster.getGains();
ones_n                          = ones(1,18);
gains.controlStrategy           = ones_n*4;
gains.positionKp                = ones_n*6;
gains.positionKi                = ones_n*0.01;
gains.positionKd                = ones_n*1;
gains.torqueKp                  = ones_n*1;
gains.torqueKi                  = ones_n*0;
gains.torqueKd                  = ones_n*.1;
gains.torqueMaxOutput           = ones_n*8.;
gains.torqueMinOutput           = -gains.torqueMaxOutput;
gains.positionIClamp            = ones_n*1;
gains.velocityKp                = ones_n*1;
gains.positionMaxOutput         = ones_n*10;
gains.positionMinOutput         = ones_n*-10;
gains.torqueOutputLowpassGain   = ones_n*.5;
gains.torqueFF                  = ones_n*0.15;
snakeMonster.set('gains', gains);
snakeMonster.setFeedbackFrequency(200);
snakeMonster.setCommandLifetime(0);
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
