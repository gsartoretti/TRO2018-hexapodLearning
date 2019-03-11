function [cpg] = stabilizationPID(cpg)

%Gives a rotation matrix for correct orientation given the world frame
R = XYrot(cpg.pose);

FKbody = cpg.requestedLegPositions;

FKworld = cpg.pose * FKbody;

FKworldCor = R(1:3,1:3) * FKworld;
cpg.feetTemp = FKworldCor;

% Set the stance to the lowest tripod
use145 = mean(FKworldCor(3,[1,4,5])) < mean(FKworldCor(3,[2,3,6]));
cpg.isStance([1,4,5]) = use145;
cpg.isStance([2,3,6]) = ~use145;

%% Compute Height from Ground Plane

dirVec = [0;0;1];

% Get positions of feet on ground
groundPositions = FKworldCor(:,cpg.isStance);
cpg.planeTemp = groundPositions;
cpg.planePoint = groundPositions(:,1);
 

%Determine normal of plane formed by ground feet
cpg.groundNorm = cross(groundPositions(:,1) - groundPositions(:,2), groundPositions(:,1) - groundPositions(:,3));

cpg.groundD = (cpg.groundNorm' * groundPositions(:,1));

%find intersection
t = cpg.groundD / (cpg.groundNorm' * dirVec);


%find height
cpg.zDist = norm(dirVec * t);


 %% Adjust legs for Z

cpg.zHistory(cpg.zHistoryCnt) = cpg.zDist;
cpg.zHistoryCnt = cpg.zHistoryCnt + 1;
cpg.zHistoryCnt = mod(cpg.zHistoryCnt,10) + 1;

zErr = cpg.bodyHeight - median(cpg.zHistory);

FKworldCor(3,:) = FKworldCor(3,:) + zErr;


cpg.dxWorld = FKworldCor - FKworld;

cpg.dx = -(cpg.pose \ cpg.dxWorld);

end
