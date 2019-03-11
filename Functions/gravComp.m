function [gravCompTorques] = gravComp(smk, legs, gravVec)
    gravCompTorques = smk.getLegGravCompTorques(legs, gravVec');
    gravCompTorques = gravCompTorques(1:3,:);
    gravCompTorques = (gravCompTorques(:))';
end

