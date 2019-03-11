def gravComp(smk, legs, gravVec):
    gravCompTorques = smk.getLegGravCompTorques(legs, gravVec.T)
    gravCompTorques = gravCompTorques[0:3]
    gravCompTorques = gravCompTorques.T
    return gravCompTorques
