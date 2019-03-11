import pickle
import tools
import setupfunctions as setup
import tools
import SMCF
def setupWrapper():
# Wrapper sets up the snake monster for use in the CPG code.

    print('Setting up snake monster...')

    with open('names.dat', 'rb') as f:
        names = pickle.load(f)
    shoulders = names[1::3]
    imu = tools.HebiLookup.newGroupFromNames(shoulders)
    snakeData = setup.setupSnakeMonsterShoulderData();
    smk = sm.SnakeMonsterKinematics()

    with open('offsets.dat', 'rb') as f:
        offsets = pickle.load(f)

    setup.setupSnakeMonster()
    cmd = tools.CommandStruct()
    CF = SMCF.SMComplementaryFilter(snakeData, 'accelOffsets', accelOffsets, 'gyroOffsets', gyroOffsets);
    CF.update(imu.getNextFeedback());

    print('Setup complete!')
    return (imu, smk, cmd, CF)
