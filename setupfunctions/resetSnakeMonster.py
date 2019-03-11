
def resetSnakeMonster():

    # lays it on the ground
    setupSnakeMonster()
    snakeMonster.setCommandLifetime(0)
    cmd = tools.CommandStruct()
    cmd.torque = zeros(1,18)
    cmd.position = nan(1,18)
    snakeMonster.set(cmd)
