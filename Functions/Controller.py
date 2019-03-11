import threading
from inputs import get_gamepad # pip install inputs
import numpy as np


class Controller(threading.Thread):
    #UNTITLED2 Summary of self class goes here
    #   Detailed explanation goes here

    # Channels

    # 2         4
    # ^         ^
    # |         |
    # L _ > 1   L _ > 3


    deadband = None
    ch = [0,0,0,0]

    pressed = None
    lastPressed = None
    
    #Setup
    def __init__(self, *args):

        self.dir = -1

        self.deadband = 0.35

        self.pressed = np.zeros(13)
        self.lastPressed = np.zeros(13)
        self.running = True
        super().__init__(*args)
        self.start()

    def run(self):
        while self.running:
            try:
                #self.ch = [0,0,0,0]
                events = get_gamepad()
                for event in events:
                    if not event.ev_type == 'Sync' and not event.ev_type == 'Misc':
                        if event.code == 'BTN_TRIGGER':
                            self.pressed[0] = event.state
                        elif event.code == 'BTN_THUMB':
                            self.pressed[1] = event.state
                        elif event.code == 'BTN_THUMB2':
                            self.pressed[2] = event.state
                        elif event.code == 'BTN_TOP':
                            self.pressed[3] = event.state
                        elif event.code == 'BTN_TOP2':
                            self.pressed[4] = event.state
                        elif event.code == 'BTN_PINKIE':
                            self.pressed[5] = event.state
                        elif event.code == 'BTN_BASE':
                            self.pressed[6] = event.state
                        elif event.code == 'BTN_BASE2':
                            self.pressed[7] = event.state
                        elif event.code == 'BTN_BASE3':
                            self.pressed[8] = event.state
                        elif event.code == 'BTN_BASE4':
                            self.pressed[9] = event.state
                        ## Left joypad
                        elif event.code == 'ABS_Y':
                            self.ch[1] = 128 - event.state
                        elif event.code == 'ABS_X':
                            self.ch[0] = event.state - 128
                    else:
                            pass
            
            except inputs.UnpluggedError:
                pass
    #Update
    def updateController(self):
        self.lastPressed = self.pressed

    #Setters

    def setDeadband(self, deadband):
        self.deadband = deadband

    #Getters

    def channel(self, num):
        return self.ch[num]

    def button(self, num):
        return self.pressed(num)

    def buttonPressed(self, num):
        return self.pressed(num) & ~self.lastPressed(num)

    #Tools

    def adjustValue(self,up,down,value,dV):
        if self.buttonPressed(up):
            value = value + dV
        elif self.buttonPressed(down):
            value = value - dV
if __name__ == '__main__':
    c = Controller()
