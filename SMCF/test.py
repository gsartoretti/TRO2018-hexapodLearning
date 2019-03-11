import hebiapi
import time

hebi = hebiapi.HebiLookup()

__NAMES__ = ['SA078', 'SA040', 'SA048', 
		'SA058', 'SA057', 'SA001', 
		'SA041', 'SA072', 'SA077', 
		'SA051', 'SA059', 'SA012', 
		'SA081', 'SA050', 'SA018', 
		'SA046', 'SA056', 'SA026']

group = hebi.getGroupFromNames(['SA020','SA073'])
# group = hebi.getConnectedGroupFromName('SA078')


while True:
	print(group.getNumModules())
	print(group.getFeedback().getAccelerometers())
	time.sleep(0.2)
