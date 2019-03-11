import scipy.io as io
import hebiapi
HebiLookup = hebiapi.HebiLookup()
class Struct:
    pass
class CommandStruct:
    pass

def load(filename, varname=None, objarray=False, matrix=False):
    if varname is None:
        varname = filename.split('.')[0]
    output = None
    if objarray:
        output = [obj[0][0] for obj in io.loadmat(filename)[varname].T]
    elif matrix:
        output = [[cell for cell in row] for row in io.loadmat(filename)[varname]]
    return output
