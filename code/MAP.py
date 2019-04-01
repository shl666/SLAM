import numpy as np

class MAP():
    def __init__(self):
        self.MAP = {}
        self.MAP['res']   = 0.05 #meters
        self.MAP['xmin']  = -30  #meters
        self.MAP['ymin']  = -30
        self.MAP['xmax']  =  30
        self.MAP['ymax']  =  30
        self.MAP['sizex']  = int(np.ceil((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) #cells
        self.MAP['sizey']  = int(np.ceil((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['map'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP['logoddmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']))
        self.MAP['zerosmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']))
        self.MAP['wallmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'])) #DATA TYPE: char or int8
        self.MAP['x_im'] = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        self.MAP['y_im'] = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map
        self.MAP['floormap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey'],3))

