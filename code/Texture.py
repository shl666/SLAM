import os
from PIL import Image
from ParticleFilter import ParticleFilter
import numpy as np
import sys
from matplotlib import pyplot as plt

class Texture(ParticleFilter):
    def __init__(self,prefix,count=20,dataset=20,noisy=True, NOISE_FSCTOR = (1,1,2),TEXTURE = True):
        ParticleFilter.__init__(self,prefix,count,dataset,noisy,NOISE_FSCTOR)
        self.depth_list = []
        self.rgbi_list = []
        self.rgbj_list = []
        self.rgb_list = []
        self.floor_count = 0
        if TEXTURE:
            self.LoadRBGD()
        
    def LoadRBGD(self):
        print('Start loading Depth data ...')
        
        with np.load(self.prefix+"/Kinect%d.npz"%self.dataset) as data:
            self.disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
            self.rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
            
        path = '../data/dataRGBD/Disparity%d/'%self.dataset
        dirs = os.listdir(path)
        for i in range(len(dirs)):
            img = Image.open(path+'disparity{}_{}.png'.format(self.dataset,i+1))
            disparity_img = np.array(img.getdata(),np.uint16).reshape(img.size[1], img.size[0])
            dd = -0.00304*disparity_img +3.31
            depth = 1.03/dd
            rgbi = ((np.arange(dd.shape[0]).reshape(-1,1)*526.37+dd*(-4.5*1750.46)+19276)/585.051).astype(int)
            rgbj = ((np.arange(dd.shape[1]).reshape(1,-1)*np.ones(dd.shape)*526.37+16662)/585.051).astype(int)
            self.depth_list.append(depth)
            self.rgbi_list.append(rgbi)
            self.rgbj_list.append(rgbj)
#            sys.stdout.write('{}% completed    \r'.format(i * 100 / len(dirs) )
#            sys.stdout.flush()
        print('Finished loading Depth data.')
        rgb_path = '../data/dataRGBD/RGB%d/'%self.dataset
        dirs = os.listdir(rgb_path)
        print('Start loading RGB data ...')
        for i in range(len(dirs)):
            img = plt.imread(rgb_path+'rgb{}_{}.png'.format(self.dataset,i+1))
            self.rgb_list.append(img)
#            sys.stdout.write('{}% completed    \r'.format(i * 100 / len(dirs) )
#            sys.stdout.flush()
        print('Finish loading RGB data.')
            
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))

    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1]
    
    def Pixel2World(self,idx,best_particle):
        m,n = self.depth_list[idx].shape
        K = np.array([[585.05108211, 0, 242.94140713],
                      [0, 585.05108211, 315.83800193],
                      [0,0,1]])
        K_inv = np.linalg.inv(K)
        r = 0
        p = 0.36
        y = 0.021
        cr,sr = np.cos(r), np.sin(r)
        cp,sp = np.cos(p), np.sin(p)
        cy,sy = np.cos(y), np.sin(y)
        Roc = np.array([[0, -1, 0, 0],
                       [0,  0, -1, 0],
                       [ 1,   0, 0, 0],
                       [ 0,   0, 0, 1]])
        Rr = np.array([[cr, -sr, 0, 0],
                       [sr,  cr, 0, 0],
                       [ 0,   0, 1, 0],
                       [ 0,   0, 0, 1]])
        Rp = np.array([[ cp,   0,  sp, 0],
                       [  0,   1,   0, 0],
                       [-sp,   0,  cp, 0],
                       [  0,   0,   0, 1]])
        Ry = np.array([[  1,   0,   0, 0],
                       [  0,  cy, -sy, 0],
                       [  0,  sy,  cy, 0],
                       [  0,   0,   0, 1]])
        t = np.array([[1,0,0, 0.18],
                       [0,1,0, 0.005],
                       [0,0,1, 0.36],
                       [0,0,0, 1]])
        T_b2k = Roc @ Rr @ Rp @ Ry @ t
        T_k2b = np.linalg.inv(T_b2k)
        pts_pixels = np.ones((3,m*n)) #[:,i*m+j]
        pts_pixels[0] = (np.ones((m,n))*np.arange(n).reshape(1,-1)).flatten()
        pts_pixels[1] = (np.ones((m,n))*(np.arange(m)[::-1]).reshape(-1,1)).flatten()
        # pixel2camera
        pts_camera = self.Homogenize(self.depth_list[idx].reshape(-1) * (K_inv @ pts_pixels))
        # camera2body
        pts_body = T_k2b @ pts_camera
        #body2world
        pts_body = self.Dehomogenize(pts_body)
        pts_body[-1] /= pts_body[-1]
        tx = best_particle[0,0]
        ty = best_particle[1,0]
        theta = best_particle[2,0]
        R = np.array([[np.cos(theta), -np.sin(theta), tx],
                      [np.sin(theta), np.cos(theta),  ty],
                      [0,             0,              1 ]])
        pts_world = R @ pts_body# 3 x m*n, get(i,j)->pts_world[:,i*m+j]
        return pts_world
    
    def MappingFloor(self, stamp, idx):
        depth = self.depth_list[idx].reshape(-1)
        best_particle = self.particles[:,np.argmax(self.weights)].reshape(3,1)
        pts_world = self.Pixel2World(idx,best_particle)
        filter_logic = np.logical_and(-1<depth,depth<2)
        pts_world = pts_world[:,filter_logic]
        x_world = pts_world[0]
        y_world = pts_world[1]
        
        rgb_idx = self.FindNearest(self.rgb_stamps, stamp)
        rgb_img = self.rgb_list[rgb_idx]
        rgbi = self.rgbi_list[rgb_idx].reshape(-1)[filter_logic]
        rgbj = self.rgbj_list[rgb_idx].reshape(-1)[filter_logic]
        rgb_indice = np.vstack((rgbi,rgbj))
        
        # convert from meters to cells
        xis = np.ceil((x_world - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((y_world - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        indGood1 = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)),
                                                 (xis < self.MAP['sizex'])), (yis < self.MAP['sizey']))
        indGood2 = np.logical_and(np.logical_and(0<rgbi, rgbi<rgb_img.shape[0]),
                                  np.logical_and(0<rgbj, rgbj<rgb_img.shape[1]))
        indGood = np.logical_and(indGood1,indGood2)
        self.MAP['floormap'][xis[indGood],yis[indGood]] = rgb_img[rgbi[indGood],rgbj[indGood]]
        
        if self.floor_count % 200 == 0:
            plt.figure(figsize = (10,10))
            plt.imshow(self.MAP['floormap'])
            plt.title('floor')
            plt.savefig('images/{}texture_{}.png'.format(self.dataset,self.floor_count//200))
        self.floor_count += 1
