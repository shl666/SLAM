import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from MAP import MAP
        
class ParticleFilter(MAP):
    def __init__(self,prefix = "../data", count=1,dataset=20,noisy=True,NOISE_FACTOR = (1,1,2)):
        MAP.__init__(self)
        self.prefix = prefix
        self.dataset = dataset
        self.noise_factor = NOISE_FACTOR
        self.plotcount = 0
        self.w = 0
        self.d = 0
        self.t_sb = np.array([[298.33,0]]).T/1000 #sensor coord to body coord 
        self.best_particle = np.zeros((3,1))
        self.best_particle_map = np.zeros((2,1))
        self.count = count
        self.xs = np.arange(-0.2,0.2+0.05,0.05)
        self.ys = np.arange(-0.2,0.2+0.05,0.05)
        self.particles, self.weights = self.pw_initial()
        self.LoadData()
        self.best_particle_record = [self.best_particle_map]
        self.angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        self.ranges = self.lidar_ranges[:,0]
        self.Neff_tresh = 35
        self.noisy = noisy
        
    def pw_initial(self):
        particles = np.zeros((3,self.count))
        weights = np.ones((self.count))/self.count
        return particles, weights 
    
    def FindNearest(self,array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def LoadData(self):
        print('Start loading motion and lidar data ...')
        with np.load(self.prefix+"/Encoders%d.npz"%self.dataset) as data:
            self.encoder_counts = data["counts"] # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"] # encoder time stamps

        with np.load(self.prefix+"/Hokuyo%d.npz"%self.dataset) as data:
            self.lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
            self.lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
            self.lidar_range_min = data["range_min"] # minimum range value [m]
            self.lidar_range_max = data["range_max"] # maximum range value [m]
            self.lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

        with np.load(self.prefix+"/Imu%d.npz"%self.dataset) as data:
            self.imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
            self.imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
            self.imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
            
        ## low pass fliter and average
        order = 1
        fs = 100  # sample rate, Hz
        cutoff = 10  # desired cutoff frequency of the filter, Hz
        self.imu_angular_velocity[2] = self.butter_lowpass_filter(self.imu_angular_velocity[2], cutoff, fs, order)
        self.imu_angular_velocity[2] += np.append(self.imu_angular_velocity[2][1:],[0])\
                                        + np.append(self.imu_angular_velocity[2][2:],[0,0])
        self.imu_angular_velocity[2] /=3
        self.imu_angular_velocity[2][-1] = self.imu_angular_velocity[2][-3]
        self.imu_angular_velocity[2][-2] = self.imu_angular_velocity[2][-3]
        
        self.encoder_counts += np.concatenate([self.encoder_counts[:,1:],np.zeros((4,1)).astype(int)],1)\
                                        + np.concatenate([self.encoder_counts[:,2:],np.zeros((4,2)).astype(int)],1)
        self.encoder_counts //=3
        self.encoder_counts[:,-1] = self.encoder_counts[:,-3]
        self.encoder_counts[:,-2] = self.encoder_counts[:,-3]

        print('Finishied loading motion and lidar data.')
            
    def GetU(self, stamp):
        encoder_count = self.encoder_counts[:,self.FindNearest(self.encoder_stamps, stamp)]
        angular_velocity = self.imu_angular_velocity[:,self.FindNearest(self.imu_stamps,stamp)]
        dl = (encoder_count[0]+encoder_count[2])/2*0.0022
        dr = (encoder_count[1]+encoder_count[3])/2*0.0022
        
        self.w = angular_velocity[2] * np.ones(self.count)
        self.d = (dl+dr)/2 * np.ones(self.count)

    def DTDDM(self, time_stamp, prev_stamp):
        # particles[i] = np.array([[x,y,theta]]).T
        self.GetU(time_stamp)
        time_delta = time_stamp - prev_stamp
        self.particles[0] += self.d*np.sinc(self.w*time_delta/2) * np.cos(self.particles[2] + self.w*time_delta/2)
        self.particles[1] += self.d*np.sinc(self.w*time_delta/2) * np.sin(self.particles[2] + self.w*time_delta/2)
        self.particles[2] += self.w*time_delta
    
    def resample(self):
        particle_new = np.zeros((3, self.count))
        tempweight = self.weights[0]
        j=0
        for i in range(self.count):
            u = np.random.uniform(0, 1/self.count)
            beta = u + i/self.count
            while beta>tempweight:
                j += 1
                tempweight += self.weights[j]
            particle_new[:, i] = self.particles[:, j]
        self.particles = particle_new
        self.weights = np.ones((self.count))/self.count
        
    def Prediction(self,prev_stamp,time_stamp):
        self.Mapping(update_flag = False)
        time_delta = time_stamp - prev_stamp
        
        if self.noisy:
            mu, sigma = 0, 1e-3
            self.particles[0] += self.noise_factor[0]*np.random.normal(mu, sigma, self.count)
            self.particles[1] += self.noise_factor[1]*np.random.normal(mu, sigma, self.count)
            self.particles[2] += self.noise_factor[2]*np.random.normal(mu, sigma, self.count)#*self.w*time_delta
            
            
        self.DTDDM(time_stamp, prev_stamp)
        
    def Update(self,l):
        ## update weight
        ranges = self.lidar_ranges[:,l]
        angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        indValid = np.logical_and((ranges < 30),(ranges> 0.1))
        self.ranges = ranges[indValid]
        self.angles = angles[indValid]
            
        
        for i in range(self.count):
            #get xy position in the world frame
            xw0, yw0 = self.P2W(self.ranges,self.angles,self.particles[:,i])
            vp = np.stack((xw0,yw0))
            cpr = self.mapCorrelation(vp)
            if cpr.max() != 0:
                x_idx, y_idx = np.unravel_index(np.argmax(cpr, axis=None), cpr.shape)
                self.particles[0,i] += self.xs[x_idx]
                self.particles[1,i] += self.ys[y_idx]
                xw0, yw0 = self.P2W(self.ranges,self.angles,self.particles[:,i])
                vp = np.stack((xw0,yw0))
                self.weights[i] = np.exp(cpr.max()/10) 
        self.weights = self.weights.reshape(-1)/self.weights.sum()
        
        ## update particles
        Neff = (1/(self.weights**2)).sum()
        if Neff<self.Neff_tresh:
            self.resample()
        
    def Mapping(self, update_flag = True):
        if np.linalg.norm((self.particles[:,np.argmax(self.weights)].reshape(3,1)-self.best_particle)[:-1,:])<0.5:
            self.best_particle = self.particles[:,np.argmax(self.weights)].reshape(3,1)
        xw0_best, yw0_best = self.P2W(self.ranges,self.angles,self.best_particle[:,0])
        # convert from meters to cells
        xis = np.ceil((xw0_best - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((yw0_best - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1
        
        self.best_particle_map[0,0] = np.ceil((self.best_particle[0] - self.MAP['xmin']) 
                                              / self.MAP['res'] ).astype(np.int16)-1
        self.best_particle_map[1,0] = np.ceil((self.best_particle[1] - self.MAP['ymin']) 
                                              / self.MAP['res'] ).astype(np.int16)-1
        bpmap_x = int(self.best_particle_map[0,0])
        bpmap_y = int(self.best_particle_map[1,0])
        self.best_particle_record.append(np.array([[bpmap_x,bpmap_y]]).T)
        
        # mapping 
        
        if update_flag:
            covered_pts_record = []
            for i in range(xis.shape[0]):
                covered_pts = self.bresenham2D(bpmap_x, 
                                          bpmap_y, 
                                          xis[i], yis[i])[:,:-1].astype(np.int16)
                covered_pts_record.append(covered_pts)
            covered_pts = np.concatenate(covered_pts_record,1)

            indGood = np.logical_and(np.logical_and(np.logical_and((covered_pts[0] > 1), (covered_pts[1] > 1)), \
                                                    (covered_pts[0] < self.MAP['sizex'])), 
                                     (covered_pts[1] < self.MAP['sizey']))
            covered_pts_corrected = np.concatenate([covered_pts[0][indGood].reshape(1,-1),
                                                    covered_pts[1][indGood].reshape(1,-1)])
            indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), \
                                                    (xis < self.MAP['sizex'])), (yis < self.MAP['sizey']))
            boundary_corrected = np.concatenate([xis[indGood].reshape(1,-1),yis[indGood].reshape(1,-1)])
            
            self.MAP['zerosmap'][covered_pts_corrected[0],covered_pts_corrected[1]]=np.log(1/4)
            self.MAP['zerosmap'][xis[indGood],yis[indGood]]=np.log(4)
            self.MAP['logoddmap'] += self.MAP['zerosmap']
#             self.MAP['logoddmap'][self.MAP['logoddmap']>20] = 20
#             self.MAP['logoddmap'][self.MAP['logoddmap']<-30] = -30
#             self.MAP['logoddmap'][self.MAP['logoddmap']>40] = 40
            self.MAP['logoddmap'][self.MAP['logoddmap']<-40] = -40
            self.MAP['map'][(self.MAP['logoddmap']>0) * (self.MAP['map']!=3)]=1
            self.MAP['map'][(self.MAP['logoddmap']<0) * (self.MAP['map']!=3)]=2
            self.MAP['wallmap'][self.MAP['logoddmap']>0] = 1
#             self.MAP['wallmap'][self.MAP['logoddmap']<0] = -1
            self.MAP['zerosmap'] = np.zeros((self.MAP['sizex'],self.MAP['sizey']),dtype=np.int8)
        
        
        indGood = np.logical_and(np.logical_and(np.logical_and((self.best_particle_map[0] > 1), 
                                                               (self.best_particle_map[1] > 1)), 
                                                (self.best_particle_map[0] < self.MAP['sizex'])),
                                 (self.best_particle_map[1] < self.MAP['sizey']))
        self.best_particle_map = self.best_particle_map.astype(np.int16)
        self.MAP['map'][self.best_particle_map[0][indGood],self.best_particle_map[1][indGood]]=3
        
        self.plotcount += 1
        if self.plotcount %500 == 0:
            plt.figure(figsize = (10,10))
            plt.imshow(self.MAP['map'],cmap='hot')
            plt.title('lidar')
            plt.savefig('images/{}scan_{}.png'.format(self.dataset,self.plotcount//500))
            #plt.show()
        
        
    def P2W(self,ranges,angles,temp_particle):
        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        # xy position in the body frame
        xb0 = xs0 + np.cos(temp_particle[2])*self.t_sb[0]
        yb0 = ys0 + np.sin(temp_particle[2])*self.t_sb[1]
        # xy position in the world frame
        temp = np.vstack((xb0,yb0,np.ones(xb0.shape)))
        theta = temp_particle[2]
        R = np.array([[np.cos(theta), -np.sin(theta), temp_particle[0]],
                      [np.sin(theta), np.cos(theta),  temp_particle[1]],
                      [0,             0,              1              ]])
        res = R @ temp
        xw0 = res[0]
        yw0 = res[1]
        return xw0, yw0
        
    def mapCorrelation(self, vp):
        '''
        INPUT 
        vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  

        OUTPUT 
        cpr             sum of the cell values of all the positions hit by range sensor
        '''
        nx = self.MAP['wallmap'].shape[0]
        ny = self.MAP['wallmap'].shape[1]
        xmin = self.MAP['x_im'][0]
        xmax = self.MAP['x_im'][-1]
        xresolution = (xmax-xmin)/(nx-1)
        ymin = self.MAP['y_im'][0]
        ymax = self.MAP['y_im'][-1]
        yresolution = (ymax-ymin)/(ny-1)
        nxs = self.xs.size
        nys = self.ys.size
        cpr = np.zeros((nxs, nys))
        for jy in range(nys):
            y1 = vp[1,:] + self.ys[jy] # 1 x 1076
            iy = np.int16(np.round((y1-ymin)/yresolution))
            for jx in range(0,nxs):
                x1 = vp[0,:] + self.xs[jx] # 1 x 1076
                ix = np.int16(np.round((x1-xmin)/xresolution))
                valid = np.logical_and(np.logical_and((iy >=0), (iy < ny)),
                                       np.logical_and((ix >=0), (ix < nx)))
                cpr[jx,jy] = np.sum(self.MAP['wallmap'][ix[valid],iy[valid]])
        return cpr
    
    def bresenham2D(self, sx, sy, ex, ey):
        '''
        Bresenham's ray tracing algorithm in 2D.
        Inputs:
          (sx, sy)	start point of ray
          (ex, ey)	end point of ray
        '''
        sx = int(round(sx))
        sy = int(round(sy))
        ex = int(round(ex))
        ey = int(round(ey))
        dx = abs(ex-sx)
        dy = abs(ey-sy)
        steep = abs(dy)>abs(dx)
        if steep:
            dx,dy = dy,dx # swap 

        if dy == 0:
            q = np.zeros((dx+1,1))
        else:
            q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
        if steep:
            if sy <= ey:
                y = np.arange(sy,ey+1)
            else:
                y = np.arange(sy,ey-1,-1)
            if sx <= ex:
                x = sx + np.cumsum(q)
            else:
                x = sx - np.cumsum(q)
        else:
            if sx <= ex:
                x = np.arange(sx,ex+1)
            else:
                x = np.arange(sx,ex-1,-1)
            if sy <= ey:
                y = sy + np.cumsum(q)
            else:
                y = sy - np.cumsum(q)
        return np.vstack((x,y))
    
    
