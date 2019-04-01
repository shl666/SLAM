import numpy as np
from Texture import Texture
from IPython.display import display, clear_output
import sys
from matplotlib import pyplot as plt
############# need to be set###########
dataset = 20
TEXTURE = True #must switch to Flase when use dataset23
NOISE_FACTOR = (1,1,2) #scalar
NOISY = True
COUNT = 100
#NOISY = False
#COUNT = 1
prefix = "../data"
#######################################

with np.load(prefix+"/Encoders%d.npz"%dataset) as data:
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load(prefix+"/Hokuyo%d.npz"%dataset) as data:
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load(prefix+"/Imu%d.npz"%dataset) as data:
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

if TEXTURE:
    with np.load(prefix+"/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

if TEXTURE:
    START_TIME_STAMP = min(encoder_stamps[0],lidar_stamps[0],imu_stamps[0],disp_stamps[0],rgb_stamps[0])
    END_TIME_STAMP = max(encoder_stamps[-1],lidar_stamps[-1],imu_stamps[-1],disp_stamps[-1],rgb_stamps[-1])
else:
    START_TIME_STAMP = min(encoder_stamps[0],lidar_stamps[0],imu_stamps[0])
    END_TIME_STAMP = max(encoder_stamps[-1],lidar_stamps[-1],imu_stamps[-1])

time_stamp = START_TIME_STAMP
prev_stamp = START_TIME_STAMP

TIME_GAP = 0.001
ENCODER_INDEX=0
LIDAR_INDEX=0
DISP_INDEX=0
LIDAR_FLAG = True
ENCODER_FLAG = True
DISP_FLAG = True


tx = Texture(prefix, COUNT,dataset,NOISY,NOISE_FACTOR,TEXTURE)
while (time_stamp<END_TIME_STAMP):
    sys.stdout.write('{}% completed    \r'.format((time_stamp-START_TIME_STAMP) * 100 / (END_TIME_STAMP-START_TIME_STAMP)) )
    sys.stdout.flush()
    if time_stamp>encoder_stamps[ENCODER_INDEX] and ENCODER_FLAG:
        tx.Prediction(prev_stamp,time_stamp)
        prev_stamp = time_stamp
        if ENCODER_INDEX<encoder_stamps.shape[0]-1:
            ENCODER_INDEX += 1
        else:
            ENCODER_FLAG = False
    elif time_stamp>lidar_stamps[LIDAR_INDEX] and LIDAR_FLAG:
        tx.Update(LIDAR_INDEX)
        tx.Mapping()
        if LIDAR_INDEX<lidar_stamps.shape[0]-1:
            LIDAR_INDEX += 1
        else:
            LIDAR_FLAG = False
    elif TEXTURE:
        if time_stamp>disp_stamps[DISP_INDEX] and DISP_FLAG:
            tx.MappingFloor(time_stamp,DISP_INDEX)
            if DISP_INDEX<disp_stamps.shape[0]-1:
                DISP_INDEX += 1
            else:
                DISP_FLAG = False
    time_stamp += TIME_GAP
print('mission complete')


trace = np.hstack(tx.best_particle_record)
plt.figure(figsize = (20,20))
plt.imshow(tx.MAP['map'], zorder=10)
plt.plot(trace[1],trace[0] , '.b', zorder=30)
if TEXTURE:
    res = np.zeros(tx.MAP['floormap'].shape)
    for i in range(tx.MAP['floormap'].shape[0]):
        for j in range(tx.MAP['floormap'].shape[1]):
            if tx.MAP['map'][i,j] == 1:
                res[i,j] = np.array([255.,0,0])
            elif tx.MAP['map'][i,j] == 2:
                res[i,j] = np.array([255.,255.,0])
    for i in range(tx.MAP['floormap'].shape[0]):
        for j in range(tx.MAP['floormap'].shape[1]):
            if all(tx.MAP['floormap'][i,j] != np.array([0,0,0])) and (tx.MAP['map'][i,j] == 2 or tx.MAP['map'][i,j] == 1):
                res[i,j] = tx.MAP['floormap'][i,j]
    plt.imshow(res , zorder=20)
plt.savefig('images/{}_final.png'.format(dataset))
plt.show()
