import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import core as core
from env import SingleSpineEnvSingle
from dataLoad.loadNii import get_spinedata
def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    return spine_data # namedtuple:
def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnvSingle.SpineEnv(spine_data, degree_threshold, **cfg)
    return env
def draw():
    cfg = {'deg_threshold':[-360., 360., -360., 360.],
           'reset':{'rdrange':[-90, 90],
                    'state_shape':(160, 80, 64)},
           'step':{'rotate_mag':[5, 5], 'discrete_action':False}
           }
    dataDirs =  [r'spineData/sub-verse621_L1_ALL_msk.nii.gz']
    pedicle_points = np.asarray([[[35,47,65],[36,47,105]]])
    # dataDirs =  [r'spineData/sub-verse621_L3_ALL_msk.nii.gz']
    # pedicle_points = np.asarray([[[38,47,62],[39,47,104]]])
    # dataDirs =  [r'spineData/sub-verse521_dir-ax_L1_ALL_msk.nii.gz']
    # pedicle_points = np.asarray([[[30,46,65],[30,46,108]]])
    # dataDirs =  [r'spineData/sub-verse537_dir-iso_L4_ALL_msk.nii.gz']
    # pedicle_points = np.asarray([[[46,44,63],[46,44,101]]])
    
    for dataDir, pedicle_point in zip(dataDirs, pedicle_points):
        dataDir = os.path.join(os.getcwd(), dataDir)
        pedicle_point_in_zyx = True #坐标是zyx形式吗？
        spine_data = build_data_load(dataDir, pedicle_point, pedicle_point_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
        '''---2 Build Environment  ---'''
        envs = build_Env(spine_data, cfg['deg_threshold'], cfg)  # 只修改了初始化函数，其他函数待修改
        _, o_3D = envs.reset(random_reset = False)
        rewards = []
        step = 0
        x_axis = []
        while step<100:
            step += 1
            radian_L = np.array([.0, round(0.0-0.05*step, 2)])
            # radian_L = np.array([round(0.0-0.05*step, 2), .0])
            x_axis.append(round(0.0-0.05*step, 2))
            reward = envs.simulate_volume(radian_L)
            rewards.append(reward)
        x_axis.reverse()
        rewards.reverse()
        step = 0
        while step<100:
            step += 1
            radian_L = np.array([.0, round(0.0+0.05*step, 2)])
            # radian_L = np.array([round(0.0+0.05*step, 2), .0])
            x_axis.append(round(0.0+0.05*step, 2))
            reward = envs.simulate_volume(radian_L)
            rewards.append(reward)
        print(x_axis)
        print(rewards)
        plt.plot(x_axis, rewards)
        plt.show()
        plt.savefig('imgs/vertical_plot621L1.png')
draw()