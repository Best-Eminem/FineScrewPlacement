import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from env import SingleSpineEnv
from dataLoad.loadNii import get_spinedata
from deap import algorithms, base, creator, tools
import time
import SimpleITK as sitk
import statistics

def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    return spine_data 

def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnv.SpineEnv(spine_data, degree_threshold, **cfg)
    return env

if __name__ == '__main__':
    screw_index = 1
    cfg = {'deg_threshold':[-30., 30., -60., 60.],#[-65., 65., -45., 25.],
           'screw_index':screw_index,
           'reset':{'rdrange':[-45, 45],
                    'state_shape':(160, 80, 64),},
           'step':{'rotate_mag':[10, 10], 'discrete_action':False}
           }
    dataDirs = [
                r'spineData/sub-verse621_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L2_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L3_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L4_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L5_ALL_msk.nii.gz',
                ]
    pedicle_points = np.asarray([
                                [[35,47,65],[36,47,105]],
                                [[36,48,62],[38,48,102]],
                                [[38,47,62],[39,47,104]],
                                [[43,48,60],[44,48,107]],
                                [[48,52,60],[46,51,122]],
                                 ])
    angles =  np.asarray([
                        [[0.05, -0.55],[-0.12, -0.55]],
                        [[0.1, -0.50],[-0.07, -0.45]],
                        [[0.1, -0.65],[-0.07, -0.40]],
                        [[0.05, -0.70],[-0.02, -0.70]],
                        [[0.5, 0.0],[-0.60, -0.2]],
                        ])
    env_matrixs = []
    for env_matrix in angles:
        for side in env_matrix:
            for angle in side:
                temp = []
                matrix = np.arange(0, angle, 0.10 if angle>=0 else -0.10)
                temp.extend(matrix)
                temp.extend([angle]*(7-len(temp)))
                env_matrixs.append(temp)
    # Instantiate environment
    env_matrixs = np.array(env_matrixs)
    env_matrixs = env_matrixs.reshape((5,2,2,7))
    spine_datas = []
    for dataDir, pedicle_point in zip(dataDirs, pedicle_points):
        dataDir = os.path.join(os.getcwd(), dataDir)
        pedicle_point_in_zyx = True #坐标是zyx形式吗？
        spine_datas.append(build_data_load(dataDir, pedicle_point, pedicle_point_in_zyx, input_z=64, input_y=80, input_x=160)) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
    '''---2 Build Environment  ---'''
    envs = []
    for spine_data in spine_datas:
        env = build_Env(spine_data, cfg['deg_threshold'], cfg)  # 只修改了初始化函数，其他函数待修改
        envs.append(env)
    for j in range(len(envs)):
        env = envs[j]
        _, o_3D = env.reset(random_reset = False)
        env_matrix = env_matrixs[j]
        left_matrix, right_matrix = env_matrix[0], env_matrix[1]
        for index in range(7):
            radian_L = np.array([left_matrix[0][index],left_matrix[1][index]])
            radian_R = np.array([right_matrix[0][index],right_matrix[1][index]])
            v, state_3D, state_matrix = env.simulate_volume(radian_L, radian_R, draw_screw=True)
            # radius, length = r, l
            max_radius_L, max_radius_R, line_len_L, line_len_R = state_matrix[0], state_matrix[1],state_matrix[2],state_matrix[3],
            print('movie_surface_621_L{}_{} max_radius_L, line_len_L, max_radius_R, line_len_R {}'.format(str(j+1), str(index+1), [max_radius_L, line_len_L, max_radius_R, line_len_R]))
            state3D_itk = sitk.GetImageFromArray(np.transpose(state_3D, (2, 1, 0)))
            sitk.WriteImage(state3D_itk, os.path.join(os.getcwd(),'movie_surface_621_L{}_{}.nii.gz'.format(str(j+1), str(index+1))))