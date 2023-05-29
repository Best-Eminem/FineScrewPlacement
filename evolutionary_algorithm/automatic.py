import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
from env import SingleSpineEnvSingle
from dataLoad.loadNii import get_spinedata
from deap import algorithms, base, creator, tools
import time
import SimpleITK as sitk
import statistics

def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    return spine_data 

def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnvSingle.SpineEnv(spine_data, degree_threshold, **cfg)
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
                # r'spineData/sub-verse500_dir-ax_L1_ALL_msk.nii.gz',
                # r'spineData/sub-verse506_dir-iso_L1_ALL_msk.nii.gz',
                # r'spineData/sub-verse521_dir-ax_L1_ALL_msk.nii.gz',
                # r'spineData/sub-verse621_L1_ALL_msk.nii.gz',

                # r'spineData/sub-verse518_dir-ax_L2_ALL_msk.nii.gz',
                # r'spineData/sub-verse536_dir-ax_L2_ALL_msk.nii.gz',
                # r'spineData/sub-verse586_dir-iso_L2_ALL_msk.nii.gz',
                # r'spineData/sub-verse621_L2_ALL_msk.nii.gz',

                # r'spineData/sub-verse510_dir-ax_L3_ALL_msk.nii.gz',
                # r'spineData/sub-verse518_dir-ax_L3_ALL_msk.nii.gz',
                # r'spineData/sub-verse818_dir-ax_L3_ALL_msk.nii.gz',
                # r'spineData/sub-verse621_L3_ALL_msk.nii.gz',

                # r'spineData/sub-verse514_dir-ax_L4_ALL_msk.nii.gz',
                # r'spineData/sub-verse534_dir-iso_L4_ALL_msk.nii.gz',
                # r'spineData/sub-verse537_dir-iso_L4_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L4_ALL_msk.nii.gz',
                
                # r'spineData/sub-verse505_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse510_dir-ax_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse614_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse621_L5_ALL_msk.nii.gz',
                ]
    pedicle_points = np.asarray([
                                # [[39,49,58],[39,48,105]],
                                # [[38,43,67],[38,43,108]],
                                # [[30,46,65],[30,46,108]],
                                # [[35,47,65],[36,47,105]],
                                
                                # [[33,42,64],[37,44,103]],
                                # [[33,40,57],[31,45,96]],
                                # [[33,43,66],[36,43,101]],
                                # [[36,48,62],[38,48,102]],
                                 
                                # [[33,44,67],[33,42,101]],
                                # [[33,43,59],[38,45,101]],
                                # [[33,47,61],[36,46,108]],
                                # [[38,47,62],[39,47,104]],
                                 
                                # [[59,45,60],[51,44,109]],
                                # [[35,43,63],[33,46,105]],
                                # [[46,44,63],[46,44,101]],
                                # [[43,48,60],[44,48,107]],
                                 
                                # [[34,43,61],[34,41,102]],
                                # [[45,52,68],[45,43,110]],
                                # [[42,45,64],[40,44,113]],
                                [[48,52,60],[46,51,122]],
                                 ])
    # Instantiate environment
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
    L5 = envs[-1]
    _, o_3D = L5.reset(random_reset = False)
    radian_Ls = np.array([0.00, -0.0])
    v, state_3D, r, l = L5.simulate_volume(radian_Ls)
    print(v,r,l)
    # for j in range(len(envs)):
    #     means=[]
    #     bestV = 0
    #     radius, length = 0,0
    #     best_angle = None
    #     best_state3D = None
    #     T1 = time.time()
    #     env = envs[j]
    #     _, o_3D = env.reset(random_reset = False)
    #     horizon_range = np.arange(0,0.52,0.05) if screw_index == 0 else np.arange(-0.52,0,0.05)
    #     for i in horizon_range:
    #         for k in np.arange(-1.0,0.2,0.05):
    #             radian_Ls = np.array([i,k])
    #             v, state_3D, r, l = env.simulate_volume(radian_Ls)
    #             if v>bestV:
    #                 bestV = v
    #                 best_angle = [i,k]
    #                 best_state3D = state_3D
    #                 radius, length = r, l
    #             means.append(v)
    #     print('--------------',dataDirs[j],'-----begin-----')
    #     print('最优值:%.2f' % bestV)
    #     print('最优角度:', best_angle)
    #     print('半径，长度:', radius, length)
    #     T2 = time.time()
    #     print('程序运行时间:%s秒' % ((T2 - T1)))
    #     print('--------------',dataDirs[j],'-----end-------')
    #     print("\n")
    #     state3D_itk = sitk.GetImageFromArray(np.transpose(best_state3D, (2, 1, 0)))
        # sitk.WriteImage(state3D_itk, os.path.join(os.getcwd(),'surface_621_L{}_{}.nii.gz'.format(str(j+1), 'left' if screw_index==0 else 'right')))