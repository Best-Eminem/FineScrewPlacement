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

# 定义自定义函数，这里以 2 变量函数为例
def my_func(x1, y1,env):
    radian_L = np.array([x1,y1])
    # radian_R = np.array([x2,y2])
    result, _,_,_ = env.simulate_volume(radian_L)
    return result

def evaluate(individual_h, env):
    # 计算自定义函数的值
    x1, y1 = individual_h
    x1 = max(-0.53, min(x1, 0.53))
    y1 = max(-1.1, min(y1, 1.1))
    return (my_func(x1, y1,env),)

def get_random_horizon():
    horizon = random.uniform(-0.4, 0.4)
    while(horizon < -0.4 or horizon > 0.4):
        horizon = random.uniform(-0.4, 0.4)
    return horizon

def get_random_sagittal():
    sagittal = random.uniform(-1.0, 1.0)
    while(sagittal < -1.0 or sagittal > 1.0):
        sagittal = random.uniform(-1.0, 1.0)
    return sagittal

if __name__ == '__main__':
    screw_index = 0
    cfg = {'deg_threshold':[-30., 30., -60., 60.],#[-65., 65., -45., 25.],
           'screw_index':screw_index,
           'reset':{'rdrange':[-45, 45],
                    'state_shape':(160, 80, 64),},
           'step':{'rotate_mag':[10, 10], 'discrete_action':False}
           }

    print(cfg)
    dataDirs = [
                # r'spineData/sub-verse500_dir-ax_L1_ALL_msk.nii.gz',
                # r'spineData/sub-verse506_dir-iso_L1_ALL_msk.nii.gz',
                # r'spineData/sub-verse521_dir-ax_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L1_ALL_msk.nii.gz',

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
                # r'spineData/sub-verse621_L4_ALL_msk.nii.gz',
                
                # r'spineData/sub-verse505_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse510_dir-ax_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse614_L5_ALL_msk.nii.gz',
                # r'spineData/sub-verse621_L5_ALL_msk.nii.gz',
                ]
    pedicle_points = np.asarray([
                                # [[39,49,58],[39,48,105]],
                                # [[38,43,67],[38,43,108]],
                                # [[30,46,65],[30,46,108]],
                                [[35,47,65],[36,47,105]],
                                
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
                                # [[48,52,60],[46,51,122]],
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
    # radian_Ls = [
    #             [0.065,-0.567],
    #             [-0.056,-0.627],
    #             [0.100,-0.693,],
    #             [0.087,-0.358],
    #             [0.200,0.048],
    #             ]
    # radian_Rs = [
    #             [-0.316,-0.295],
    #             [-0.217,-0.216],
    #             [-0.033,0.095],
    #             [-0.542,0.198],
    #             [-0.200,0.068],
    #             ]
    radian_Ls = [
                [0.,-0.],
                [-0.,-0.],
                [0.,-0.,],
                [0.,-0.],
                [0.,0.],
                ]
    radian_Rs = [
                [-0.,-0.],
                [-0.,-0.],
                [-0.,0.],
                [-0.,0.],
                [-0.,0.],
                ]
    gene = True
    if gene:
        for j in range(len(envs)):
            means=[]
            T1 = time.time()
            for i in range(1):
                env = envs[j]
                _, o_3D = env.reset(random_reset = True)
                # 定义优化问题
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                toolbox = base.Toolbox()

                # 定义遗传算法的参数
                # toolbox.register("horizon", random.uniform, -0.53, 0.53, bounds=(-0.53, 0.53))
                # toolbox.register("sagittal", random.uniform, -1.1, 1.1, bounds=(-1.1, 1.1))
                toolbox.register("horizon", get_random_horizon)
                toolbox.register("sagittal", get_random_sagittal)
                toolbox.register("individual", tools.initCycle, creator.Individual, 
                                (toolbox.horizon, toolbox.sagittal), n=1)

                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("evaluate", evaluate, env=env)
                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
                toolbox.register("select", tools.selTournament, tournsize=3)#锦标赛选择

                # 运行遗传算法
                pop = toolbox.population(n=200)
                algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=200, verbose=False)

                # 输出结果
                best_ind = tools.selBest(pop, k=1)[0]
                means.append(best_ind.fitness.values[0])
            print(means)
            # mean = statistics.mean(means)
            # std = statistics.stdev(means)
            print('--------------',dataDirs[j],'-----begin-----')
            print('最优解:%.3f,%.3f' % (best_ind[0],best_ind[1]))
            print('最优值:%.3f' % best_ind.fitness.values[0])
            # print('平均最优值:%.2f' % mean)
            # print('方差:%.2f' % std)
            T2 = time.time()
            print('程序运行时间:%s秒' % ((T2 - T1)))
            print('--------------',dataDirs[j],'-----end-------')
            print("\n")
            
    else:
        volumes = []
        for i in range(len(envs)):
            radian_L = np.array(radian_Ls[i])
            radian_R = np.array(radian_Rs[i])
            env = envs[i]
            volume, state_3D = env.simulate_volume(radian_L, radian_R, draw_screw = True)
            volumes.append(volume)
            state3D_itk = sitk.GetImageFromArray(np.transpose(state_3D, (2, 1, 0)))
            sitk.WriteImage(state3D_itk, os.path.join(os.getcwd(),'standard_621_L{}.nii.gz'.format(str(i+1))))
        print(volumes)