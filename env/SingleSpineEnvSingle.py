from copy import deepcopy
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from utils import pedical_utils as utils3
import torch

import utils

class SpineEnv(gym.Env):
    def __init__(self, spineData, degree_threshold, **opts):
        """
        Args:
            spineData:
                - mask_coards: Coordinate of every pixel. shape:(3,m,n,t)
                - mask_array: Segmentation results of spine, 1-in 0-out. shape:(slice, height, width)
                - pedicle_points: center points in pedicle (x,y,z). shape:(2,3)
        """
        self.reset_opt = opts['reset']
        self.step_opt = opts['step']
        self.screw_index = opts['screw_index']
        self.state_shape = self.reset_opt['state_shape']
        self.discrete_action = self.step_opt['discrete_action']
        self.mask_array= spineData['mask_array']
        self.mask_coards = spineData['mask_coards']
        self.pedicle_points = spineData['pedicle_points']
        self.centerPointL = self.pedicle_points[self.screw_index]
        self.action_num = 2 #δhorizon,δverticle
        self.rotate_mag = self.step_opt['rotate_mag'] # 直线旋转度数的量级 magtitude of rotation (δhorizon,δverticle) of line
        self.reward_weight = [0.1, 0.1] # 计算每步reward的权重 weights for every kind of reward (), [line_delta, radius_delta] respectively
        self.degree_threshold = degree_threshold # 用于衡量终止情况的直线经纬度阈值 [minimum latitude, maximum latitude, minimum longitude, maximum longitude]

        self.min_action = -1.0 # threshold for sum of policy_net output and random exploration
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(self.action_num,)) #用来检查动作的取值范围
        # self.trans_mag = np.array(self.step_opt.trans_mag) # 定点的移动尺度范围
        self.rotate_mag = np.array(self.rotate_mag) # 旋转的尺度范围, 两个方向
        self.weight = np.array(self.reward_weight)
        self.degree_threshold = np.array(self.degree_threshold)
        self.radiu_thres = None #[self.step_opt.radiu_thres[0] + spineData.cpoint_l[0], self.step_opt.radiu_thres[1] +spineData.cpoint_l[0]]
        self.line_thres =  None #self.step_opt.line_thres
        self.done_radius = 0.5 #医学中允许的置钉最小半径（用于衡量是否为终止状态） allows minimum radius
        
        dist = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        self.cp_threshold = (self.centerPointL, np.min(dist)-1) # The position of the allowed points, represented as (center of sphere, radius)
        self.seed()
        
        self.state_matrix = None
        self.steps_before_done = None # 表示到停止的时候一共尝试了多少step

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computeInitDegree(self,random_reset=True): # TODO: it still needs to refine after we can get more point
        # 初始化倾斜角度，添加噪声
        # if self.reset_opt.initdegree:
        #     deg = self.reset_opt.initdegree
        # else:
        #     deg = [0.,0.]
        # if self.reset_opt.is_rand_d:
        #     return np.array(deg) + self.np_random.uniform(
        #         low = self.reset_opt.rdrange[0], high = self.reset_opt.rdrange[1], size = [2,])
        # else:
        #     return np.array(deg)
        # return np.array([0.,0.])
        if random_reset:
            # return self.np_random.uniform(low = self.degree_threshold[0], high = self.degree_threshold[1], size = [1,])
            horizon = self.np_random.uniform(low = self.degree_threshold[0], high = self.degree_threshold[1], size = [1,])[0]
            sagtal = self.np_random.uniform(low = self.degree_threshold[2], high = self.degree_threshold[3], size = [1,])[0]
            return np.array([horizon,sagtal])
        else: return np.array([0.,0.])

    def computeInitCPoint(self): # TODO: it still needs to refine after we can get more point
        # 初始化定点位置，添加噪声
        # if self.reset_opt.initpoint:
        #     cpoint = self.reset_opt.initpoint
        # else:
        #     cpoint = self.centerPointL
        # if self.reset_opt.is_rand_p:
        #     return cpoint + self.np_random.uniform(
        #         low = self.reset_opt.rprange[0], high = self.reset_opt.rprange[1], size = [3,])
        # else:
        #     return np.array(cpoint)
        return self.centerPointL

    def normalize(self, array):
        mu = np.mean(array)
        std = np.std(array)
        return (array-mu) / std
    
    def reset(self, random_reset=False):
        self.steps_before_done = None # 设置当前步数为None
        init_degree_L = self.computeInitDegree(random_reset)
        init_cpoint_L = self.computeInitCPoint()
        dire_vector_L = utils3.coorLngLat2Space(init_degree_L, None, default=True) #螺钉方向向量
        self.dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, init_cpoint_L)
        self.dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, init_cpoint_L, dire_vector_L)
        self.pre_max_radius_L, self.pre_line_len_L, self.endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, init_cpoint_L, dire_vector_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_L, line_dist=self.dist_mat_line_L)
        state_3D = self.draw_state(self.endpoints_L)
        state_list = [self.pre_max_radius_L, self.pre_line_len_L]
        state_list.extend(init_degree_L)
        self.state_matrix = np.asarray(state_list, dtype = np.float32)
        # self.state_matrix中存储的是degree，而送入网络时的是弧度
        state_ = self.state_matrix * 1.0
        return np.asarray(state_, dtype=np.float32), self.normalize(state_3D)

    def stepPhysics(self, state_matrix, delta_degree_L, delta_cpoint = None):
        # todo 如果选择弧度值，这里需要改变
        radius_L, length_L, degree_L = state_matrix[0], state_matrix[1], state_matrix[2:4]
        result = {'d_L': degree_L + delta_degree_L,
                #   'p': [degree, cpoint + delta_cpoint],
                #   'dp': [degree + delta_degree, cpoint + delta_cpoint]
                  }
        return result['d_L']

    def getReward(self, state):
        #state [radius,length,degree]
        line_len_L = state[1]
        # 没入长度越长越好,reward基于上一次的状态来计算。
        len_delta_L = line_len_L - self.pre_line_len_L
        self.pre_line_len_L = line_len_L
        # 半径越大越好
        radius_delta_L = state[0] - self.pre_max_radius_L
        self.pre_max_radius_L = state[0]
        return len_delta_L, radius_delta_L

    def step(self, action_L):
        if self.discrete_action:
            discrete_vec = np.array([-0.25,-0.2,-0.15,-0.1,-0.05,.0,0.05,0.1,0.15,0.2,0.25])
            # discrete_vec = np.array([-0.05,.0,0.05])
            # discrete_vec = np.array([-0.1,-0.05,.0,0.05,0.1])
            discrete_vec = np.rad2deg(discrete_vec)
            rotate_deg_L = np.array([discrete_vec[action_L[0]], discrete_vec[action_L[1]]])
            # rotate_deg_L = np.array([.0, discrete_vec[action_L[1]]])
        else:
            # action_L = np.clip(action_L, -1.0, 1.0)
            rotate_deg_L = self.rotate_mag * action_L[0:2]
        this_degree_L = self.stepPhysics(self.state_matrix, rotate_deg_L, delta_cpoint = None)
        this_dirpoint_L = utils3.coorLngLat2Space(this_degree_L, angles_R=None, R=1., default=True)
        self.dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        self.dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        max_radius_L, line_len_L, self.endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_L, line_dist=self.dist_mat_line_L)

        state_list = [max_radius_L, line_len_L]
        state_list.extend(this_degree_L)
        state_3D = self.draw_state(self.endpoints_L)
        self.state3D_array = deepcopy(state_3D)
        self.state_matrix = np.asarray(state_list, dtype=np.float32)
        # self.state_matrix中存储的是degree，而送入网络时的是弧度
        if max_radius_L < 0.: # todo 仍需要再思考
            line_len_L = 0.01
            self.state_matrix[1] = 0.01

        # Judge whether done
        done_L = self.state_matrix[0] < self.done_radius \
            or not (self.degree_threshold[0]<= self.state_matrix[2] <= self.degree_threshold[1]) \
            or not (self.degree_threshold[2]<= self.state_matrix[3] <= self.degree_threshold[3]) \
            # or not utils3.pointInSphere(self.state_matrix[3:], self.cp_threshold)
        done_L = bool(done_L)
        done = done_L
        # pre_volume = (3.14*(self.pre_line_len_L*self.pre_max_radius_L*self.pre_max_radius_L))
        pre_volume = 2*self.pre_line_len_L*self.pre_max_radius_L
        len_delta_L, radius_delta_L = self.getReward(self.state_matrix)
        # reward为当前螺钉长度减去一个base值
        # reward = (self.state_matrix[2] + self.state_matrix[3])/ (70*2) 
        # now_volume = (3.14*(self.state_matrix[1]*self.state_matrix[0]*self.state_matrix[0]))
        now_volume = 2*self.state_matrix[1]*self.state_matrix[0]
        # reward = np.log(now_volume) - np.log(pre_volume) #reward为体积差
        delta_volume = (now_volume - pre_volume)/1000
        # reward = now_volume/10000.0 #reward为体积差
        reward = now_volume/1000.0#reward为体积差
        # reward = self.state_matrix[1]/70.0
        state_ = self.state_matrix * 1.0
        return np.asarray(state_, dtype=np.float32), reward, done, \
            {'len_delta_L': len_delta_L, 'radius_delta_L': radius_delta_L, 'action_left':rotate_deg_L, 'delta_volume':delta_volume}, self.normalize(state_3D)

    def render_(self, fig, info=None, is_vis=False, is_save_gif=False, img_save_path=None, **kwargs):
        # fig = plt.figure()
        if is_vis:
            plt.ion()
        visual_ = self.mask_array #+ np.where(self.dist_mat <= 1.2, 2, 0)
        x_visual = np.max(visual_[:, :, :], 0)
        z_visual = np.max(visual_[:, :, :], 2)

        plt.clf()
        ax2 = fig.add_subplot(221)
        ax2.imshow(np.transpose(x_visual, (1, 0)))
        ax2.scatter(self.endpoints_L['radiu_p'][1], self.endpoints_L['radiu_p'][2], c='r')
        ax2.scatter(self.endpoints_L['start_point'][1], self.endpoints_L['start_point'][2], c='g')
        ax2.scatter(self.endpoints_L['end_point'][1], self.endpoints_L['end_point'][2], c='g')
        ax2.scatter(self.endpoints_L['cpoint'][1], self.endpoints_L['cpoint'][2], c='g')

        ax2.set_xlabel('Y-axis')
        ax2.set_ylabel('Z-axis')
        ax2.invert_yaxis()
        
        ax3 = fig.add_subplot(222)
        ax3.imshow(np.transpose(z_visual, (1, 0)))
        ax3.scatter(self.endpoints_L['radiu_p'][0], self.endpoints_L['radiu_p'][1], c='r')
        ax3.scatter(self.endpoints_L['start_point'][0], self.endpoints_L['start_point'][1], c='g')
        ax3.scatter(self.endpoints_L['end_point'][0], self.endpoints_L['end_point'][1], c='g')
        ax3.scatter(self.endpoints_L['cpoint'][0], self.endpoints_L['cpoint'][1], c='g')

        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')
        ax3.invert_yaxis()
        if info is not None:
            ax3.text(2, -60, '#Reward:' + '%.4f' % info['r'], color='red', fontsize=20)
            ax3.text(2, -90, '#TotalR:' + '%.4f' % info['reward'], color='red', fontsize=20)
            ax3.text(2, -120, '#action_L:' + '%s' % info['action_left'], color='red', fontsize=15)
            ax3.text(2, 110, '#frame:%.4d' % info['frame'], color='red', fontsize=20)
            ax2.text(2, -30, '#radius:%.2f' % (self.state_matrix[0]), color='red', fontsize=15)
            ax2.text(2, -45, '#length:%.2f' % (self.state_matrix[1]), color='red', fontsize=15)
            # ax2.text(2, -60, '#volume_L:%.1f' % (3.14*self.state_matrix[1]*self.state_matrix[0]*self.state_matrix[0]), color='red', fontsize=15)
            ax2.text(2, -60, '#surface_L:%.1f' % (2.0*self.state_matrix[1]*self.state_matrix[0]), color='red', fontsize=15)
            ax2.text(2, -75, '#angle_L:%.1f, %.1f' % (self.state_matrix[2], self.state_matrix[3]), color='red', fontsize=15)
        if is_save_gif:
            if info is not None:
                fig.savefig(img_save_path + '/Epoch%d_%d.jpg' % (info['epoch'], info['frame']))
            else: fig.savefig(img_save_path + '/Epoch_{}.jpg'.format('test'))

        if is_vis:
            plt.show()
            plt.pause(0.9)
            plt.ioff()
        return fig

    def draw_state(self, endpoints_L):
        state_3D = deepcopy(self.mask_array)
        state_3D[endpoints_L['radiu_p']] = 2
        return np.array(state_3D, dtype=np.float32)

    def simulate_reward(self, radian_L, base_state):
        this_degree_L = np.rad2deg(radian_L)
        this_dirpoint_L = utils3.coorLngLat2Space(this_degree_L, angles_R=None, R=1., default = True)
        dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        max_radius_L, line_len_L, endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_L, line_dist=dist_mat_line_L)

        state_list = [max_radius_L, line_len_L]
        state_list.extend(this_degree_L)
        state_matrix = np.asarray(state_list, dtype=np.float32)
        # self.state_matrix中存储的是degree，而送入网络时的是弧度
        if max_radius_L < 0.: # todo 仍需要再思考
            line_len_L = 0.01
            state_matrix[1] = 0.01
        pre_max_radius_L, pre_line_len_L = base_state[0],base_state[1]
        # pre_volume = (3.14*(pre_line_len_L*pre_max_radius_L*pre_max_radius_L))
        now_volume = (3.14*(state_matrix[1]*state_matrix[0]*state_matrix[0]))
        # reward = np.log(now_volume) - np.log(pre_volume) #reward为体积差
        reward = now_volume/10000.0 #reward为体积差
        # reward = state_matrix[1]/70.0
        return reward
    
    def simulate_volume(self, radian_L):
        this_degree_L = np.rad2deg(radian_L)
        this_dirpoint_L = utils3.coorLngLat2Space(this_degree_L, angles_R=None, R=1., default = True)
        dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        max_radius_L, line_len_L, endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_L, line_dist=dist_mat_line_L)

        state_list = [max_radius_L, line_len_L]
        state_list.extend(this_degree_L)
        state_matrix = np.asarray(state_list, dtype=np.float32)
        # self.state_matrix中存储的是degree，而送入网络时的是弧度
        if max_radius_L < 0.: # todo 仍需要再思考
            line_len_L = 0.01
            state_matrix[1] = 0.01
        # now_volume = (3.14*(state_matrix[1]*state_matrix[0]*state_matrix[0])) 
        now_volume = (2*3.14*(state_matrix[1]*state_matrix[0])) 
         
        state3D_array = self.draw_state(endpoints_L)
        return now_volume, state3D_array, state_matrix[0], state_matrix[1]
