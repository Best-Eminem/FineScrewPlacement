import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from utils import pedical_utils as utils3

import utils

class SpineEnv(gym.Env):
    def __init__(self, spineData):
        """
        Args:
            spineData:
                - mask_coards: Coordinate of every pixel. shape:(3,m,n,t)
                - mask_array: Segmentation results of spine, 1-in 0-out. shape:(slice, height, width)
                - pedicle_points: center points in pedicle (x,y,z). shape:(2,3)
        """
        # self.reset_opt = opts['reset']
        # self.step_opt = opts['step']

        self.mask_array= spineData['mask_array']
        self.mask_coards = spineData['mask_coards']
        self.pedicle_points = spineData['pedicle_points']
        self.centerPointL = self.pedicle_points[0]
        self.centerPointR = self.pedicle_points[1]
        self.action_num = 2 
        self.rotate_mag = [0.5, 0.5] # 直线旋转度数的量级 magtitude of rotation (δlatitude,δlongitude) of line
        self.reward_weight = [0.5, 0.5] # 计算每步reward的权重 weights for every kind of reward (), [line_delta, radius_delta] respectively
        self.degree_threshold = [-10., 10., -15., 15.] # 用于衡量终止情况的直线经纬度阈值 [minimum latitude, maximum latitude, minimum longitude, maximum longitude]

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
        
        self.state = None
        self.steps_beyond_done = None # 表示到停止的时候一共尝试了多少step

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computeInitDegree(self): # TODO: it still needs to refine after we can get more point
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
        return np.array([0.,0.])

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

    def reset(self):
        self.steps_beyond_done = None # 设置当前步数为None
        init_degree = self.computeInitDegree()
        init_cpoint = self.computeInitCPoint()
        dire_vector = utils3.coorLngLat2Space(init_degree) #螺钉方向向量
        self.dist_mat_point = utils3.spine2point(self.mask_coards, self.mask_array, init_cpoint)
        self.dist_mat_line = utils3.spine2line(self.mask_coards, self.mask_array, init_cpoint, dire_vector)
        self.pre_max_radius, self.pre_line_len, self.endpoints = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, init_cpoint, dire_vector, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point, line_dist=self.dist_mat_line)
        # self.state = np.concatenate([[self.pre_max_radius], init_degree, init_cpoint])
        state_list = [self.pre_max_radius, self.pre_line_len]
        state_list.extend(init_degree)
        self.state = np.asarray(state_list, dtype=np.float32)
        # self.state中存储的是degree，而送入网络时的是弧度
        state_ = self.state * 1.0
        # state_[1:3] = np.deg2rad(state_[1:3])
        state_[2:] = np.deg2rad(state_[2:])
        return np.asarray(state_, dtype=np.float32)

    def stepPhysics(self, delta_degree, delta_cpoint = None):
        # todo 如果选择弧度值，这里需要改变
        radius, length, degree = self.state[0], self.state[1], self.state[2:]
        result = {'d': degree + delta_degree,
                #   'p': [degree, cpoint + delta_cpoint],
                #   'dp': [degree + delta_degree, cpoint + delta_cpoint]
                  }
        return result['d']

    def getReward(self, state):
        #state [radius,length,degree]
        line_len = state[1]
        # 没入长度越长越好,reward基于上一次的状态来计算。
        len_delta = line_len - self.pre_line_len
        self.pre_line_len = line_len
        # 半径越大越好
        radius_delta = state[0] - self.pre_max_radius
        self.pre_max_radius = state[0]
        return len_delta, radius_delta

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # ------------------------------------------
        #Cast action to float to strip np trappings
        rotate_deg = self.rotate_mag * action[0:2]
        # move_cp = self.trans_mag * action[2:]
        # step forward
        this_degree = self.stepPhysics(rotate_deg, delta_cpoint = None)
        this_dirpoint = utils3.coorLngLat2Space(this_degree, R=1., default = True)
        self.dist_mat_point = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        self.dist_mat_line = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint)
        max_radius, line_len, self.endpoints = utils3.getLenRadiu(self.mask_coards, self.mask_array, self.centerPointL,
                                                                  this_dirpoint, R=1,
                                                                  line_thres=self.line_thres,
                                                                  radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point, line_dist=self.dist_mat_line)
        
        state_list = [max_radius, line_len]
        state_list.extend(this_degree)
        self.state = np.asarray(state_list, dtype=np.float32)
        if max_radius <= 0.: # todo 仍需要再思考
            line_len = 0.

        # Judge whether done
        done = self.state[0] < self.done_radius \
            or not (self.degree_threshold[0]<= self.state[2] <= self.degree_threshold[1]) \
            or not (self.degree_threshold[2]<= self.state[3] <= self.degree_threshold[3]) \
            # or not utils3.pointInSphere(self.state[3:], self.cp_threshold)
        done = bool(done)

        # -------------------------
        # Compute reward
        if not done:
            len_delta, radius_delta = self.getReward(self.state) #当前的reward的计算，均为针对上一步的，可考虑改为针对历史最优值来计算
            reward = self.weight[0] * len_delta + self.weight[1] * radius_delta
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            len_delta, radius_delta = self.getReward(self.state)
            reward = self.weight[0] * len_delta + self.weight[1] * radius_delta
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
                            You are calling 'step()' even though this environment has already returned
                            done = True. You should always call 'reset()' once you receive 'done = True'
                            Any further steps are undefined behavior.
                            """)
            self.steps_beyond_done += 1
            len_delta, radius_delta = self.getReward(self.state, line_len)
            reward = -1000.  
        state_ = self.state * 1.0
        # state_[1:3] = np.deg2rad(state_[1:3])
        state_[2:] = np.deg2rad(state_[2:])
        return np.asarray(state_, dtype=np.float32), reward, done, {'len_delta': len_delta, 'radius_delta': radius_delta}

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
        ax2.scatter(self.endpoints['radiu_p'][1], self.endpoints['radiu_p'][2], c='r')
        ax2.scatter(self.endpoints['start_point'][1], self.endpoints['start_point'][2], c='g')
        ax2.scatter(self.endpoints['end_point'][1], self.endpoints['end_point'][2], c='g')
        ax2.set_xlabel('Y-axis')
        ax2.set_ylabel('Z-axis')
        
        ax3 = fig.add_subplot(222)
        ax3.imshow(np.transpose(z_visual, (1, 0)))
        ax3.scatter(self.endpoints['radiu_p'][0], self.endpoints['radiu_p'][1], c='r')
        ax3.scatter(self.endpoints['start_point'][0], self.endpoints['start_point'][1], c='g')
        ax3.scatter(self.endpoints['end_point'][0], self.endpoints['end_point'][1], c='g')
        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')
        if info is not None:
            # ax2.text(2, -9, '#len_d:' + '%.4f' % info['len_delta'], color='red', fontsize=10)
            # ax2.text(2, -2, '#radius_d:' + '%.4f' % info['radiu_delta'], color='red', fontsize=10)
            # ax2.text(2, -18, '#action_x:' + '%.4f' % info['action'][2], color='red', fontsize=10)
            # ax2.text(2, -10, '#action_y:' + '%.4f' % info['action'][3], color='red', fontsize=10)
            # ax2.text(2, -2, '#action_z:' + '%.4f' % info['action'][4], color='red', fontsize=10)
            ax3.text(2, -30, '#Reward:' + '%.4f' % info['r'], color='red', fontsize=20)
            ax3.text(2, -2, '#TotalR:' + '%.4f' % info['reward'], color='red', fontsize=20)
            ax3.text(2, 110, '#frame:%.4d' % info['frame'], color='red', fontsize=20)
            ax2.text(2, 110, '#radius:' + '%.4f' % self.state[0], color='red', fontsize=20)
            ax2.text(2, 140, '#length:' + '%.4f' % self.state[1], color='red', fontsize=20)

        if is_save_gif:
            if info is not None:
                fig.savefig(img_save_path + '/Epoch%d_%d.jpg' % (info['epoch'], info['frame']))
            else: fig.savefig(img_save_path + '/Epoch_{}.jpg'.format('test'))

        if is_vis:
            plt.show()
            plt.pause(0.9)
            plt.ioff()
        return fig
