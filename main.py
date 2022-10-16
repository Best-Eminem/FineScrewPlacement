import cv2
import os
import time
import random
import argparse
import logging
import json
from gym import Env
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple, deque
from config import cfg
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from dataLoad.loadNii import get_spinedata
from env import SingleSpineEnv
from networks import get_p_net, get_q_net
from utils.model_utils import load_pretrain, restore_from, disable_gradient, enable_gradient, copy_net
from utils.average_meter import AverageMeter
from utils.log_utils import init_log, print_speed, add_file_handler
from utils.img_utils import images_to_video

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='Spinal Nailing')
parser.add_argument('--cfg', type=str, default = None,
                    help='configuration files of training')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Functions to build
def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    logger.info("-------------------build spine data----------------------")
    spine_data, spacing = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    # cfg.Env.step.line_rd = float(max(cfg.Env.step.line_rd, spacing)) # 根据spacing修改计算直线长度时的直线半径。
    logger.info("-------------------build data done-----------------------")
    return spine_data # namedtuple:

def build_Env(spine_data):
    logger.info("-------------------build env-----------------------------")
    env = SingleSpineEnv.SpineEnv(spine_data)
    logger.info("-------------------build env done------------------------")
    return env

def build_exper_pool(capacity = 1e6):
    """build experience pool

    Args:
        capacity (_int_, optional): _capacity of experience pool_. Defaults to 1e6.

    Returns:
        _ReplayMemory_: _deque(双端队列)_
    """
    logger.info("build experience pool")
    Experience = namedtuple('Experience', ('state', 'state_3D', 'action', 'reward', 'next_state', 'next_state_3D', 'terminal'))
    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque(maxlen=capacity)

        def push(self, *args):
            self.memory.append(Experience(*args))  ## append a new experience

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def sample_train(self,batch_size):
            experiences = self.sample(batch_size)
            experiences_batch = Experience(*zip(*experiences))
            state_batch = torch.stack(experiences_batch.state)
            state_3D_batch = torch.stack(experiences_batch.state_3D)
            action_batch = torch.stack(experiences_batch.action)
            reward_batch = torch.stack(experiences_batch.reward)
            next_state_batch = torch.stack(experiences_batch.next_state)
            next_state_3D_batch = torch.stack(experiences_batch.next_state_3D)
            terminal_batch = torch.stack(experiences_batch.terminal)
            state_action_batch = torch.cat((state_batch, action_batch), dim=1)
            # state_3d_next_state_3D_batch = torch.cat((state_3D_batch, next_state_3D_batch), dim=1)
            return state_batch, state_3D_batch, action_batch, reward_batch, next_state_batch, next_state_3D_batch, terminal_batch, state_action_batch #, state_3d_next_state_3D_batch

        def __len__(self):  ## len(experience)
            return len(self.memory)
    experience_pool = ReplayMemory(int(capacity))
    logger.info("build pool done")
    return experience_pool

def build_nets(pnet_pretrained, mode='train'):
    if mode == 'train':
        logger.info("build pnet and qnet")
        policy_net = get_p_net('conv0',in_channel=1,out_channel=2)
        q_net = get_q_net('conv0',in_channel=1,out_channel=1)
        if pnet_pretrained:
            logger.info("policy_net load pretrained model from {}".format(pnet_pretrained))
            load_pretrain(policy_net, pnet_pretrained)
            logger.info("load model done")
        target_p_net = deepcopy(policy_net)
        target_q_net = deepcopy(q_net)
        disable_gradient(target_p_net)
        disable_gradient(target_q_net)
        logger.info("build net done")
        return policy_net, q_net, target_p_net, target_q_net
    else:
        logger.info("build pnet")
        policy_net = get_p_net('conv0',in_channel=1,out_channel=2)
        if pnet_pretrained:
            logger.info("policy_net load pretrained model from {}".format(pnet_pretrained))
            load_pretrain(policy_net, pnet_pretrained)
            logger.info("load model done")
        logger.info("build net done")
        return policy_net

def build_opt_lr(pnet, qnet):
    logger.info('build optimizer')
    optimizer_p = optim.Adam(pnet.parameters(), lr = 1e-3)
    optimizer_q = optim.Adam(qnet.parameters(), lr = 1e-3)
    # lr_scheduler # TODO: add the learning rate scheduler to adjust the learning rate
    logger.info('build optimizer done')
    return optimizer_p, optimizer_q

def policy_action(state, env, policy_net): # state is tensor
    with torch.no_grad():
        output = policy_net(state)[0].cpu()
        action = env.action_space.high[0] * output
    return action

def explore_one_step(env, state, state_3D, experience_pool, policy_net):
    if len(state_3D.shape)==3:
        state_3D.unsqueeze_(0)
    def explore_action():
        with torch.no_grad():
            input = torch.unsqueeze(state_3D, 0)
            input = input.to(device=device)
            output = policy_net(input)[0].cpu().detach()
            action = env.action_space.high[0] * output
            action = torch.normal(action, cfg.Train.EXPLORE_NOISE) #从给定参数means,std的离散正态分布中抽取随机数
            action = torch.clamp(action, min=env.action_space.low[0], max=env.action_space.high[0])
        return action
    action = explore_action()
    _state, r, done, other, _state_3D = env.step(action.numpy()) #return state_, reward, done, {'len_delta': len_delta, 'radius_delta': radius_delta}, state_3D
    reward = torch.tensor(r, dtype=torch.float)  # r
    # next_state = torch.tensor(np.concatenate(np.asarray(_state)), dtype=torch.float)  # s'
    next_state = torch.tensor(_state, dtype=torch.float)
    next_state_3d = torch.tensor(_state_3D, dtype=torch.float)
    if len(next_state_3d.shape) == 3:
        next_state_3d.unsqueeze_(0)
    # next_state_3d.unsqueeze_(0)
    terminal = torch.tensor(int(done) * 1.0, dtype=torch.float)  # t
    # Store the transition in experience pool
    experience_pool.push(state, state_3D, action, reward, next_state, next_state_3d, terminal)  # (s,s_3d,a,r,s',s'3d,t), tensors
    return done, next_state, next_state_3d, r

def update_q_net(env, q_net, target_p_net,target_q_net, optimizer_q, r, s_3d, a, ns, ns_3d, d):
    s_3d = s_3d.to(device=device)
    a = a.to(device=device)
    ns_3d = ns_3d.to(device=device)
    r = r.to(device=device)
    d = d.to(device=device)
    curr_q_value = q_net(s_3d, a).squeeze()
    next_action = target_p_net(ns_3d)
    # nns_3d = env.simulate_step_batch(ns, next_action)
    # next_s_3d_ns_3d = torch.cat((ns_3d, nns_3d), dim=1)
    target_next_q_value = target_q_net(ns_3d, next_action).squeeze()
    target_q_value = r + cfg.Train.GAMMA * target_next_q_value * (1 - d)
    # mean square loss
    loss = torch.nn.MSELoss()(curr_q_value, target_q_value)
    # Optimize the model
    optimizer_q.zero_grad()
    loss.backward()
    optimizer_q.step()
    return loss.item()

def update_policy_net(env, policy_net, q_net, optimizer_p, s_3d, s):
    s_3d = s_3d.to(device=device)
    curr_action = policy_net(s_3d)
    # ns_3d = env.simulate_step_batch(s, curr_action)
    # curr_s_3d_ns_3d = torch.cat((s_3d, ns_3d), dim=1)
    ## using q network
    disable_gradient(q_net)
    loss = -1.0 * torch.mean(q_net(s_3d, curr_action))
    # Optimize the model
    optimizer_p.zero_grad()
    loss.backward()
    optimizer_p.step()
    enable_gradient(q_net)
    return loss.item()
# 过估计趋势，
# DDPG，D3算法。
def evaluate(env, policy_net, epoch):
    state, state_3D = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    state_3D = torch.tensor(state_3D, dtype=torch.float32)
    reward = 0
    frame = 0
    fig = plt.figure()
    while frame < cfg.Evaluate.steps_threshold:
        if len(state_3D.shape)==3:
            state_3D.unsqueeze_(0)
            state_3D.unsqueeze_(0)
        state_3D = state_3D.to(device=device)
        frame = frame + 1
        action = policy_action(state_3D, env, policy_net).numpy()
        next_state, r, done, others, next_state_3D = env.step(action)
        reward = reward + r
        state_3D = torch.tensor(next_state_3D, dtype=torch.float)
        if len(state_3D.shape)==3:
            state_3D.unsqueeze_(0)
            state_3D.unsqueeze_(0)
        state_3D = state_3D.to(device=device)
        action = policy_action(state_3D, env, policy_net).numpy()
        info = {'reward': reward, 'r': r, 'len_delta': others['len_delta'], 'radiu_delta': others['radius_delta'],
                'epoch': epoch, 'frame': frame, 'action': action}

        fig = env.render_(fig, info, **cfg.Evaluate)

        if done or frame == cfg.Evaluate.steps_threshold:
            if cfg.Evaluate.is_save_gif:
                images_to_video(cfg.Evaluate.img_save_path, '*.jpg', isDelete=True, savename = 'Epoch%d'%(epoch))
            break

def train(env, policy_net, q_net, target_p_net, target_q_net, experience_pool, optimizer_q, optimizer_p, tb_writer):
    average_meter = AverageMeter()
    start_epoch = 0
    if not os.path.exists('./snapshot/'):
        os.makedirs('./snapshot/')
    end = time.time()
    iter_num = 0
    for epoch in tqdm(range(start_epoch, cfg.Train.EPOCHS)):
        explore_steps = 0
        reward = 0
        # Initialize the environment and state
        state, state_3D = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state_3D = torch.tensor(state_3D, dtype=torch.float32)
        while explore_steps < cfg.Train.EPOCH_STEPS:
            explore_steps += 1
            done, next_state, next_state_3d, r = explore_one_step(env, state, state_3D, experience_pool, policy_net)
            # fig = plt.figure()
            # fig = env.render_(fig, None, **cfg.Evaluate)
            state = next_state
            state_3d = next_state_3d
            reward += r #
            # perfrom one step of the optimization
            WARM_UP_SIZE = 50
            if len(experience_pool) > WARM_UP_SIZE:
                s, s_3d, a, r, ns, ns_3d, d, sa = experience_pool.sample_train(cfg.Train.BATCH_SIZE) #state_batch, state_3D_batch, action_batch, reward_batch, next_state_batch, next_state_3D_batch, terminal_batch, state_action_batch
                loss_q = update_q_net(env, q_net, target_p_net, target_q_net, optimizer_q, r, s_3d, a, ns, ns_3d, d)
                loss_p = update_policy_net(env, policy_net, q_net, optimizer_p, s_3d, s)
                iter_num += 1
                if epoch % cfg.Train.UPDATE_INTERVAL == 0:
                    copy_net(policy_net, target_p_net, cfg.Train.UPDATE_WEIGHT)
                    copy_net(q_net, target_q_net, cfg.Train.UPDATE_WEIGHT)

            if done:
                break # one episode

        # Save and evaluate model
        if epoch % cfg.Train.SAVE_INTERVAL == 0 and len(experience_pool) > cfg.Train.WARM_UP_SIZE:
            # torch.save({'epoch': epoch,
            #             'pnet_dict': policy_net.state_dict(),
            #             'qnet_dict': q_net.state_dict(),
            #             'optimizer_p': optimizer_p.state_dict(),
            #             'optimizer_q': optimizer_q.state_dict()}, cfg.Train.SNAPSHOT_DIR+'\\checkpoint_e%d.pth' % (epoch))
            torch.save(policy_net.state_dict(), cfg.Train.SNAPSHOT_DIR+'/policy_model%d.pth' % (epoch))
            evaluate(env, policy_net, epoch) # TODO

        # Estimate time and show loss
        if len(experience_pool) > cfg.Train.WARM_UP_SIZE:
            epoch_time = time.time() - end
            epoch_info = {}
            epoch_info['epoch_time'] = epoch_time
            epoch_info['loss_p'] = loss_p
            epoch_info['loss_q'] = loss_q
            epoch_info['reward'] = reward
            average_meter.update(**epoch_info)

            for k, v in epoch_info.items():
                tb_writer.add_scalar(k, v, epoch)
            if epoch % cfg.Train.SAVE_INTERVAL == 0:
                info = "Epoch: [{}/{}]\n".format(epoch + 1, cfg.Train.EPOCHS)
                for cc, (k, v) in enumerate(epoch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                            getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                            getattr(average_meter, k))
                logger.info(info)
                print_speed(epoch, average_meter.epoch_time.avg, cfg.Train.EPOCHS)
        end = time.time()

def evaluateOthers():
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
        cfg.Evaluate.img_save_path = cfg.Evaluate.img_save_path + cfg.Version
        cfg.Train.LOG_DIR = cfg.Train.LOG_DIR + cfg.Version
        cfg.Train.SNAPSHOT_DIR = cfg.Train.SNAPSHOT_DIR + cfg.Version
        cfg.qnet.in_channels = cfg.Env.step.action_num + cfg.Env.step.state_num
    if not os.path.exists('evaluate_results') and cfg.Evaluate.is_save_gif:
        os.makedirs('evaluate_results')
    #####-------注意，本例中，mask_array与坐标的排列方式均采用x,y,z形式来计算，zyx形式的要转换为xyz形式-----------------##### 
    '''---1 Data Preprocess    ---'''
    dataDir = r'./spineData/sub-verse821_L2_seg-vert_msk.nii.gz'
    pedicle_points = np.asarray([[27,66,63],[27,67,111]])
    pedicle_points_in_zyx = True #坐标是zyx形式吗？
    spine_data = build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
    '''---2 Build Environment  ---'''
    env = build_Env(spine_data)  # 只修改了初始化函数，其他函数待修改
    '''---3 Build Networks     ---'''
    pnet_pretrained = None
    policy_net = build_nets(pnet_pretrained, mode='test')
    '''---4 Resume networks and optimizers ---'''
    Train_RESUME = 'snapshot/checkpoint_e90.pth' ## whether to resume training, set value to 'None' or the path to the previous model.
    if Train_RESUME:
        logger.info("Resume from {}".format(Train_RESUME))
        policy_net = restore_from(pnet = policy_net, ckpt_path = Train_RESUME, mode = 'test')
     # 将模型放入GPU
    policy_net.to(device)
    '''---5 Start Test'''
    state, state_3D = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    state_3D = torch.tensor(state_3D, dtype=torch.float32)
    reward = 0
    frame = 0
    fig = plt.figure()
    while frame < cfg.Evaluate.steps_threshold:
        if len(state_3D.shape)==3:
            state_3D.unsqueeze_(0)
            state_3D.unsqueeze_(0)
        state_3D = state_3D.to(device=device)
        frame = frame + 1
        action = policy_action(state_3D, env, policy_net).numpy()
        next_state, r, done, others, next_state_3D = env.step(action)
        reward = reward + r
        state_3D = torch.tensor(next_state_3D, dtype=torch.float)
        if len(state_3D.shape)==3:
            state_3D.unsqueeze_(0)
            state_3D.unsqueeze_(0)
        state_3D = state_3D.to(device=device)
        action = policy_action(state_3D, env, policy_net).numpy()
        info = {'reward': reward, 'r': r, 'len_delta': others['len_delta'], 'radiu_delta': others['radius_delta'],
                'epoch': 0, 'frame': frame, 'action': action}

        fig = env.render_(fig, info, **cfg.Evaluate)

        if done or frame == cfg.Evaluate.steps_threshold:
            if cfg.Evaluate.is_save_gif:
                images_to_video(cfg.Evaluate.img_save_path, '*.jpg', isDelete=True, savename = 'TestResult')
            break
def main():
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
        cfg.Evaluate.img_save_path = cfg.Evaluate.img_save_path + cfg.Version
        cfg.Train.LOG_DIR = cfg.Train.LOG_DIR + cfg.Version
        cfg.Train.SNAPSHOT_DIR = cfg.Train.SNAPSHOT_DIR + cfg.Version
        cfg.qnet.in_channels = cfg.Env.step.action_num + cfg.Env.step.state_num
    if not os.path.exists(cfg.Train.LOG_DIR):
        os.makedirs(cfg.Train.LOG_DIR)
    if not os.path.exists(cfg.Evaluate.img_save_path) and cfg.Evaluate.is_save_gif:
        os.makedirs(cfg.Evaluate.img_save_path)
    init_log('global', logging.INFO)
    if cfg.Train.LOG_DIR:
        add_file_handler('global',
                         os.path.join(cfg.Train.LOG_DIR, 'logs.txt'),
                         logging.INFO)
    # logger.info("config \n {}".format(json.dumps(cfg, indent=4)))
    tb_writer = SummaryWriter(cfg.Train.LOG_DIR)
    
    #####-------注意，本例中，mask_array与坐标的排列方式均采用x,y,z形式来计算，zyx形式的要转换为xyz形式-----------------##### 
    
    '''---1 Data Preprocess    ---'''
    """
    ***'./spineData/sub-verse835_dir-iso_L1_seg-vert_msk.nii.gz':[[25,56,69],[25,57,111]]
    './spineData/sub-verse835_dir-iso_L2_seg-vert_msk.nii.gz':[[25,56,69],[25,57,111]]
    './spineData/sub-verse835_dir-iso_L3_seg-vert_msk.nii.gz':[[25,56,69],[25,57,111]]
    ***'./spineData/sub-verse835_dir-iso_L4_seg-vert_msk.nii.gz':[[27,55,60],[27,52,111]]
    './spineData/sub-verse821_L1_seg-vert_msk.nii.gz':[[25,56,69],[27,52,111]]
    ***'./spineData/sub-verse821_L2_seg-vert_msk.nii.gz':[[27,66,63],[27,67,111]]
    './spineData/sub-verse821_L3_seg-vert_msk.nii.gz':[[25,56,69],[25,57,111]]
    './spineData/sub-verse821_L5_seg-vert_msk.nii.gz':[[25,56,69],[25,57,111]]
    """
    dataDir = r'./spineData/sub-verse821_L2_seg-vert_msk.nii.gz'
    pedicle_points = np.asarray([[27,66,63],[27,67,111]])
    pedicle_points_in_zyx = True #坐标是zyx形式吗？
    spine_data = build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
    '''---2 Build Environment  ---'''
    env = build_Env(spine_data)  # 只修改了初始化函数，其他函数待修改
    '''---3 Build Networks     ---'''
    pnet_pretrained = None
    policy_net, q_net, target_p_net, target_q_net = build_nets(pnet_pretrained)
    # 将模型放入GPU
    policy_net.to(device)
    q_net.to(device)
    target_p_net.to(device)
    target_q_net.to(device)
    '''---4 Build Exploration pool ---'''
    experience_pool = build_exper_pool()
    '''---5 Build Optimizer    ---'''
    optimizer_p, optimizer_q = build_opt_lr(policy_net, q_net)
    '''---6 Resume networks and optimizers ---'''
    Train_RESUME = None ## whether to resume training, set value to 'None' or the path to the previous model.
    if Train_RESUME:
        logger.info("Resume from {}".format(Train_RESUME))
        policy_net,q_net,optimizer_p,optimizer_q, Train_START_EPOCH =\
            restore_from(policy_net, q_net, optimizer_p, optimizer_q, Train_RESUME)
    '''---7 Start Training'''
    train(env, policy_net, q_net, target_p_net, target_q_net, experience_pool, optimizer_q, optimizer_p, tb_writer)

if __name__ == "__main__":
    # main()
    evaluateOthers()



