import numpy as np
import multiprocessing
import sys
import os
import SimpleITK as sitk
sys.path.append(os.getcwd())
import torch
from torch.optim import Adam
import gym
import matplotlib.pyplot as plt
import time
import core as core
from tqdm import tqdm
from multiprocessing import Manager,Process
from logx import EpochLogger
from env import SingleSpineEnvSingle
from dataLoad.loadNii import get_spinedata
from utils.img_utils import images_to_video
from torch.utils.tensorboard import SummaryWriter
# from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_L_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_a_Left_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act_L, rew, val, logp_a_Left):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_L_buf[self.ptr] = act_L
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_a_Left_buf[self.ptr] = logp_a_Left
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act_L=self.act_L_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp_L=self.logp_a_Left_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    # cfg.Env.step.line_rd = float(max(cfg.Env.step.line_rd, spacing)) # 根据spacing修改计算直线长度时的直线半径。
    return spine_data # namedtuple:
    # spine_datas = []
    # for dir, points in zip(dataDir, pedicle_points):
    #     spine_data = get_spinedata(dir, points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    #     spine_datas.append(spine_data)
    # # cfg.Env.step.line_rd = float(max(cfg.Env.step.line_rd, spacing)) # 根据spacing修改计算直线长度时的直线半径。
    # return spine_datas # namedtuple:

def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnvSingle.SpineEnv(spine_data, degree_threshold, **cfg)
    return env

def evluateothers(args, env_fn, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict()):
    cfg = {'deg_threshold':[-30., 30., -50., 0],#[-50., 50., -90., 90.],#[-65., 65., -45., 25.],
           'screw_index':args.screw_index,
           'reset':{'rdrange':[-45, 45],
                    'state_shape':(160, 80, 64) if not args.Leaning_to_Optimize else args.LTO_length*5+1},
           'step':{'rotate_mag':[10, 10], 'discrete_action':args.discrete_action}
           }
    
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataDirs = [
                r'spineData/sub-verse500_dir-ax_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse506_dir-iso_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse521_dir-ax_L1_ALL_msk.nii.gz',
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
                                [[39,49,58],[39,48,105]],
                                [[38,43,67],[38,43,108]],
                                [[30,46,65],[30,46,108]],
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
    index = 0
    max_rewards=[]
    for dataDir, pedicle_point in zip(dataDirs, pedicle_points):
        index += 1
        dataDir = os.path.join(os.getcwd(), dataDir)
        pedicle_point_in_zyx = True #坐标是zyx形式吗？
        spine_data = build_data_load(dataDir, pedicle_point, pedicle_point_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
        '''---2 Build Environment  ---'''
        env = build_Env(spine_data, cfg['deg_threshold'], cfg)  # 只修改了初始化函数，其他函数待修改
        # assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        # Create actor-critic module
        obs_dim = env.state_shape
        act_dim = env.action_num
        ac = actor_critic(obs_dim, act_dim, discrete=args.discrete_action, LTO = args.Leaning_to_Optimize, **ac_kwargs).to(device)
        Train_RESUME = os.path.join(args.snapshot_dir, 'ppo_400.pth') ## whether to resume training, set value to 'None' or the path to the previous model.
        if Train_RESUME:
            ckpt = torch.load(Train_RESUME)
            # epoch = ckpt['epoch']
            ac.load_state_dict(ckpt)
        
        _, o_3D = env.reset(random_reset = False)
        o_3D = torch.Tensor(o_3D).to(device)
        step = 0
        fig = plt.figure()
        rewards = 0
        max_reward = 0
        # fig = env.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path=args.imgs_dir)
        while step < args.steps:
            step += 1
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_left = ac.evaluate(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
            # TRY NOT TO MODIFY: execute the game and log data.
            state_, reward, done, others, o_3D = env.step(action_left.squeeze_(0).cpu().numpy() if len(action_left.shape) == 2 else action_left)
        
            action_left = others['action_left']
            o_3D, next_done = torch.Tensor(o_3D).to(device), torch.Tensor([1.] if done else [0.]).to(device)
            if reward > max_reward:
                max_reward = reward
                max_screw = env.state3D_array
            rewards = rewards + reward
            info = {'reward': rewards, 'r': reward, 'len_delta_L': others['len_delta_L'], 'radiu_delta_L': others['radius_delta_L'],
                    'epoch': 0, 'frame': step, 
                    'action_left':'{:.3f}, {:.3f}'.format(action_left[0], action_left[1])}

            fig = env.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path=args.imgs_dir)
            # if done:
            #     break
        max_rewards.append(max_reward)
        state3D_itk = sitk.GetImageFromArray(np.transpose(max_screw, (2, 1, 0)))
        sitk.WriteImage(state3D_itk, os.path.join(args.imgs_dir ,os.path.basename(dataDir)))
        images_to_video(args.imgs_dir, '*.jpg', isDelete=True, savename = os.path.basename(dataDir))

def evaluate(args, envs, agent, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index in range(len(envs)):
        env = envs[index]
        _, o_3D = env.reset(random_reset = False)
        o_3D = torch.Tensor(o_3D).to(device)
        step = 0
        fig = plt.figure()
        rewards = 0
        obs_dim = env.state_shape
        act_dim = env.action_num
        while step < args.steps:
            step += 1
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_left = agent.evaluate(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)

            # TRY NOT TO MODIFY: execute the game and log data.
            state_, reward, done, others, o_3D = env.step(action_left.squeeze_(0).cpu().numpy() if len(action_left.shape) == 2 else action_left)
            
            action_left = others['action_left']
            o_3D, next_done = torch.Tensor(o_3D).to(device), torch.Tensor([1.] if done else [0.]).to(device)
            rewards = rewards + reward
            info = {'reward': rewards, 'r': reward, 'len_delta_L': others['len_delta_L'], 'radiu_delta_L': others['radius_delta_L'],
                    'epoch': 0, 'frame': step, 
                    'action_left':'{:.3f}, {:.3f}'.format(action_left[0], action_left[1])}

            fig = env.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path=args.imgs_dir)
            # if done:
            #     break
        # state3D_itk = sitk.GetImageFromArray(env.state3D_array)
        # sitk.WriteImage(state3D_itk, os.path.join(args.imgs_dir ,os.path.basename(dataDir)))
        images_to_video(args.imgs_dir, '*.jpg', isDelete=True, savename = 'Env_%d_%d'%(index+1, epoch))

def ppo(args, env_fn, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=100, epochs=400, gamma=0.99, clip_ratio=0.2, pi_lr=4e-4,
        vf_lr=1e-3, train_pi_iters=10, train_v_iters=10, lam=0.97, max_ep_len=100,
        target_kl=0.05, logger_kwargs=dict(), save_freq=20):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    # run_name = f"{'FineScrewPlacement'}__{seed}__{time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))}"
    run_name = f"{os.path.basename(args.imgs_dir)}__{time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))}"
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set up logger and save configuration
    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())
    cfg = {'screw_index':args.screw_index,
           'deg_threshold':[0., 30., -50., 10.] if args.screw_index==0 else [-30., 0., -50., 10.],#[-50., 50., -90., 90.],#[-65., 65., -45., 25.],
           'reset':{'rdrange':[-45, 45],
                    'state_shape':(160, 80, 64) if not args.Leaning_to_Optimize else args.LTO_length*5+1},
           'step':{'rotate_mag':[10, 10], 'discrete_action':args.discrete_action}
           }

    print(cfg)
    print(args)
    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataDirs = [
                r'spineData/sub-verse621_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse500_dir-ax_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse506_dir-iso_L1_ALL_msk.nii.gz',
                r'spineData/sub-verse521_dir-ax_L1_ALL_msk.nii.gz',

                r'spineData/sub-verse518_dir-ax_L2_ALL_msk.nii.gz',
                r'spineData/sub-verse536_dir-ax_L2_ALL_msk.nii.gz',
                r'spineData/sub-verse586_dir-iso_L2_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L2_ALL_msk.nii.gz',

                r'spineData/sub-verse510_dir-ax_L3_ALL_msk.nii.gz',
                r'spineData/sub-verse518_dir-ax_L3_ALL_msk.nii.gz',
                r'spineData/sub-verse818_dir-ax_L3_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L3_ALL_msk.nii.gz',

                r'spineData/sub-verse514_dir-ax_L4_ALL_msk.nii.gz',
                r'spineData/sub-verse534_dir-iso_L4_ALL_msk.nii.gz',
                r'spineData/sub-verse537_dir-iso_L4_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L4_ALL_msk.nii.gz',
                
                r'spineData/sub-verse505_L5_ALL_msk.nii.gz',
                r'spineData/sub-verse510_dir-ax_L5_ALL_msk.nii.gz',
                r'spineData/sub-verse614_L5_ALL_msk.nii.gz',
                r'spineData/sub-verse621_L5_ALL_msk.nii.gz',
                ]
    pedicle_points = np.asarray([
                                [[35,47,65],[36,47,105]],
                                [[39,49,58],[39,48,105]],
                                [[38,43,67],[38,43,108]],
                                [[30,46,65],[30,46,108]],
                                
                                [[33,42,64],[37,44,103]],
                                [[33,40,57],[31,45,96]],
                                [[33,43,66],[36,43,101]],
                                [[36,48,62],[38,48,102]],
                                 
                                [[33,44,67],[33,42,101]],
                                [[33,43,59],[38,45,101]],
                                [[33,47,61],[36,46,108]],
                                [[38,47,62],[39,47,104]],
                                 
                                [[59,45,60],[51,44,109]],
                                [[35,43,63],[33,46,105]],
                                [[46,44,63],[46,44,101]],
                                [[43,48,60],[44,48,107]],
                                 
                                [[34,43,61],[34,41,102]],
                                [[45,52,68],[45,43,110]],
                                [[42,45,64],[40,44,113]],
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
    obs_dim = envs[0].state_shape
    act_dim = envs[0].action_num

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, discrete=args.discrete_action, LTO = args.Leaning_to_Optimize, **ac_kwargs).to(device)

    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch) #/ num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act_L, adv, logp_old_L = (data['obs'] if args.Leaning_to_Optimize else data['obs'].unsqueeze(1)).to(device), data['act_L'].to(device), data['adv'].to(device), data['logp_L'].to(device)

        # Policy loss
        pi_L, logp_L = ac.pi(obs, act_L)

        ratio_L = torch.exp(logp_L - logp_old_L)
        clip_adv_L = torch.clamp(ratio_L, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi_L = -(torch.min(ratio_L * adv, clip_adv_L)).mean()

        # Useful extra info
        approx_kl_L = (logp_old_L - logp_L).mean().item()
        ent_L = pi_L.entropy().mean().item()
        clipped_L = ratio_L.gt(1+clip_ratio) | ratio_L.lt(1-clip_ratio)
        clipfrac_L = torch.as_tensor(clipped_L, dtype=torch.float32).mean().item()

        pi_info = dict(kl=approx_kl_L, ent=ent_L, cf=clipfrac_L)
        return loss_pi_L, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'].unsqueeze(1).to(device), data['ret'].to(device)
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            # if kl > 1.5 * target_kl:
            #     print('Early stopping at step %d due to reaching max kl.'%i)
            #     break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))
        return pi_l_old, v_l_old, kl, ent, cf, 

    # Prepare for interaction with environment
    start_time = time.time()
    env_index = 0
    env = envs[env_index]
    _, o_3D = env.reset(random_reset = True)
    o_3D = torch.Tensor(o_3D).to(device)
    ep_ret, ep_len = 0, 0 
    # Main loop: collect experience in env and update/log each epoch
    global_step = 0
    for epoch in tqdm(range(1, epochs+1)):
        frac = 1.0 - (epoch - 1.0) / epochs
        lrnow = frac * pi_lr
        pi_optimizer.param_groups[0]["lr"] = lrnow
        vf_optimizer.param_groups[0]["lr"] = lrnow
        for t in range(local_steps_per_epoch):
            a_Left, v, logp_a_Left = ac.step(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
            state_, r, d, info, next_o_3D = env.step(a_Left)
            ep_ret += r
            ep_len += 1
            o_3D = o_3D.squeeze(0).squeeze(0).cpu().numpy() if len(o_3D.shape) == 5 else o_3D.cpu().numpy()
            buf.store(o_3D, a_Left, r, v, logp_a_Left)
            # logger.store(VVals=v)
            
            # # Update obs (critical!)
            o_3D = next_o_3D
            o_3D = torch.Tensor(o_3D).to(device)
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            global_step = (epoch-1)*local_steps_per_epoch+t
            writer.add_scalar("charts/step_return", r, global_step)

            print(f"global_step={global_step}, step_return={r}")
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended: #or timeout:
                    _, v, _ = ac.step(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    writer.add_scalar("charts/episodic_return", ep_ret, epoch)
                    writer.add_scalar("charts/episodic_length", ep_len, epoch)
                # o_3D, ep_ret, ep_len = env.reset(), 0, 0
                # for env in envs:
                #     _, _= env.reset(random_reset = False) # 每回合结束时把所所有环境reset
                # env_index = (epoch)%len(envs)
                # env = envs[env_index]
                if epoch_ended:
                    if epoch % 20 == 0:
                        env_index += 1
                        # pi_optimizer.param_groups[0]["lr"] = pi_lr
                        # vf_optimizer.param_groups[0]["lr"] = vf_lr
                    env = envs[env_index%len(envs)]
                _, o_3D = env.reset(random_reset = True)
                o_3D = torch.Tensor(o_3D).to(device)
                ep_ret, ep_len = 0, 0 

        # # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     logger.save_state({'env': env}, None)

        # Perform PPO update!
        LossPi, LossV, KL, Entropy, ClipFrac = update()
        if epoch % save_freq == 0:
            torch.save(ac.state_dict(), args.snapshot_dir+'/ppo_%d.pth' % (epoch))
            ac = ac.eval()
            evaluate(args, envs, ac, epoch)
            ac = ac.train()
        writer.add_scalar("charts/learning_rate", pi_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", LossV, global_step)
        writer.add_scalar("losses/policy_loss", LossPi, global_step)
        writer.add_scalar("losses/entropy", Entropy, global_step)
        # writer.add_scalar("losses/old_approx_kl", KL, global_step)
        writer.add_scalar("losses/approx_kl", KL, global_step)
        writer.add_scalar("losses/clipfrac", ClipFrac, global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FineScrewPlacement')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--screw_index', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--imgs_dir', type=str, default='./spinningup/imgs_3d_discrete_3_surface_all')
    parser.add_argument('--snapshot_dir', type=str, default='./spinningup/snapshot_3d_discrete_3_surface_all')
    parser.add_argument('--KL', type=str, default='No KL')
    parser.add_argument('--clip', type=str, default='clip in env')
    parser.add_argument('--Leaning_to_Optimize', type=bool, default=False)
    parser.add_argument('--discrete_action', type=bool, default=True)
    parser.add_argument('--LTO_length', type=int, default=10)
    
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(args, build_Env, actor_critic=core.MyMLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        save_freq=args.save_freq, logger_kwargs=logger_kwargs,max_ep_len=args.steps)
    
    # evluateothers(args, build_Env, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))