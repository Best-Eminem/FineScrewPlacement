# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataLoad.loadNii import get_spinedata
from env import SingleSpineEnv
from utils.img_utils import images_to_video

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total_timesteps", type=int, default=20000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=5,
        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    # cfg.Env.step.line_rd = float(max(cfg.Env.step.line_rd, spacing)) # 根据spacing修改计算直线长度时的直线半径。
    return spine_data # namedtuple:

def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnv.SpineEnv(spine_data, degree_threshold, **cfg)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def evaluate(envs, agent, epoch):
    next_obs_2d, next_obs = envs.reset(random_reset = False)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    step = 0
    fig = plt.figure()
    rewards = 0
    while step<100:
        step +=1
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze_(0).unsqueeze_(0) if len(next_obs.shape) == 3 else next_obs)
        # TRY NOT TO MODIFY: execute the game and log data.
        _, reward, done, others, next_obs = envs.step(action.squeeze_(0).cpu().numpy() if len(action.shape) == 2 else action)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([1.] if done else [0.]).to(device)
        rewards = rewards + reward

        info = {'reward': rewards, 'r': reward, 'len_delta': others['len_delta'], 'radiu_delta': others['radius_delta'],
                'epoch': 0, 'frame': step, 'action':'{:.3f}, {:.3f}'.format(action[0], action[1])}

        fig = envs.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path='./ppo_imgs')

    images_to_video('./ppo_imgs', '*.jpg', isDelete=True, savename = 'Update%d'%(epoch))

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv3d(1, 32, 8, stride=4)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(32, 64, 4, stride=2)),
            nn.ReLU(True),
            layer_init(nn.Conv3d(64, 64, 3, stride=1)),
            nn.ReLU(True),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 16 * 6 * 4, 512)),
            nn.ReLU(True),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512, 2), std=0.01),
        )
        self.actor_logstd = nn.Parameter(-0.5 * torch.ones(1, 2))

    def get_value(self, x):
        return self.critic(self.network(x / 2.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 2.0)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # log_prob(value)计算value在定义的正态分布（mean,1）中对应的概率的对数
        action = torch.clamp(action, min=-1.0, max=1.0)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

def evluateothers():
    cfg = {'deg_threshold':[-360., 360., -360., 360.],
           'reset':{'rdrange':[-90, 90],
                    'state_shape':(160, 80, 64)},
           'step':{'rotate_mag':[5, 5]},}
    
    args = parse_args()
    if not os.path.exists('./ppo_imgs'):
        os.makedirs('./ppo_imgs')
    if not os.path.exists('./ppo_snapshot'):
        os.makedirs('./ppo_snapshot')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataDir = r'./spineData/sub-verse835_dir-iso_L2_seg-vert_msk.nii.gz'
    pedicle_points = np.asarray([[31,57,63],[30,58,106]])
    pedicle_points_in_zyx = True #坐标是zyx形式吗？
    spine_data = build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
    '''---2 Build Environment  ---'''
    envs = build_Env(spine_data, cfg['deg_threshold'], cfg)  # 只修改了初始化函数，其他函数待修改
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent()
    Train_RESUME = 'ppo_snapshot_not_reset/ppo_160.pth' ## whether to resume training, set value to 'None' or the path to the previous model.
    if Train_RESUME:
        ckpt = torch.load(Train_RESUME)
        # epoch = ckpt['epoch']
        agent.load_state_dict(ckpt)
    agent.to(device)
    
    next_obs_2d, next_obs = envs.reset(random_reset = False)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    step = 0
    fig = plt.figure()
    rewards = 0
    while step<200:
        step +=1
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze_(0).unsqueeze_(0) if len(next_obs.shape) == 3 else next_obs)
        # TRY NOT TO MODIFY: execute the game and log data.
        _, reward, done, others, next_obs = envs.step(action.squeeze_(0).cpu().numpy() if len(action.shape) == 2 else action)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([1.] if done else [0.]).to(device)
        rewards = rewards + reward

        info = {'reward': rewards, 'r': reward, 'len_delta': others['len_delta'], 'radiu_delta': others['radius_delta'],
                'epoch': 0, 'frame': step, 'action':'{:.3f}, {:.3f}'.format(action[0], action[1])}

        fig = envs.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path='./ppo_imgs')

    images_to_video('./ppo_imgs', '*.jpg', isDelete=True, savename = 'Update%d'%(0))

def train():
    # env parameters
    cfg = {'deg_threshold':[-360., 360., -360., 360.],
           'reset':{'rdrange':[-90, 90],
                    'state_shape':(160, 80, 64)},
           'step':{'rotate_mag':[5, 5]},}
    
    args = parse_args()
    if not os.path.exists('./ppo_imgs'):
        os.makedirs('./ppo_imgs')
    if not os.path.exists('./ppo_snapshot'):
        os.makedirs('./ppo_snapshot')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    # )
    dataDir = r'./spineData/sub-verse835_dir-iso_L1_seg-vert_msk.nii.gz'
    pedicle_points = np.asarray([[25,56,69],[25,57,111]])
    pedicle_points_in_zyx = True #坐标是zyx形式吗？
    spine_data = build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160) #spine_data 是一个包含了mask以及mask坐标矩阵以及椎弓根特征点的字典
    '''---2 Build Environment  ---'''
    envs = build_Env(spine_data, cfg['deg_threshold'], cfg)  # 只修改了初始化函数，其他函数待修改
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.state_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (envs.action_num,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_2d, next_obs = envs.reset(random_reset = False)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in tqdm(range(1, num_updates + 1)):
        # next_obs_2d, next_obs = envs.reset(random_reset = False)
        # next_obs = torch.Tensor(next_obs).to(device)
        # next_done = torch.zeros(args.num_envs).to(device)
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 探寻一条包含若干steps的轨迹
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze_(0).unsqueeze_(0) if len(next_obs.shape) == 3 else next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            _, reward, done, info, next_obs = envs.step(action.squeeze_(0).cpu().numpy() if len(action.shape) == 2 else action)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([1.] if done else [0.]).to(device)

            print(f"global_step={global_step}, episodic_return={reward}")
            writer.add_scalar("charts/episodic_return", reward, global_step)
            writer.add_scalar("charts/episodic_length", step+1, global_step)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze_(0).unsqueeze_(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.state_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (envs.action_num,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds].unsqueeze_(1), b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if update%10 == 0:
            torch.save(agent.state_dict(), './ppo_snapshot/'+'/ppo_%d.pth' % (update))
            agent = agent.eval()
            evaluate(envs, agent, update)
            agent = agent.train()
    envs.close()
    writer.close()
if __name__ == "__main__":
    train()
    # evluateothers()