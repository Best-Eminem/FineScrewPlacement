from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()
cfg = __C
__C.META_ARC = "Vertebra Nailing"
__C.Version = '01' # 用于标识当前训练版本，改变日志等保存路径 used to identify the different Training config, Change it when you start a new train, otherwise logs etc will be overwriten
# ------------------------------------------
# options for Data Preprocessing and Environment setting
# ------------------------------------------
__C.Data = CN()
__C.Data.typename = "nii" # 原始脊柱数据格式，对应不同文件读取函数 format of raw data : [nifti(nii) or dicom(dcm)]
__C.Data.name = "./dataset/verse004_seg.nii.gz" # path to the raw data
__C.Data.cropbbox = [[16,106],[182,232],[2,52]] # 用于裁剪脊柱的边界框 the bbox index to crop the spine,start from 1, if None, do not crop
__C.Data.fixpoint = [66,198,22] # 椎弓根中点坐标 center of the pedicle, start from 1
__C.Data.spinelabel = 22 # 脊柱的分割标签 segmentation label of spine

__C.Env = CN(new_allowed=True)
__C.Env.name = "envLeft" # 设置椎弓根环境，对应不同环境设置函数 ["envLeft", "envRight", "envBoth"] represent environment for left pedicle or right prdicle or both, respectively. TODO envRight and envBoth is unavailable

__C.Env.reset = CN()
__C.Env.reset.initdegree = None # 初始经纬度 None means default value, i.e., [0,0]
__C.Env.reset.initpoint = [0, -2.4, -2.4] # 初始定点坐标 None means default value, i.e., [0,0,0]
__C.Env.reset.is_rand_d = True # 是否为初始定经纬度添加随机噪声
__C.Env.reset.rdrange = [-90, 90] # range of random degree
__C.Env.reset.is_rand_p = False # 是否为初始定点坐标添加随机噪声
__C.Env.reset.rprange = [-0.1, 0.1] # range of random cross point

__C.Env.step = CN()
__C.Env.step.radiu_thres = [-3, 3] # 用于计算半径时，裁剪z轴 crop z axis (A<->P) to get correct radius todo, maybe next time
__C.Env.step.line_thres = None # 用于计算直线长度时，裁剪z轴 crop z axis to get correct length, 'None' means do not crop todo
__C.Env.step.min_action = -1.0 # threshold for sum of policy_net output and random exploration
__C.Env.step.max_action = 1.0
__C.Env.step.action_num = 5 # the number of action
__C.Env.step.state_num = 6 # the number of states
__C.Env.step.trans_mag = [0.1, 0.1, 0.1] # 直线上一点的移动的量级 magtitude of movement of the point (x,y,z) on the line
__C.Env.step.rotate_mag = [0.5, 0.5] # 直线经纬度旋转的量级 magtitude of rotation (δlatitude,δlongitude) of line
__C.Env.step.reward_weight = [0., 1.] # 计算每步reward的权重 weights for every kind of reward (), [line_delta, radius_delta] respectively
__C.Env.step.deg_threshold = [-180., 180., -180., 180.] # 用于衡量终止情况的直线经纬度阈值 [minimum latitude, maximum latitude, minimum longitude, maximum longitude]

__C.Env.step.line_rd = 1. # 用于计算直线长度时，设置直线半径。define the radius of line to compute the line length
__C.Env.step.update_para = 'p' # 用于环境的step操作中，设置更新直线角度还是定点 'd': update degree, 'p': update cpoint, 'dp' update both of them
__C.Env.step.done_r = 0.5 # 医学中允许的置钉最小半径（用于衡量是否为终止状态） allows minimum radius

# -----------------------------------
# options for policy net and Qnet
# -----------------------------------
__C.pnet = CN() # 之后需要修改网络，只用修改参数和网络即可
__C.pnet.name = 'linear0' # linear0 means there is 1 hidden layer
__C.pnet.in_channels = __C.Env.step.state_num
__C.pnet.out_channels = __C.Env.step.action_num
__C.pnet.hide_channels = 100
__C.pnet.pretrained = None # whether to load a pretrained model. set value to 'None' or path to pretrained model.


__C.qnet = CN()
__C.qnet.name = 'linear0' # linear0 means there is 1 hidden layer
__C.qnet.in_channels = __C.Env.step.action_num + __C.Env.step.state_num
__C.qnet.out_channels = 1
__C.qnet.hide_channels = 100
__C.qnet.pretrained = None # whether to load a pretrained model. set value to 'None' or path to pretrained model.
# -----------------------------------
# options for Training both of nets
# -----------------------------------
__C.Train = CN()
__C.Train.EPOCHS = 200
__C.Train.EPOCH_STEPS = 50
__C.Train.BATCH_SIZE = 50 # batch-train
__C.Train.WARM_UP_SIZE = __C.Train.BATCH_SIZE
__C.Train.UPDATE_INTERVAL = 10 # target_p_net and target_q_net are updated every #UPDATE_INTERVAL steps
__C.Train.GAMMA = 0.99 # used in target_q_value = r + cfg.Train.GAMMA * target_next_q_value * (1 - d)
__C.Train.EXPLORE_NOISE = 0.05 # noise of exploring action
__C.Train.UPDATE_WEIGHT = 0.9 # used in p_targ.data.mul_(UPDATE_WEIGHT); p_targ.data.add_((1 - UPDATE_WEIGHT) * p.data)
__C.Train.LEARN_RATE = 1e-3
__C.Train.START_EPOCH = 0
__C.Train.RESUME = None # whether to resume training, set value to 'None' or the path to the previous model.
__C.Train.SAVE_INTERVAL = 10 # intervals to save and evaluate model
__C.Train.SNAPSHOT_DIR = './snapshot/' # path to save snapshot
__C.Train.LOG_DIR = './logs/' #

__C.Evaluate = CN()
__C.Evaluate.steps_threshold = 300 # used to limit the forward steps when evaluation
__C.Evaluate.is_vis = False
__C.Evaluate.is_save_gif = True
__C.Evaluate.img_save_path = './imgs/'





