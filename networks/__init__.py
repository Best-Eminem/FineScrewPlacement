from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from networks.policyNet import linearPNet
from networks.QNet import linearQNet

POLICYS = {'linear0': linearPNet}
QS = {'linear0': linearQNet}

def get_p_net(name, **kwargs):
    return POLICYS[name](**kwargs)

def get_q_net(name, **kwargs):
    return QS[name](**kwargs)