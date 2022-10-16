from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def load_pretrain(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model.load_state_dict(pretrained_dict)
    return model

def load_pretrains_test(pnet, qnet, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    pnet.load_state_dict(pretrained_dict['pnet_dict'])
    qnet.load_state_dict(pretrained_dict['qnet_dict'])
    return pnet, qnet

def restore_from(pnet, qnet=None, optimizer_p=None, optimizer_q=None, ckpt_path=None, mode='train'):
    if mode == 'train':
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        pnet.load_state_dict(ckpt['pnet_dict'])
        qnet.load_state_dict(ckpt['qnet_dict'])
        optimizer_p.load_state_dict(ckpt['optimizer_p'])
        optimizer_q.load_state_dict(ckpt['optimizer_q'])
        return pnet, qnet, optimizer_p,optimizer_q, epoch
    else:
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        pnet.load_state_dict(ckpt['pnet_dict'])
        return pnet

def enable_gradient(network):
    for p in network.parameters():
        p.requires_grad = True

def disable_gradient(network):
    for p in network.parameters():
        p.requires_grad = False

def copy_net(source_net, target_net, UPDATE_WEIGHT = 0.9):
    with torch.no_grad():
        for p, p_targ in zip(source_net.parameters(), target_net.parameters()):
            p_targ.data.mul_(UPDATE_WEIGHT)
            p_targ.data.add_((1 - UPDATE_WEIGHT) * p.data)
