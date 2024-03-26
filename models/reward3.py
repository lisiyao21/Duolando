import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

def slipping_penalty(pose_seqf):
    global_vel = pose_seqf[:, :, :3]
    global_vel[:, :-1] = global_vel[:, 1:] - global_vel[:, :-1]
    global_vel[:, -1] = global_vel[:, -2]

    slip_reward = -torch.sqrt((global_vel ** 2).sum(-1)) * 100
    slip_reward[slip_reward > -2] = 1

    return slip_reward

def transl_lower_half_consistency(predicted_vel, pose_seqf):
    global_vel = pose_seqf[:, :, :3]
    global_vel[:, :-1] = global_vel[:, 1:] - global_vel[:, :-1]
    global_vel[:, -1] = global_vel[:, -2]

    diff = -torch.sqrt(torch.sum((predicted_vel - global_vel)**2, dim=-1)) * 100
    diff[diff > -3] = 1

    # diff += 1
    return diff

def up_down_half_consistency(pose_seqf):
    n, t, c = pose_seqf.size()


    pose = pose_seqf.reshape([n, t, c//3, 3])

    up_norm = torch.cross(pose[:, :, 14, :], pose[:, :, 13, :])
    up_norm /= up_norm.norm(dim=-1)[:, :, None]
    
    up_direct = torch.sum(up_norm * (pose[:, :, 15, :] - pose[:, :, 12, :]), dim=-1)
    up_direct /= up_direct.abs() + 1e-5
    up_norm *= up_direct[:, :,  None]
    up_norm[:, :, 1] = 0

    down_norm = torch.cross(pose[:, :, 4, :], pose[:, :, 5, :])
    down_norm /= down_norm.norm(dim=-1)[:, :, None]
    
    down_direct = torch.sum(down_norm * (pose[:, :, 4, :] - pose[:, :, 1, :] + pose[:, :, 5, :] - pose[:, :, 2, :] + pose[:, :, 4, :] - pose[:, :, 7, :] + pose[:, :, 5, :] - pose[:, :, 8, :]), dim=-1)
    down_direct /= down_direct.abs() + 1e-5
    down_norm *= up_direct[:, :, None]
    down_norm[:, :, 1] = 0

    reward = (up_norm * down_norm).sum(dim=-1).min(dim=-1)[0] * 10
    reward[reward >= 0 ] = 1.0

    return reward


def reward3(music_seq, pose_seql, pose_seqf, predicted_vel):
    # print(len(pose_seqf), flush=True)
    N, T, C = pose_seqf.size()
    reward_up = torch.zeros(N, T).cuda() + 1
    
    reward_down = transl_lower_half_consistency(predicted_vel, pose_seqf)
    # up_dwon_consistency_reward = up_down_half_consistency(pose_seqf)


    reward_lhand = torch.zeros(N, T).cuda() + 1
    reward_rhand = torch.zeros(N, T).cuda() + 1
    reward_transl = torch.zeros(N, T).cuda() + 1
    # reward_transl = slipping_penalty(pose_seqf)
    return (reward_up, reward_down, reward_lhand, reward_rhand, reward_transl)

