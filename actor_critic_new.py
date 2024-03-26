import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from datasets.dd100lf_all2 import DD100lfAll as DD100lf
from datasets.dd100lf_demo import DD100lfDemo
from datasets.aclf import AClf
# from models.vqvae import VQVAE

from utils.log import Logger
# from utils.save import save_smplx
from utils.save import save_smplx, save_pos3d
from utils.visualize import visualize2 as visualize
from utils.visualize import visualize as visualize1
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
warnings.filterwarnings('ignore')

import torch.nn.functional as F
# a, b, c, d = check_data_distribution('/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_train')

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden

def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)
    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    r = R.from_matrix(rots)  # from_matrix
    aas = r.as_rotvec()
    return np.reshape(aas, orig_shape + (3,))
def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


class AC():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        vqvae = self.model.eval()
        gpt = self.model2.train()
        gpt.module.freeze_drop()
        transl_vqvae = self.model3.eval()
        # qnet = self.modelq.train()

        reward_fun = self.reward
        config = self.config
        ddm = []
        if hasattr(config, 'demo') and config.demo:
            ddm = True
        else:
            ddm = False
        ddm = True
        data = self.config.data
        # criterion = nn.MSELoss()
        labeled_data = self.train_loader
        test_loader = self.test_loader
        optimizer = self.optimizer
        # optimizer_q = self.optimizer_q
        log = Logger(self.config, self.expdir)
        updates = 0
        
        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(config.transl_vqvae_weight)
        transl_vqvae.load_state_dict(checkpoint['model'])

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'])
        # if hasattr(config, 'init_weight_q') and config.init_weight_q is not None and config.init_weight_q is not '':
        #     print('Use pretrained model!')
        #     print(config.init_weight_q)  
        #     checkpoint = torch.load(config.init_weight_q)
        #     gpt.module.load_state_dict(checkpoint['model'], strict=False)
            
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        
        # maintain alll sequences for replay buffer
        sample_sequences = []
        replay_mem = []
        first_time_mem = []
        

        # print('Here!!!!!!!!!!!!!!!!!!80', flush=True)
        
        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            sample_sequences = []
            log.set_progress(epoch_i, len(self.train_loader))
            # use current weight to generate samples 
            with torch.no_grad():
                gpt.eval()
                dance_names = []
                quants_out = []

                reward_up_stat = []
                reward_down_stat = []
                reward_lhand_stat = []
                reward_rhand_stat = []
                reward_transl_stat = []

                if ddm:
                    for i_eval, batch_eval in enumerate(tqdm(self.demo_loader, desc='Generating Dance Poses')):
                    # Prepare data
                    # if hasattr(config, 'demo') and config.demo:
                    #     music_seq = batch_eval['music'].to(self.device)
                    #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                    # else:
                        music_seq, pose_seql = batch_eval['music'], batch_eval['pos3dl']
                        music_seq = music_seq.to(self.device)
                        pose_seql = pose_seql.to(self.device)
                        
                        
                        fname = batch_eval['fname'][0]
                        dance_names.append(fname)

                        
                        
                        # lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                        transll = pose_seql[:, :, :3].clone()
                        transll = transll - transll[:, :1, :3].clone()

                        pose_seql[:, :, :3] = 0
                        # pose_seqf[:, :, :3] = 0

                        quants_predl = vqvae.module.encode(pose_seql)
                        # quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                        # quants_transl = transl_vqvae.module.encode(lftransl)


                        # if isinstance(quants_predf, tuple):
                        y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predl)))
                        x = (torch.ones(1,1).long().cuda()*19, 
                            torch.ones(1,1).long().cuda()*41, 
                            torch.ones(1,1).long().cuda()*268,  
                            torch.ones(1,1).long().cuda()*197)
                        transl = (torch.ones(1,1).long().cuda()*321, )
                        zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                        pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                        lf_transl = transl_vqvae.module.decode(transl_z)

                        # print(pose_sample.size(),flush=True)
                        pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                        pose_sample = pose_sample.cpu().data.numpy()

                            # the root of left hand
                        left_twist = pose_sample[:, :, 60:63]
                        # 25,40
                        pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_sample[:, :, 63:66]
                        # 40,55
                        pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_sample[:, :, :3]  # the root
                        pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_sample[:, :, :3] = root
                        
                        # if isinstance(zs, tuple):
                        #     quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                        # else:
                        #     quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                        pose_seql[:, :, :3] = transll
                        pose_seql = pose_seql.cpu().data.numpy()
                        left_twist = pose_seql[:, :, 60:63]
                        # 25,40
                        pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_seql[:, :, 63:66]
                        # 40,55
                        pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_seql[:, :, :3]  # the root
                        pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_seql[:, :, :3] = root
                        
                        rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
                        N, T = rewards[0].shape
                        rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).min(axis=-1) for ii in range(len(rewards)))

                        reward_up_stat.append(np.mean(rewards[0]))
                        reward_down_stat.append(np.mean(rewards[1]))
                        reward_lhand_stat.append(np.mean(rewards[2]))
                        reward_rhand_stat.append(np.mean(rewards[3]))
                        reward_transl_stat.append(np.mean(rewards[4]))

                        zs = tuple(zs[ii][0].cpu().long().data.numpy()[0] for ii in range(len(zs)))
                        y = tuple(y[ii][0].cpu().long().data.numpy() for ii in range(len(y)))
                        # print(len(music_seq), len(y), len(y[0][0]), len(zs), len(zs[0][0]), len(rewards[0][0]), flush=True)
                        transl_z = transl_z[0].cpu().data.numpy()[0]
                        # print(y, zs, transl_z, rewards, flush=True)
                        sample_sequences.append((music_seq.cpu().data.numpy()[0], y, zs+(transl_z,), rewards))
                            
                else:
                    for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                        
                        music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                        music_seq = music_seq.to(self.device)
                        pose_seql = pose_seql.to(self.device)
                        pose_seqf = pose_seqf.to(self.device)
                        
                        fname = batch_eval['fname'][0]
                        dance_names.append(fname)

                        lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                        transll = pose_seql[:, :, :3].clone()
                        transll = transll - transll[:, :1, :3].clone()

                        pose_seql[:, :, :3] = 0
                        pose_seqf[:, :, :3] = 0

                        quants_predl = vqvae.module.encode(pose_seql)
                        quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                        quants_transl = transl_vqvae.module.encode(lftransl)

                        y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
                        x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
                        # print(len(quants_transl[0]), flush=True)
                        transl = (quants_transl[0][:, :1], )

                        zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                        pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                        lf_transl = transl_vqvae.module.decode(transl_z)

                        # print(pose_sample.size(),flush=True)
                        pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                        pose_sample = pose_sample.cpu().data.numpy()

                            # the root of left hand
                        left_twist = pose_sample[:, :, 60:63]
                        # 25,40
                        pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_sample[:, :, 63:66]
                        # 40,55
                        pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_sample[:, :, :3]  # the root
                        pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_sample[:, :, :3] = root
                        
                        # if isinstance(zs, tuple):
                        #     quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                        # else:
                        #     quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                        pose_seql[:, :, :3] = transll
                        pose_seql = pose_seql.cpu().data.numpy()
                        left_twist = pose_seql[:, :, 60:63]
                        # 25,40
                        pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_seql[:, :, 63:66]
                        # 40,55
                        pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_seql[:, :, :3]  # the root
                        pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_seql[:, :, :3] = root
                        
                        rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
                        N, T = rewards[0].shape
                        rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).min(axis=-1) for ii in range(len(rewards)))

                        reward_up_stat.append(np.mean(rewards[0]))
                        reward_down_stat.append(np.mean(rewards[1]))
                        reward_lhand_stat.append(np.mean(rewards[2]))
                        reward_rhand_stat.append(np.mean(rewards[3]))
                        reward_transl_stat.append(np.mean(rewards[4]))

                        zs = tuple(zs[ii][0].cpu().long().data.numpy()[0] for ii in range(len(zs)))
                        y = tuple(y[ii][0].cpu().long().data.numpy() for ii in range(len(y)))
                        # print(len(music_seq), len(y), len(y[0][0]), len(zs), len(zs[0][0]), len(rewards[0][0]), flush=True)
                        transl_z = transl_z[0].cpu().data.numpy()[0]
                        # print(y, zs, transl_z, rewards, flush=True)
                        sample_sequences.append((music_seq.cpu().data.numpy()[0], y, zs+(transl_z,), rewards))
                    
            if epoch_i == 1:
                first_time_mem.extend(sample_sequences)
            else:
                print('add past memory!', flush=True)
                sample_seq_copy = sample_sequences[:]
                sample_sequences.extend(first_time_mem)
                random_mem = random.choices(replay_mem, k=5*len(sample_sequences) if len(replay_mem) > 5*len(sample_sequences) else len(replay_mem))
                sample_sequences.extend(random_mem)
                replay_mem.extend(sample_seq_copy)
            
            # buff = 0
            replay_buffer = self._build_rl_loader(sample_sequences)

            r_up_mean, r_down_mean, r_lhand_mean, r_rhand_mean, r_transl_mean = np.mean(reward_up_stat), np.mean(reward_down_stat), np.mean(reward_lhand_stat), np.mean(reward_rhand_stat), np.mean(reward_transl_stat)
            # print(len(labeled_data), flush=True)
            # print('Here!!!!!!!!!!!!!!!!!!166', flush=True)
            for iter_ii, batch_rl in enumerate(replay_buffer):

                ## check whether the reward during training is correct
                music_feat, y, quants_input, quants_target, rewards = batch_rl['music_feat'], batch_rl['quants_l'], batch_rl['quants_input'], batch_rl['quants_target'], batch_rl['rewards']
               
                # print('cond size', cond[0].size(), flush=True)
                # print('quants input szie', quants_input[0])
                # print(quants_target)
                # print(quants_target[0]) 
                # print(rewards)
                # print(rewards[0], flush=True)

                music_feat = music_feat.to(self.device)
                y = tuple(y[ii].to(self.device) for ii in range(len(y)))
                quants_input = tuple(quants_input[ii].to(self.device) for ii in range(len(quants_input)))
                quants_target = tuple(quants_target[ii].to(self.device) for ii in range(len(quants_target)))
                rewards = tuple(rewards[ii].to(self.device) for ii in range(len(rewards)))
                cond = (music_feat, ) + y 

                # print(quants_input[-1], quants_target[-1], rewards[-1], flush=True)

                # print(music_feat.size(), flush=True)
                # print(y[0].size(), flush=True)
                # print(quants_input[0].size(), flush=True)
                # print(quants_target[0].size(), flush=True)
                # print(rewards[0].size(), flush=True)
                # with torch.no_grad():
                #     # state = gpt.module.state(quants_input)
                #     gpt.eval()
                #     qnet.eval()
                #     state = gpt.state(quants_input, cond)
                #     probs, _ = qnet(state)
                    # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs
                    # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                    # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                    # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                    # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                    # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                    # print('transl_before-train:', ix_transl, flush=True)

                # with torch.no_grad():
                #     gpt.eval()
                #     # logits, _ = gpt(quants_input, cond=cond)

                #     # print(logits[-1])
                #     # print(torch.mean(logits[-1], dim=-1))
                #     # print(logits[-1].min(dim=-1))
                #     # print(logits[-1].max(dim=-1),)
                #     # print(logits[-1].)
                #     state = gpt.module.state(quants_input, cond)
                #     probs_pi, _ = gpt.module.actor(state)
                #     # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs_pi
                #     # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                #     # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                #     # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                #     # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                #     # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                #     # print('transl_pi:', ix_transl, flush=True)
                #     # print()
                # qnet.train()
                gpt.train()

                optimizer.zero_grad()
                # state = state.clone().detach()
                # state.requires_grad = True
                # _, q_loss, _ = qnet(state.clone().detach(), quants_target, rewards, probs_pi)
        
                _, loss = gpt(quants_input, cond, rewards, quants_target)

                loss.mean().backward()
                optimizer.step()
                
                # check_loss2 = q_loss.clone().item()
                # print(check_loss1, check_loss2, flush=True)

                # 2. q --> pi
                # with torch.no_grad():
                #     # state = gpt.module.state(quants_input)
                #     qnet.eval()
                #     probs, _, _ = qnet(state)
                #     # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs
                #     # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                #     # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                #     # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                #     # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                #     # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                #     # print('transl:', ix_transl, flush=True)
                # gpt.train()
                # optimizer.zero_grad()
                # gpt.module.freeze_drop()
                # _, pi_loss = gpt.module.actor(state, probs)
    
                # 3. imitation learning loss
                # gpt.train()
                
                # batch_labeled = next(iter(labeled_data))
                # music_seq, pose_seql, pose_seqf  = batch_labeled['music'], batch_labeled['pos3dl'], batch_labeled['pos3df'] 
                # music_seq = music_seq.to(self.device)
                # pose_seql, pose_seqf = pose_seql.to(self.device), pose_seqf.to(self.device)

                # transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0

                # # music
                # pose_seql[:, :, :3] = 0
                # pose_seqf[:, :, :3] = 0
                
                # print(pose_seq.size())
                

                # with torch.no_grad():
                #     quants_predl = vqvae.module.encode(pose_seql)
                #     quants_predf = vqvae.module.encode(pose_seqf)
                #     quants_transl = transl_vqvae.module.encode(transl)

                #     quants_cond = tuple(quants_predl[ii][0][:, :config.motion_len+config.look_forward_size].clone().detach() for ii in range(len(quants_predl)))
                #     quants_input = tuple(quants_predf[ii][0][:, :config.motion_len].clone().detach() for ii in range(len(quants_predf)))
                #     quants_target = tuple(quants_predf[ii][0][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_predf)))
                #     quants_transl_input = tuple(quants_transl[ii][:, :config.motion_len].clone().detach() for ii in range(len(quants_transl)))
                #     quants_transl_target = tuple(quants_transl[ii][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_transl)))
                                    
                # output, bc_loss = gpt(quants_input+quants_transl_input, (music_seq[:, config.music_motion_rate:config.music_motion_rate+config.music_len, ], ) + quants_cond, quants_target+quants_transl_target)
                
                # policy_loss = pi_loss
                # #  + config.lambda_bc * bc_loss

                # policy_loss.mean().backward()

                # # update parameters
                # optimizer.step()

                stats = {
                    'updates': updates,
                    'reward_down': r_down_mean,
                    'reward_tranl': r_transl_mean,
                    'loss': loss.mean().item(),
                    # 'bc_loss': bc_loss.mean().item(),
                    # 'pi_loss': pi_loss.mean().item()
                    
                    # 'entropy': entropy.clone().detach().mean()
                }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                # 'modelq': qnet.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0:
                print('validation...')

                with torch.no_grad():
                    gpt.eval()

                    self.device = torch.device('cuda' if config.cuda else 'cpu')
                    
                    results = []
                    leaders = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants_out = {}
                    dance_names = []
                    
                    if ddm:
                        for i_eval, batch_eval in enumerate(tqdm(self.demo_loader, desc='Generating Demo Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
                            music_seq, pose_seql = batch_eval['music'], batch_eval['pos3dl']
                            music_seq = music_seq.to(self.device)
                            pose_seql = pose_seql.to(self.device)
                            
                            
                            fname = batch_eval['fname'][0]
                            dance_names.append(fname)

                            
                            
                            # lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                            transll = pose_seql[:, :, :3].clone()
                            transll = transll - transll[:, :1, :3].clone()

                            pose_seql[:, :, :3] = 0
                            # pose_seqf[:, :, :3] = 0

                            quants_predl = vqvae.module.encode(pose_seql)
                            # quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                            # quants_transl = transl_vqvae.module.encode(lftransl)


                            # if isinstance(quants_predf, tuple):
                            y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predl)))
                            # x = tuple(torch.randint(0, 512, [1, 1]).cuda() for i in range(len(quants_predl)))
                            # # print(len(quants_transl[0]), flush=True)
                            # transl = (torch.randint(0, 512, [1, 1]).cuda(), )
                            # else:
                            #     y = quants_predl[0].clone()
                            #     x = quants_predf[0][:, :1].clone()
                            
                            x = (torch.ones(1,1).long().cuda()*19, 
                                    torch.ones(1,1).long().cuda()*41, 
                                    torch.ones(1,1).long().cuda()*268,  
                                    torch.ones(1,1).long().cuda()*197)
                            # x = tuple(torch.randint(0, 512, [1, 1]).cuda() for i in range(len(quants_predl)))
                            # print(len(quants_transl[0]), flush=True)
                            transl = (torch.ones(1,1).long().cuda()*321, )

                            zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                            pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                            lf_transl = transl_vqvae.module.decode(transl_z)

                            # print(pose_sample.size(),flush=True)
                            pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                            pose_sample = pose_sample.cpu().data.numpy()

                                # the root of left hand
                            left_twist = pose_sample[:, :, 60:63]
                            # 25,40
                            pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_sample[:, :, 63:66]
                            # 40,55
                            pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_sample[:, :, :3]  # the root
                            pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_sample[:, :, :3] = root
                            results.append(pose_sample.copy())
                            if isinstance(zs, tuple):
                                quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                            else:
                                quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                            pose_seql[:, :, :3] = transll
                            pose_seql = pose_seql.cpu().data.numpy()
                            left_twist = pose_seql[:, :, 60:63]
                            # 25,40
                            pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_seql[:, :, 63:66]
                            # 40,55
                            pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_seql[:, :, :3]  # the root
                            pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_seql[:, :, :3] = root
                            leaders.append(pose_seql.copy())
                    else:
                        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                            # Prepare data
                            # if hasattr(config, 'demo') and config.demo:
                            #     music_seq = batch_eval['music'].to(self.device)
                            #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                            # else:
                            music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                            music_seq = music_seq.to(self.device)
                            pose_seql = pose_seql.to(self.device)
                            pose_seqf = pose_seqf.to(self.device)
                            
                            fname = batch_eval['fname'][0]
                            dance_names.append(fname)

                            
                            
                            lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                            transll = pose_seql[:, :, :3].clone()
                            transll = transll - transll[:, :1, :3].clone()

                            pose_seql[:, :, :3] = 0
                            pose_seqf[:, :, :3] = 0

                            quants_predl = vqvae.module.encode(pose_seql)
                            quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                            quants_transl = transl_vqvae.module.encode(lftransl)


                            if isinstance(quants_predf, tuple):
                                y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
                                x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
                                # print(len(quants_transl[0]), flush=True)
                                transl = (quants_transl[0][:, :1], )
                            else:
                                y = quants_predl[0].clone()
                                x = quants_predf[0][:, :1].clone()
                            
                            if hasattr(config, 'random_init_test') and config.random_init_test:
                                if isinstance(quants, tuple):
                                    for iij in range(len(x)):
                                        x[iij][:, 0] = torch.randint(512, (1, ))
                                else:
                                    x[:, 0] = torch.randint(512, (1, ))

                            zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                            pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                            lf_transl = transl_vqvae.module.decode(transl_z)

                            # print(pose_sample.size(),flush=True)
                            pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                            pose_sample = pose_sample.cpu().data.numpy()

                                # the root of left hand
                            left_twist = pose_sample[:, :, 60:63]
                            # 25,40
                            pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_sample[:, :, 63:66]
                            # 40,55
                            pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_sample[:, :, :3]  # the root
                            pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_sample[:, :, :3] = root
                            results.append(pose_sample.copy())
                            if isinstance(zs, tuple):
                                quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                            else:
                                quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                            pose_seql[:, :, :3] = transll
                            pose_seql = pose_seql.cpu().data.numpy()
                            left_twist = pose_seql[:, :, 60:63]
                            # 25,40
                            pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_seql[:, :, 63:66]
                            # 40,55
                            pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_seql[:, :, :3]  # the root
                            pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_seql[:, :, :3] = root
                            leaders.append(pose_seql.copy())

                    visualize(results, leaders, config.testing, self.visdir, dance_names, epoch_i, None)
                    
                gpt.train()
                gpt.module.freeze_drop()
            self.schedular.step()

    def train2(self):
        vqvae = self.model.eval()
        gpt = self.model2.train()
        gpt.module.freeze_drop()
        transl_vqvae = self.model3.eval()
        # qnet = self.modelq.train()

        reward_fun = self.reward
        config = self.config
        ddm = []
        if hasattr(config, 'demo') and config.demo:
            ddm = True
        else:
            ddm = False
        data = self.config.data
        # criterion = nn.MSELoss()
        labeled_data = self.train_loader
        test_loader = self.test_loader
        optimizer = self.optimizer
        # optimizer_q = self.optimizer_q
        log = Logger(self.config, self.expdir)
        updates = 0
        
        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'])
        checkpoint = torch.load(config.transl_vqvae_weight)
        transl_vqvae.load_state_dict(checkpoint['model'])

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'])
        # if hasattr(config, 'init_weight_q') and config.init_weight_q is not None and config.init_weight_q is not '':
        #     print('Use pretrained model!')
        #     print(config.init_weight_q)  
        #     checkpoint = torch.load(config.init_weight_q)
        #     gpt.module.load_state_dict(checkpoint['model'], strict=False)
            
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        
        # maintain alll sequences for replay buffer
        sample_sequences = []
        

        # print('Here!!!!!!!!!!!!!!!!!!80', flush=True)
        
        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(self.train_loader))
            # use current weight to generate samples 
            with torch.no_grad():
                gpt.eval()
                dance_names = []
                quants_out = []

                reward_up_stat = []
                reward_down_stat = []
                reward_lhand_stat = []
                reward_rhand_stat = []
                reward_transl_stat = []


                demo_flag = True if hasattr(config, 'use_demo') and config.use_demo else False 
                
                if demo_flag:
                    for i_eval, batch_eval in enumerate(tqdm(self.demo_loader, desc='Generating Dance Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
                        music_seq, pose_seql = batch_eval['music'], batch_eval['pos3dl']
                        music_seq = music_seq.to(self.device)
                        pose_seql = pose_seql.to(self.device)
                        
                        
                        fname = batch_eval['fname'][0]
                        dance_names.append(fname)

                        
                        
                        # lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                        transll = pose_seql[:, :, :3].clone()
                        transll = transll - transll[:, :1, :3].clone()

                        pose_seql[:, :, :3] = 0
                        # pose_seqf[:, :, :3] = 0

                        quants_predl = vqvae.module.encode(pose_seql)
                        # quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                        # quants_transl = transl_vqvae.module.encode(lftransl)


                        # if isinstance(quants_predf, tuple):
                        y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predl)))
                        x = tuple(torch.randint(56, 57, [1, 1]).cuda() for i in range(len(quants_predl)))
                        # print(len(quants_transl[0]), flush=True)
                        transl = (torch.randint(56, 57, [1, 1]).cuda(), )
                        # else:
                        #     y = quants_predl[0].clone()
                        #     x = quants_predf[0][:, :1].clone()
                        
                        if hasattr(config, 'random_init_test') and config.random_init_test:
                            if isinstance(quants, tuple):
                                for iij in range(len(x)):
                                    x[iij][:, 0] = torch.randint(512, (1, ))
                            else:
                                x[:, 0] = torch.randint(512, (1, ))

                        zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                        pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                        lf_transl = transl_vqvae.module.decode(transl_z)

                        # print(pose_sample.size(),flush=True)
                        pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                        pose_sample = pose_sample.cpu().data.numpy()

                            # the root of left hand
                        left_twist = pose_sample[:, :, 60:63]
                        # 25,40
                        pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_sample[:, :, 63:66]
                        # 40,55
                        pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_sample[:, :, :3]  # the root
                        pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_sample[:, :, :3] = root
                        # results.append(pose_sample.copy())
                        # if isinstance(zs, tuple):
                        #     quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                        # else:
                        #     quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                        pose_seql[:, :, :3] = transll
                        pose_seql = pose_seql.cpu().data.numpy()
                        left_twist = pose_seql[:, :, 60:63]
                        # 25,40
                        pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_seql[:, :, 63:66]
                        # 40,55
                        pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_seql[:, :, :3]  # the root
                        pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_seql[:, :, :3] = root
                      
                        rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
                        N, T = rewards[0].shape
                        rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).min(axis=-1) for ii in range(len(rewards)))

                        reward_up_stat.append(np.mean(rewards[0]))
                        reward_down_stat.append(np.mean(rewards[1]))
                        reward_lhand_stat.append(np.mean(rewards[2]))
                        reward_rhand_stat.append(np.mean(rewards[3]))
                        reward_transl_stat.append(np.mean(rewards[4]))

                        zs = tuple(zs[ii][0].cpu().long().data.numpy()[0] for ii in range(len(zs)))
                        y = tuple(y[ii][0].cpu().long().data.numpy() for ii in range(len(y)))
                        # print(len(music_seq), len(y), len(y[0][0]), len(zs), len(zs[0][0]), len(rewards[0][0]), flush=True)
                        transl_z = transl_z[0].cpu().data.numpy()[0]
                        # print(y, zs, transl_z, rewards, flush=True)
                        sample_sequences.append((music_seq.cpu().data.numpy()[0], y, zs+(transl_z,), rewards))
                else:
                    for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                        
                        music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                        music_seq = music_seq.to(self.device)
                        pose_seql = pose_seql.to(self.device)
                        pose_seqf = pose_seqf.to(self.device)
                        
                        fname = batch_eval['fname'][0]
                        dance_names.append(fname)

                        lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                        transll = pose_seql[:, :, :3].clone()
                        transll = transll - transll[:, :1, :3].clone()

                        pose_seql[:, :, :3] = 0
                        pose_seqf[:, :, :3] = 0

                        quants_predl = vqvae.module.encode(pose_seql)
                        quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                        quants_transl = transl_vqvae.module.encode(lftransl)

                        y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
                        x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
                        # print(len(quants_transl[0]), flush=True)
                        transl = (quants_transl[0][:, :1], )

                        zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                        pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                        lf_transl = transl_vqvae.module.decode(transl_z)

                        # print(pose_sample.size(),flush=True)
                        pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                        pose_sample = pose_sample.cpu().data.numpy()

                            # the root of left hand
                        left_twist = pose_sample[:, :, 60:63]
                        # 25,40
                        pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_sample[:, :, 63:66]
                        # 40,55
                        pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_sample[:, :, :3]  # the root
                        pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_sample[:, :, :3] = root
                        
                        # if isinstance(zs, tuple):
                        #     quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                        # else:
                        #     quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                        pose_seql[:, :, :3] = transll
                        pose_seql = pose_seql.cpu().data.numpy()
                        left_twist = pose_seql[:, :, 60:63]
                        # 25,40
                        pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                        # the root of right hand
                        right_twist = pose_seql[:, :, 63:66]
                        # 40,55
                        pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                        root = pose_seql[:, :, :3]  # the root
                        pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                        pose_seql[:, :, :3] = root
                        
                        rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
                        N, T = rewards[0].shape
                        rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).min(axis=-1) for ii in range(len(rewards)))

                        reward_up_stat.append(np.mean(rewards[0]))
                        reward_down_stat.append(np.mean(rewards[1]))
                        reward_lhand_stat.append(np.mean(rewards[2]))
                        reward_rhand_stat.append(np.mean(rewards[3]))
                        reward_transl_stat.append(np.mean(rewards[4]))

                        zs = tuple(zs[ii][0].cpu().long().data.numpy()[0] for ii in range(len(zs)))
                        y = tuple(y[ii][0].cpu().long().data.numpy() for ii in range(len(y)))
                        # print(len(music_seq), len(y), len(y[0][0]), len(zs), len(zs[0][0]), len(rewards[0][0]), flush=True)
                        transl_z = transl_z[0].cpu().data.numpy()[0]
                        # print(y, zs, transl_z, rewards, flush=True)
                        sample_sequences.append((music_seq.cpu().data.numpy()[0], y, zs+(transl_z,), rewards))
                
            # buff = 0
            replay_buffer = self._build_rl_loader(sample_sequences)

            r_up_mean, r_down_mean, r_lhand_mean, r_rhand_mean, r_transl_mean = np.mean(reward_up_stat), np.mean(reward_down_stat), np.mean(reward_lhand_stat), np.mean(reward_rhand_stat), np.mean(reward_transl_stat)
            # print(len(labeled_data), flush=True)
            # print('Here!!!!!!!!!!!!!!!!!!166', flush=True)
            for iter_ii, batch_rl in enumerate(replay_buffer):

                ## check whether the reward during training is correct
                music_feat, y, quants_input, quants_target, rewards = batch_rl['music_feat'], batch_rl['quants_l'], batch_rl['quants_input'], batch_rl['quants_target'], batch_rl['rewards']
               
                # print('cond size', cond[0].size(), flush=True)
                # print('quants input szie', quants_input[0])
                # print(quants_target)
                # print(quants_target[0]) 
                # print(rewards)
                # print(rewards[0], flush=True)

                music_feat = music_feat.to(self.device)
                y = tuple(y[ii].to(self.device) for ii in range(len(y)))
                quants_input = tuple(quants_input[ii].to(self.device) for ii in range(len(quants_input)))
                quants_target = tuple(quants_target[ii].to(self.device) for ii in range(len(quants_target)))
                rewards = tuple(rewards[ii].to(self.device) for ii in range(len(rewards)))
                cond = (music_feat, ) + y 

                # print(quants_input[-1], quants_target[-1], rewards[-1], flush=True)

                # print(music_feat.size(), flush=True)
                # print(y[0].size(), flush=True)
                # print(quants_input[0].size(), flush=True)
                # print(quants_target[0].size(), flush=True)
                # print(rewards[0].size(), flush=True)
                # with torch.no_grad():
                #     # state = gpt.module.state(quants_input)
                #     gpt.eval()
                #     qnet.eval()
                #     state = gpt.state(quants_input, cond)
                #     probs, _ = qnet(state)
                    # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs
                    # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                    # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                    # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                    # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                    # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                    # print('transl_before-train:', ix_transl, flush=True)

                # with torch.no_grad():
                #     gpt.eval()
                #     # logits, _ = gpt(quants_input, cond=cond)

                #     # print(logits[-1])
                #     # print(torch.mean(logits[-1], dim=-1))
                #     # print(logits[-1].min(dim=-1))
                #     # print(logits[-1].max(dim=-1),)
                #     # print(logits[-1].)
                #     state = gpt.module.state(quants_input, cond)
                #     probs_pi, _ = gpt.module.actor(state)
                #     # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs_pi
                #     # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                #     # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                #     # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                #     # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                #     # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                #     # print('transl_pi:', ix_transl, flush=True)
                #     # print()
                # qnet.train()
                gpt.train()

                optimizer.zero_grad()
                # state = state.clone().detach()
                # state.requires_grad = True
                # _, q_loss, _ = qnet(state.clone().detach(), quants_target, rewards, probs_pi)
        
                _, loss = gpt(quants_input, cond, rewards, quants_target)

                loss.mean().backward()
                optimizer.step()
                
                # check_loss2 = q_loss.clone().item()
                # print(check_loss1, check_loss2, flush=True)

                # 2. q --> pi
                # with torch.no_grad():
                #     # state = gpt.module.state(quants_input)
                #     qnet.eval()
                #     probs, _, _ = qnet(state)
                #     # prob_up, prob_down, prob_lhand, prob_rhand, prob_transl = probs
                #     # _, ix_up = torch.topk(prob_up, k=1, dim=-1)
                #     # _, ix_down = torch.topk(prob_down, k=1, dim=-1)
                #     # _, ix_lhand = torch.topk(prob_lhand, k=1, dim=-1)
                #     # _, ix_rhand = torch.topk(prob_rhand, k=1, dim=-1)
                #     # _, ix_transl = torch.topk(prob_transl, k=1, dim=-1)
                #     # print('transl:', ix_transl, flush=True)
                # gpt.train()
                # optimizer.zero_grad()
                # gpt.module.freeze_drop()
                # _, pi_loss = gpt.module.actor(state, probs)
    
                # 3. imitation learning loss
                # gpt.train()
                
                # batch_labeled = next(iter(labeled_data))
                # music_seq, pose_seql, pose_seqf  = batch_labeled['music'], batch_labeled['pos3dl'], batch_labeled['pos3df'] 
                # music_seq = music_seq.to(self.device)
                # pose_seql, pose_seqf = pose_seql.to(self.device), pose_seqf.to(self.device)

                # transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0

                # # music
                # pose_seql[:, :, :3] = 0
                # pose_seqf[:, :, :3] = 0
                
                # print(pose_seq.size())
                

                # with torch.no_grad():
                #     quants_predl = vqvae.module.encode(pose_seql)
                #     quants_predf = vqvae.module.encode(pose_seqf)
                #     quants_transl = transl_vqvae.module.encode(transl)

                #     quants_cond = tuple(quants_predl[ii][0][:, :config.motion_len+config.look_forward_size].clone().detach() for ii in range(len(quants_predl)))
                #     quants_input = tuple(quants_predf[ii][0][:, :config.motion_len].clone().detach() for ii in range(len(quants_predf)))
                #     quants_target = tuple(quants_predf[ii][0][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_predf)))
                #     quants_transl_input = tuple(quants_transl[ii][:, :config.motion_len].clone().detach() for ii in range(len(quants_transl)))
                #     quants_transl_target = tuple(quants_transl[ii][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_transl)))
                                    
                # output, bc_loss = gpt(quants_input+quants_transl_input, (music_seq[:, config.music_motion_rate:config.music_motion_rate+config.music_len, ], ) + quants_cond, quants_target+quants_transl_target)
                
                # policy_loss = pi_loss
                # #  + config.lambda_bc * bc_loss

                # policy_loss.mean().backward()

                # # update parameters
                # optimizer.step()

                stats = {
                    'updates': updates,
                    'reward_down': r_down_mean,
                    'reward_tranl': r_transl_mean,
                    'loss': loss.mean().item(),
                    # 'bc_loss': bc_loss.mean().item(),
                    # 'pi_loss': pi_loss.mean().item()
                    
                    # 'entropy': entropy.clone().detach().mean()
                }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                # 'modelq': qnet.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0:
                print('validation...')

                with torch.no_grad():
                    gpt.eval()

                    self.device = torch.device('cuda' if config.cuda else 'cpu')
                    
                    results = []
                    leaders = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants_out = {}
                    dance_names = []
                    
                    if demo_flag:
                        for i_eval, batch_eval in enumerate(tqdm(self.demo_loader, desc='Generating Demo Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
                            music_seq, pose_seql = batch_eval['music'], batch_eval['pos3dl']
                            music_seq = music_seq.to(self.device)
                            pose_seql = pose_seql.to(self.device)
                            
                            
                            fname = batch_eval['fname'][0]
                            dance_names.append(fname)

                            
                            
                            # lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                            transll = pose_seql[:, :, :3].clone()
                            transll = transll - transll[:, :1, :3].clone()

                            pose_seql[:, :, :3] = 0
                            # pose_seqf[:, :, :3] = 0

                            quants_predl = vqvae.module.encode(pose_seql)
                            # quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                            # quants_transl = transl_vqvae.module.encode(lftransl)


                            # if isinstance(quants_predf, tuple):
                            y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predl)))
                            x = tuple(torch.randint(0, 512, [1, 1]).cuda() for i in range(len(quants_predl)))
                            # print(len(quants_transl[0]), flush=True)
                            transl = (torch.randint(0, 512, [1, 1]).cuda(), )
                            # else:
                            #     y = quants_predl[0].clone()
                            #     x = quants_predf[0][:, :1].clone()
                            
                            if hasattr(config, 'random_init_test') and config.random_init_test:
                                if isinstance(quants, tuple):
                                    for iij in range(len(x)):
                                        x[iij][:, 0] = torch.randint(512, (1, ))
                                else:
                                    x[:, 0] = torch.randint(512, (1, ))

                            zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                            pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                            lf_transl = transl_vqvae.module.decode(transl_z)

                            # print(pose_sample.size(),flush=True)
                            pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                            pose_sample = pose_sample.cpu().data.numpy()

                                # the root of left hand
                            left_twist = pose_sample[:, :, 60:63]
                            # 25,40
                            pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_sample[:, :, 63:66]
                            # 40,55
                            pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_sample[:, :, :3]  # the root
                            pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_sample[:, :, :3] = root
                            results.append(pose_sample.copy())
                            if isinstance(zs, tuple):
                                quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                            else:
                                quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                            pose_seql[:, :, :3] = transll
                            pose_seql = pose_seql.cpu().data.numpy()
                            left_twist = pose_seql[:, :, 60:63]
                            # 25,40
                            pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_seql[:, :, 63:66]
                            # 40,55
                            pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_seql[:, :, :3]  # the root
                            pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_seql[:, :, :3] = root
                            leaders.append(pose_seql.copy())

                    else:
                        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                            # Prepare data
                            # if hasattr(config, 'demo') and config.demo:
                            #     music_seq = batch_eval['music'].to(self.device)
                            #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                            # else:
                            music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                            music_seq = music_seq.to(self.device)
                            pose_seql = pose_seql.to(self.device)
                            pose_seqf = pose_seqf.to(self.device)
                            
                            fname = batch_eval['fname'][0]
                            dance_names.append(fname)

                            
                            
                            lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                            transll = pose_seql[:, :, :3].clone()
                            transll = transll - transll[:, :1, :3].clone()

                            pose_seql[:, :, :3] = 0
                            pose_seqf[:, :, :3] = 0

                            quants_predl = vqvae.module.encode(pose_seql)
                            quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                            quants_transl = transl_vqvae.module.encode(lftransl)


                            if isinstance(quants_predf, tuple):
                                y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
                                x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
                                # print(len(quants_transl[0]), flush=True)
                                transl = (quants_transl[0][:, :1], )
                            else:
                                y = quants_predl[0].clone()
                                x = quants_predf[0][:, :1].clone()
                            
                            if hasattr(config, 'random_init_test') and config.random_init_test:
                                if isinstance(quants, tuple):
                                    for iij in range(len(x)):
                                        x[iij][:, 0] = torch.randint(512, (1, ))
                                else:
                                    x[:, 0] = torch.randint(512, (1, ))

                            zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                            pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                            lf_transl = transl_vqvae.module.decode(transl_z)

                            # print(pose_sample.size(),flush=True)
                            pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                            pose_sample = pose_sample.cpu().data.numpy()

                                # the root of left hand
                            left_twist = pose_sample[:, :, 60:63]
                            # 25,40
                            pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_sample[:, :, 63:66]
                            # 40,55
                            pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_sample[:, :, :3]  # the root
                            pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_sample[:, :, :3] = root
                            results.append(pose_sample.copy())
                            if isinstance(zs, tuple):
                                quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                            else:
                                quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                            pose_seql[:, :, :3] = transll
                            pose_seql = pose_seql.cpu().data.numpy()
                            left_twist = pose_seql[:, :, 60:63]
                            # 25,40
                            pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                            # the root of right hand
                            right_twist = pose_seql[:, :, 63:66]
                            # 40,55
                            pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                            root = pose_seql[:, :, :3]  # the root
                            pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                            pose_seql[:, :, :3] = root
                            leaders.append(pose_seql.copy())

                    visualize(results, leaders, config.testing, self.visdir, dance_names, epoch_i, None)
                    
                gpt.train()
                gpt.module.freeze_drop()
            self.schedular.step()
            # self.schedular_q.step()


    def demo(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()
            transl_vqvae = self.model3.eval()

            config = self.config

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=True)
            checkpoint_tranl = torch.load(config.transl_vqvae_weight)
            transl_vqvae.load_state_dict(checkpoint_tranl['model'], strict=True)

            epoch_tested = config.testing.ckpt_epoch
            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=True)
            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            
            print("Evaluation...", flush=True)
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            gpt.eval()
            vqvae.eval()

            results = []
            leaders = []
            random_id = 0  # np.random.randint(0, 1e4)
            quants_out = {}
            dance_names = []
            
            
            for i_eval, batch_eval in enumerate(tqdm(self.demo_loader, desc='Generating Dance Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
                music_seq, pose_seql = batch_eval['music'], batch_eval['pos3dl']
                music_seq = music_seq.to(self.device)
                pose_seql = pose_seql.to(self.device)
                
                
                fname = batch_eval['fname'][0]
                dance_names.append(fname)

                
                
                # lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                transll = pose_seql[:, :, :3].clone()
                transll = transll - transll[:, :1, :3].clone()

                pose_seql[:, :, :3] = 0
                # pose_seqf[:, :, :3] = 0

                quants_predl = vqvae.module.encode(pose_seql) 
                # pose_seql , _, _ = vqvae.module.decode(quants_predl)
                # quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                # quants_transl = transl_vqvae.module.encode(lftransl)


                # if isinstance(quants_predf, tuple):
                y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predl)))
                x = (torch.ones(1,1).long().cuda()*19, 
                          torch.ones(1,1).long().cuda()*41, 
                          torch.ones(1,1).long().cuda()*268,  
                          torch.ones(1,1).long().cuda()*197)
                # x = tuple(torch.randint(0, 512, [1, 1]).cuda() for i in range(len(quants_predl)))
                # print(len(quants_transl[0]), flush=True)
                transl = (torch.ones(1,1).long().cuda()*321, )
                # x = tuple(torch.randint(0, 512, [1, 1]).cuda() for i in range(len(quants_predl)))
                # # print(len(quants_transl[0]), flush=True)
                # transl = (torch.randint(0, 512, [1, 1]).cuda(), )
                # else:
                #     y = quants_predl[0].clone()
                #     x = quants_predf[0][:, :1].clone()
                
                if hasattr(config, 'random_init_test') and config.random_init_test:
                    if isinstance(quants, tuple):
                        for iij in range(len(x)):
                            x[iij][:, 0] = torch.randint(512, (1, ))
                    else:
                        x[:, 0] = torch.randint(512, (1, ))

                zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                lf_transl = transl_vqvae.module.decode(transl_z)

                # print(pose_sample.size(),flush=True)
                pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                pose_sample = pose_sample.cpu().data.numpy()

                    # the root of left hand
                left_twist = pose_sample[:, :, 60:63]
                # 25,40
                pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_sample[:, :, 63:66]
                # 40,55
                pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_sample[:, :, :3]  # the root
                pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_sample[:, :, :3] = root
                results.append(pose_sample.copy())
                if isinstance(zs, tuple):
                    quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                else:
                    quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                pose_seql[:, :, :3] = transll
                pose_seql = pose_seql.cpu().data.numpy()
                left_twist = pose_seql[:, :, 60:63]
                # 25,40
                pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_seql[:, :, 63:66]
                # 40,55
                pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_seql[:, :, :3]  # the root
                pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_seql[:, :, :3] = root
                leaders.append(pose_seql.copy())



                rotmatf = rotmat_sample.cpu().data.numpy().reshape([-1, 3, 3])
                rotmatf = get_closest_rotmat(rotmatf)
                smpl_poses_f = rotmat2aa(rotmatf).reshape(-1, 55, 3)

                rotmat_leader = batch_eval['rotmatl']
                rotmatl = rotmat_leader.cpu().data.numpy().reshape([-1, 3, 3])
                rotmatl = get_closest_rotmat(rotmatl)
                smpl_poses_l = rotmat2aa(rotmatl).reshape(-1, 55, 3)
                translf = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                save_smplx(smpl_poses_f, translf.reshape([-1, 3]).cpu().data.numpy(), smpl_poses_l, transll.reshape([-1, 3]).cpu().data.numpy(), config.testing, self.demodir, epoch_tested, fname)
                save_pos3d(pose_sample, pose_seql, config.testing, self.demodir, epoch_tested, fname)
            # for ii in range(len(all_rewards)):
            #     print(dance_names[ii], np.mean(all_rewards[ii]), flush=True)
            visualize(results, leaders, config.testing, self.demodir, dance_names, epoch_tested, None)
            




    def eval(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()
            transl_vqvae = self.model3.eval()

            config = self.config

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=True)
            checkpoint_tranl = torch.load(config.transl_vqvae_weight)
            transl_vqvae.load_state_dict(checkpoint_tranl['model'], strict=True)

            epoch_tested = config.testing.ckpt_epoch
            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=True)
            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            
            print("Evaluation...", flush=True)
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            gpt.eval()

            results = []
            leaders = []
            random_id = 0  # np.random.randint(0, 1e4)
            quants_out = {}
            dance_names = []
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
                music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                music_seq = music_seq.to(self.device)
                pose_seql = pose_seql.to(self.device)
                pose_seqf = pose_seqf.to(self.device)
                
                fname = batch_eval['fname'][0]
                dance_names.append(fname)

                
                
                lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
                transll = pose_seql[:, :, :3].clone()
                transll = transll - transll[:, :1, :3].clone()

                pose_seql[:, :, :3] = 0
                pose_seqf[:, :, :3] = 0

                quants_predl = vqvae.module.encode(pose_seql)
                quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
                quants_transl = transl_vqvae.module.encode(lftransl)


                if isinstance(quants_predf, tuple):
                    y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
                    x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
                    # print(len(quants_transl[0]), flush=True)
                    transl = (quants_transl[0][:, :1], )
                else:
                    y = quants_predl[0].clone()
                    x = quants_predf[0][:, :1].clone()
                
                if hasattr(config, 'random_init_test') and config.random_init_test:
                    if isinstance(quants, tuple):
                        for iij in range(len(x)):
                            x[iij][:, 0] = torch.randint(512, (1, ))
                    else:
                        x[:, 0] = torch.randint(512, (1, ))

                zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
                lf_transl = transl_vqvae.module.decode(transl_z)

                # print(pose_sample.size(),flush=True)
                pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)].clone() + lf_transl.clone() / 20.0
                # global_vel = vel_sample.clone()
                # pose_sample[:, 0, :3] = 0
                # for iii in range(1, pose_sample.size(1)):
                #     pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]
                pose_sample = pose_sample.cpu().data.numpy()

                    # the root of left hand
                left_twist = pose_sample[:, :, 60:63]
                # 25,40
                pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_sample[:, :, 63:66]
                # 40,55
                pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_sample[:, :, :3]  # the root
                pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_sample[:, :, :3] = root
                results.append(pose_sample.copy())
                if isinstance(zs, tuple):
                    quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
                else:
                    quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                pose_seql[:, :, :3] = transll.clone()
                pose_seql = pose_seql.cpu().data.numpy()
                left_twist = pose_seql[:, :, 60:63]
                # 25,40
                pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_seql[:, :, 63:66]
                # 40,55
                pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_seql[:, :, :3]  # the root
                pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_seql[:, :, :3] = root
                leaders.append(pose_seql.copy())
                # rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
                # N, T = rewards[0].shape
                # rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).mean(axis=-1) for ii in range(len(rewards)))
                # all_rewards.append(rewards)

                rotmatf = rotmat_sample.cpu().data.numpy().reshape([-1, 3, 3])
                rotmatf = get_closest_rotmat(rotmatf)
                smpl_poses_f = rotmat2aa(rotmatf).reshape(-1, 55, 3)

                rotmat_leader = batch_eval['rotmatl']
                rotmatl = rotmat_leader.cpu().data.numpy().reshape([-1, 3, 3])
                rotmatl = get_closest_rotmat(rotmatl)
                smpl_poses_l = rotmat2aa(rotmatl).reshape(-1, 55, 3)
                translf = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

                save_smplx(smpl_poses_f, translf.reshape([-1, 3]).cpu().data.numpy(), smpl_poses_l, transll.reshape([-1, 3]).cpu().data.numpy(), config.testing, self.evaldir, epoch_tested, fname)
                save_pos3d(pose_sample, pose_seql, config.testing, self.evaldir, epoch_tested, fname)
            # for ii in range(len(all_rewards)):
            #     print(dance_names[ii], np.mean(all_rewards[ii]), flush=True)
            visualize(results, leaders, config.testing, self.evaldir, dance_names, epoch_tested, None)
            


    # def eval(self):
    #     with torch.no_grad():
    #         vqvae = self.model.eval()
    #         gpt = self.model2.eval()
    #         transl_vqvae = self.model3.eval()
    #         reward_fun = self.reward

    #         config = self.config

    #         checkpoint = torch.load(config.vqvae_weight)
    #         vqvae.load_state_dict(checkpoint['model'], strict=False)
    #         checkpoint_tranl = torch.load(config.transl_vqvae_weight)
    #         transl_vqvae.load_state_dict(checkpoint_tranl['model'], strict=False)

    #         epoch_tested = config.testing.ckpt_epoch
    #         checkpoint = torch.load(config.vqvae_weight)
    #         vqvae.load_state_dict(checkpoint['model'], strict=False)
    #         ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
    #         self.device = torch.device('cuda' if config.cuda else 'cpu')
            
    #         print("Evaluation...", flush=True)
    #         checkpoint = torch.load(ckpt_path)
    #         gpt.load_state_dict(checkpoint['model'])
    #         gpt.eval()

    #         results = []
    #         leaders = []
    #         random_id = 0  # np.random.randint(0, 1e4)
    #         quants_out = {}
    #         dance_names = []
            
    #         all_rewards = []
            
    #         for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
    #             # Prepare data
    #             # if hasattr(config, 'demo') and config.demo:
    #             #     music_seq = batch_eval['music'].to(self.device)
    #             #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
    #             # else:
    #             music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
    #             music_seq = music_seq.to(self.device)
    #             pose_seql = pose_seql.to(self.device)
    #             pose_seqf = pose_seqf.to(self.device)
                
    #             fname = batch_eval['fname'][0]
    #             dance_names.append(fname)

                
                
    #             lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
    #             transll = pose_seql[:, :, :3].clone()
    #             transll = transll - transll[:, :1, :3].clone()

    #             pose_seql[:, :, :3] = 0
    #             pose_seqf[:, :, :3] = 0

    #             quants_predl = vqvae.module.encode(pose_seql)
    #             quants_predf = vqvae.module.encode(pose_seqf[:, :config.ds_rate])
    #             quants_transl = transl_vqvae.module.encode(lftransl)


    #             if isinstance(quants_predf, tuple):
    #                 y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
    #                 x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
    #                 # print(len(quants_transl[0]), flush=True)
    #                 transl = (quants_transl[0][:, :1], )
    #             else:
    #                 y = quants_predl[0].clone()
    #                 x = quants_predf[0][:, :1].clone()
                
    #             if hasattr(config, 'random_init_test') and config.random_init_test:
    #                 if isinstance(quants, tuple):
    #                     for iij in range(len(x)):
    #                         x[iij][:, 0] = torch.randint(512, (1, ))
    #                 else:
    #                     x[:, 0] = torch.randint(512, (1, ))

    #             zs, transl_z = gpt.module.sample(x+transl, cond=(music_seq[:, config.music_motion_rate:],)+y, shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

    #             pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
    #             lf_transl = transl_vqvae.module.decode(transl_z)

    #             # print(pose_sample.size(),flush=True)
    #             pose_sample[:, :, :3] = transll[:, :lf_transl.size(1)].clone() + lf_transl.clone() / 20.0

    #             pose_sample = pose_sample.cpu().data.numpy()

    #                 # the root of left hand
    #             left_twist = pose_sample[:, :, 60:63]
    #             # 25,40
    #             pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

    #             # the root of right hand
    #             right_twist = pose_sample[:, :, 63:66]
    #             # 40,55
    #             pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

    #             root = pose_sample[:, :, :3]  # the root
    #             pose_sample = pose_sample + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
    #             pose_sample[:, :, :3] = root
    #             results.append(pose_sample.copy())
    #             if isinstance(zs, tuple):
    #                 quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs))) 
    #             else:
    #                 quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

    #             pose_seql[:, :, :3] = transll.clone()
    #             pose_seql = pose_seql.cpu().data.numpy()
    #             left_twist = pose_seql[:, :, 60:63]
    #             # 25,40
    #             pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

    #             # the root of right hand
    #             right_twist = pose_seql[:, :, 63:66]
    #             # 40,55
    #             pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

    #             root = pose_seql[:, :, :3]  # the root
    #             pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
    #             pose_seql[:, :, :3] = root
    #             leaders.append(pose_seql.copy())
    #             rewards = reward_fun(music_seq, torch.tensor(pose_seql).float().cuda(), torch.tensor(pose_sample).float().cuda(), vel_sample) # NxTx1
    #             N, T = rewards[0].shape
    #             rewards = tuple(rewards[ii].cpu().data.numpy()[0].reshape(T//config.ds_rate, config.ds_rate).mean(axis=-1) for ii in range(len(rewards)))
    #             all_rewards.append(rewards)

    #             rotmatf = rotmat_sample.cpu().data.numpy().reshape([-1, 3, 3])
    #             rotmatf = get_closest_rotmat(rotmatf)
    #             smpl_poses_f = rotmat2aa(rotmatf).reshape(-1, 55, 3)

    #             rotmat_leader = batch_eval['rotmatl']
    #             rotmatl = rotmat_leader.cpu().data.numpy().reshape([-1, 3, 3])
    #             rotmatl = get_closest_rotmat(rotmatl)
    #             smpl_poses_l = rotmat2aa(rotmatl).reshape(-1, 55, 3)
    #             translf = transll[:, :lf_transl.size(1)] + lf_transl / 20.0

    #             save_smplx(smpl_poses_f, translf.reshape([-1, 3]).cpu().data.numpy(), smpl_poses_l, transll.reshape([-1, 3]).cpu().data.numpy(), config.testing, self.evaldir, epoch_tested, fname)
            
    #         for ii in range(len(all_rewards)):
    #             print(dance_names[ii], np.mean(all_rewards[ii]), flush=True)
    #         visualize(results, leaders, config.testing, self.evaldir, dance_names, epoch_tested, all_rewards)
    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            pose_seq_eval = batch_eval

            results.append(pose_seq_eval)
        visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)

    def analyze_code(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        training_data = self.training_data
        all_quants = None

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        random_id = 0  # np.random.randint(0, 1e4)
        
        for i_eval, batch_eval in enumerate(tqdm(self.training_data, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            pose_seq_eval = batch_eval.to(self.device)

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        print(all_quants)
                    # exit()
        # visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)
        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])

        log = datetime.datetime.now().strftime('%Y-%m-%d')
        plt.savefig(self.histdir1 + '/hist_epoch_' + str(epoch_tested)  + '_%s.jpg' % log)   #图片的存储
        plt.close()

    def sample(self,):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        quants = {}

        results = []

        if hasattr(config, 'analysis_array') and config.analysis_array is not None:
            # print(config.analysis_array)
            names = [str(ii) for ii in config.analysis_array]
            print(names)
            for ii in config.analysis_array:
                print(ii)
                zs =  [(ii * torch.ones((1, self.config.sample_code_length), device='cuda')).long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length), device='cuda')]
                pose_sample = model.module.decode(zs)
                quants['rand_seq_' + str(ii)] = zs[0].cpu().data.numpy()[0]
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]
                results.append(pose_sample)
        visualizeAndWrite(results, config, self.sampledir, names, epoch_tested, quants)

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if not(hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not(hasattr(config, 'need_not_test_data') and config.need_not_train_data):      
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config 
        if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
            print(f'using {config.structure.name} and {config.structure_generate.name} ')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            model_class2 = getattr(models, config.structure_generate.name)
            model2 = model_class2(config.structure_generate)

            model_class3 = getattr(models, config.structure_transl_vqvae.name)
            model3 = model_class3(config.structure_transl_vqvae)

            # model_class_q = getattr(models, config.structure_qnet.name)
            # modelq = model_class_q(config.structure_qnet)

            reward = getattr(models, config.reward.name)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model.float())
        model2 = nn.DataParallel(model2.float())
        model3 = nn.DataParallel(model3.float())
        # modelq = nn.DataParallel(modelq.float())

        self.model3 = model3.cuda()
        self.model2 = model2.cuda()
        self.model = model.cuda()
        # self.modelq = modelq.cuda()
        self.reward = reward

    def _build_train_loader(self):

        data = self.config.data.train
        trainset = DD100lf(data.music_root, data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move, music_dance_rate=1)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=32,
            batch_size=data.batch_size,
            pin_memory=True,
            shuffle=True, 
            drop_last=True,
        )

    def _build_test_loader(self):
        config = self.config
        data = self.config.data.test
        
        testset = DD100lf(data.music_root, data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move, music_dance_rate=1)
        #pdb.set_trace()

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False
            # collate_fn=paired_collate_fn,
        )

        if hasattr(config.data, 'demo'):
            data = self.config.data.demo
            demoset = DD100lfDemo(data.music_root, data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move, music_dance_rate=1)
            self.demo_loader = torch.utils.data.DataLoader(
                demoset,
                batch_size=1,
                shuffle=False
                # collate_fn=paired_collate_fn,
            )
        else:
            print('WULAAA!')
    def _build_rl_loader(self, sample_sequences):

        data = self.config.data.rl

        rl_set = AClf(sample_sequences, interval=data.interval, look_forward=data.look_forward, music_code_rate=data.music_code_rate, expand_rate=data.expand_rate)
        # (self, sequences, interval, look_forward, music_code_rate=4, expand_rate=100):
        return torch.utils.data.DataLoader(
            rl_set,
            num_workers=32,
            batch_size=data.batch_size,
            pin_memory=True,
            shuffle=True, 
            drop_last=True,
        )


        
        
    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        # config_q = self.config.optimizer_q
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
                                             ),
                                             **config.kwargs)
        # self.optimizer_q = optim(itertools.chain(self.modelq.module.parameters(),
        #                                      ),
        #                                      **config_q.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)
        # self.schedular_q = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_q, **config_q.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "npy")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir):
            os.mkdir(self.histdir)

        self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir):
            os.mkdir(self.imgsdir)

        self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir):
            os.mkdir(self.videodir)
        
        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.gtdir = os.path.join(self.expdir, "gt")
        if not os.path.exists(self.gtdir):
            os.mkdir(self.gtdir)

        self.jsondir1 = os.path.join(self.evaldir, "npy")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir1):
            os.mkdir(self.histdir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(os.path.join(self.gtdir, 'videos')):
            os.mkdir(os.path.join(self.gtdir, 'videos'))

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)
        
        self.demodir = os.path.join(self.evaldir, "demo")  # -- imgs, videos, jsons
        if not os.path.exists(self.demodir):
            os.mkdir(self.demodir)
        
        self.demoviddir = os.path.join(self.demodir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.demoviddir):
            os.mkdir(self.demoviddir)

        self.jsondir = os.path.join(self.visdir, "npy")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.jsondir1 = os.path.join(self.evaldir, "npy")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.jsondir2 = os.path.join(self.demodir, "npy")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir2):
            os.mkdir(self.jsondir2)

 