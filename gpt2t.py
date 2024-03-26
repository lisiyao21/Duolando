# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
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
# from models.vqvae import VQVAE
from utils.save import save_smplx, save_pos3d
from utils.log import Logger
from utils.visualize import visualize2 as visualize
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



class MCTall():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        vqvae = self.model.eval()
        gpt = self.model2.train()
        transl_vqvae = self.model3.eval()

        config = self.config
        data = self.config.data
        # criterion = nn.MSELoss()
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0
        
        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'])
        checkpoint_tranl = torch.load(config.transl_vqvae_weight)
        transl_vqvae.load_state_dict(checkpoint_tranl['model'])


        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'])
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')


        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):
                music_seq, pose_seql, pose_seqf  = batch['music'], batch['pos3dl'], batch['pos3df'] 
                music_seq = music_seq.to(self.device)
                pose_seql, pose_seqf = pose_seql.to(self.device), pose_seqf.to(self.device)

                transl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0

                # music

                pose_seql[:, :, :3] = 0
                pose_seqf[:, :, :3] = 0
                
                # print(pose_seq.size())
                optimizer.zero_grad()

                with torch.no_grad():
                    quants_predl = vqvae.module.encode(pose_seql)
                    quants_predf = vqvae.module.encode(pose_seqf)
                    quants_transl = transl_vqvae.module.encode(transl)

                    quants_cond = tuple(quants_predl[ii][0][:, :config.motion_len+config.look_forward_size].clone().detach() for ii in range(len(quants_predl)))
                    quants_input = tuple(quants_predf[ii][0][:, :config.motion_len].clone().detach() for ii in range(len(quants_predf)))
                    quants_target = tuple(quants_predf[ii][0][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_predf)))
                    quants_transl_input = tuple(quants_transl[ii][:, :config.motion_len].clone().detach() for ii in range(len(quants_transl)))
                    quants_transl_target = tuple(quants_transl[ii][:, 1:config.motion_len+1].clone().detach() for ii in range(len(quants_transl)))
                    
                # music_seq.requires_grad = True
                output, loss = gpt(quants_input+quants_transl_input, (music_seq[:, config.music_motion_rate:config.music_motion_rate+config.music_len, ], ) + quants_cond, quants_target+quants_transl_target)
                loss.mean().backward()

                # print(music_seq.grad)

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.mean().item()
                }
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    gpt.eval()
                    results = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants_out = {}
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                        music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
                        music_seq = music_seq.to(self.device)
                        pose_seql = pose_seq.to(self.device)
                        pose_seqf = pose_seq.to(self.device)
                        
                        quantsl = vqvae.module.encode(pose_seql)
                        quantsf = vqvae.module.encode(pose_seqf)
                        # print(pose_seq.size())
                        if isinstance(quantsl, tuple):
                            y = tuple(quantsl[i][0][:, :] for i in range(len(quants)))
                            x = tuple(quantsf[i][0][:, :1] for i in range(len(quants)))
                        else:
                            y = quants[0][:, :]
                            x = quants[0][:, :1]


                        zs = gpt.module.sample(x, cond=(music_seq)+y)
                        
                        pose_sample, rot_sample, vel_sample = vqvae.module.decode(zs)

                        # if config.global_vel:
                        global_vel = vel_sample.clone()
                        pose_sample[:, 0, :3] = 0
                        for iii in range(1, pose_sample.size(1)):
                            pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                        if isinstance(zs, tuple):
                            quants_out[self.dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs)))
                        else:
                            quants_out[self.dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]
                    
                        results.append(pose_sample)
                gpt.train()
            self.schedular.step()

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
                # print(music_seq.mean(), flush=True)
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
                # print('body: ', quants_predf, flush=True)
                quants_transl = transl_vqvae.module.encode(lftransl)
                # print('transl: ', quants_transl, flush=True)


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
            
            
    def visgt(self,):
        config = self.config
        # print(config,flush=True)
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        leaders = []
        dance_names = []
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        
        if hasattr(config, 'demo') and config.demo:
            print('whalala!!', flush=True)
            self.test_loader = self.demo_loader
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
            if hasattr(config, 'demo') and config.demo:
                music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3dl']
            else:
                music_seq, pose_seql, pose_seqf = batch_eval['music'], batch_eval['pos3dl'], batch_eval['pos3df']
            
            # music_seq = music_seq.to(self.device)
            # pose_seql = pose_seql.to(self.device)
            # pose_seqf = pose_seqf.to(self.device)
            
            fname = batch_eval['fname'][0]
            dance_names.append(fname)

            
            
            start_transl = (pose_seqf[:, :1, :3] - pose_seql[:, :1, :3]).clone()
            transll = pose_seql[:, :, :3].clone()
            translf = pose_seqf[:, :, :3].clone()

            transll = transll - transll[:, :1, :3].clone()
            translf = translf - translf[:, :1, :3].clone() + start_transl

            pose_seql[:, :, :3] = 0
            pose_seql = pose_seql.cpu().data.numpy()
            left_twist = pose_seql[:, :, 60:63]
            # 25,40
            pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

            # the root of right hand
            right_twist = pose_seql[:, :, 63:66]
            # 40,55
            pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

            pose_seql[:, :, :3] = transll
            root = pose_seql[:, :, :3]  # the root
            pose_seql = pose_seql + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
            pose_seql[:, :, :3] = root

            
            leaders.append(pose_seql.copy())

            pose_seqf = pose_seqf.cpu().data.numpy()
            left_twist = pose_seqf[:, :, 60:63]
            # 25,40
            pose_seqf[:, :, 75:120] = pose_seqf[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

            # the root of right hand
            right_twist = pose_seqf[:, :, 63:66]
            # 40,55
            pose_seqf[:, :, 120:165] = pose_seqf[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

            pose_seqf[:, :, :3] = translf
            root = pose_seqf[:, :, :3]  # the root
            pose_seqf = pose_seqf + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
            pose_seqf[:, :, :3] = root
            
            results.append(pose_seqf.copy())

            # quants = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]

                    # exit()
        # weights = np.histogram(all_quants, bins=1, range=[0, config.structure.l_bins], normed=False, weights=None, density=None)
        visualize(results, leaders, config.testing, self.gtdir, dance_names, 0, None)
        # visualize(results, leaders, config.testing, self.evaldir, dance_names, epoch_tested, quants_out)


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
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model.float())
        model2 = nn.DataParallel(model2.float())
        model3 = nn.DataParallel(model3.float())

        self.model3 = model3.cuda()
        self.model2 = model2.cuda()
        self.model = model.cuda()

    def _build_train_loader(self):

        data = self.config.data.train
        trainset = DD100lf(data.music_root, data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move, music_dance_rate=1)
        
        self.training_data = torch.utils.data.DataLoader(
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

        
        
    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model2.parameters(),
                                             ),
                                             **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

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

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
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

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
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

 


