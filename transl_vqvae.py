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
from datasets.dd100lf_all import DD100lfAll as DD100lf
# from models.vqvae import VQVAE

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


class MoQ():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        model = self.model.train()
        config = self.config
        data = self.config.data
        criterion = nn.MSELoss()
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0
        
        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight is not '':
            print('Use pretrained model!')
            print(config.init_weight)  
            checkpoint = torch.load(config.init_weight)
            model.load_state_dict(checkpoint['model'], strict=False)
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        #if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')


        np.random.seed(config.seed)

        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):
                music_seq, pose_seql, pose_seqf  = batch['music'], batch['pos3dl'], batch['pos3df'] 
                music_seq = music_seq.to(self.device)
                pose_seql, pose_seqf = pose_seql.to(self.device), pose_seqf.to(self.device)

                tranl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]) * 20.0
                # print(pose_seq.size())
                optimizer.zero_grad()
                # if not config.hybrid:
                output, loss, metrics = model(tranl)
                
                loss.backward()

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.item(),
                    # 'velocity_loss_if_have': metrics[0]['velocity_loss'].item() + metrics[1]['velocity_loss'].item(),
                    # 'acc_loss_if_have': metrics[0]['acceleration_loss'].item() + metrics[1]['acceleration_loss'].item()
                }
                #if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': model.state_dict(),
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
                    model.eval()
                    results = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants = {}
                    dance_names = []
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                        pose_seq_eval = batch_eval['pos3d'].to(self.device)
                        fname = batch_eval['fname'][0]
                        # print(fname)
                        dance_names.append(fname)

                        src_pos_eval = pose_seq_eval[:, :] #
                        global_shift = src_pos_eval[:, :, :3].clone()

                        if config.global_vel:
                            src_pos_eval[:, :-1, :3] = src_pos_eval[:, 1:, :3] - src_pos_eval[:, :-1, :3]
                            src_pos_eval[:, -1, :3] = src_pos_eval[:, -2, :3]
                        else:
                            src_pos_eval[:, :, :3] = 0

                        pose_seq_out, loss, _ = model(src_pos_eval)  # first 20 secs


                        if config.global_vel:
                            global_vel = pose_seq_out[:, :, :3].clone()
                            pose_seq_out[:, 0, :3] = 0
                            for iii in range(1, pose_seq_out.size(1)):
                                pose_seq_out[:, iii, :3] = pose_seq_out[:, iii-1, :3] + global_vel[:, iii-1, :]
                            # print('Use vel!')
                            # print(pose_seq_out[:, :, :3])
                        # elif config.rotmat:
                        #     pose_seq_out = torch.cat([global_shift, pose_seq_out], dim=2)
                        else:
                            pose_seq_out[:, :, :3] = global_shift
                        results.append(pose_seq_out)

                        if config.structure.use_bottleneck:
                            quants_pred = model.module.encode(src_pos_eval)
                            # print(quants_pred, flush=True)
                            if isinstance(quants_pred, tuple):
                                quants[fname] = tuple(quants_pred[ii][0].cpu().data.numpy()[0] for ii in range(len(quants_pred)))
                            else:
                                quants[fname] = model.module.encode(src_pos_eval)[0][0]
                        else:
                            quants = None
                model.train()
            self.schedular.step()  


    def eval(self):
        with torch.no_grad():
            config = self.config
            model = self.model.eval()
            epoch_tested = config.testing.ckpt_epoch
            test_loader = self.test_loader
            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model.eval()

            results = []
            leaders = []
            random_id = 0  # np.random.randint(0, 1e4)
            quants = {}
            dance_names = []
            for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                # pose_seq_eval = batch_eval['pos3d'].to(self.device)
                fname = batch_eval['fname'][0]

                pose_seql_eval, pose_seqf_eval  =  batch_eval['pos3dl'].to(self.device), batch_eval['pos3df'].to(self.device) 
                # print(fname)
                dance_names.append(fname)

                transl_eval = (pose_seqf_eval[:, :, :3] - pose_seql_eval[:, :, :3]).clone() * 20.0
                pose_seql_eval[:, :, :3] = pose_seql_eval[:, :, :3] - pose_seql_eval[:, :1, :3]

                transl_out, loss, _ = model(transl_eval)  # first 20 secs

                pose_seqf_eval[:, :, :3] = pose_seql_eval[:, :, :3] + transl_out / 20.0
                pose_seqf_eval = pose_seqf_eval.cpu().data.numpy()

                    # the root of left hand
                left_twist = pose_seqf_eval[:, :, 60:63]
                # 25,40
                pose_seqf_eval[:, :, 75:120] = pose_seqf_eval[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_seqf_eval[:, :, 63:66]
                # 40,55
                pose_seqf_eval[:, :, 120:165] = pose_seqf_eval[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_seqf_eval[:, :, :3]  # the root
                pose_seqf_eval = pose_seqf_eval + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_seqf_eval[:, :, :3] = root
                
                #####
                pose_seql_eval = pose_seql_eval.cpu().data.numpy()
                left_twist = pose_seql_eval[:, :, 60:63]
                # 25,40
                pose_seql_eval[:, :, 75:120] = pose_seql_eval[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_seql_eval[:, :, 63:66]
                # 40,55
                pose_seql_eval[:, :, 120:165] = pose_seql_eval[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_seql_eval[:, :, :3]  # the root
                pose_seql_eval = pose_seql_eval + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_seql_eval[:, :, :3] = root
                results.append(pose_seqf_eval)
                leaders.append(pose_seql_eval)

                # if config.structure.use_bottleneck:
                #     quants_pred = model.module.encode(src_pos_eval)
                #     # print(quants_pred, flush=True)
                #     if isinstance(quants_pred, tuple):
                #         quants[fname] = tuple(quants_pred[ii][0].cpu().data.numpy()[0] for ii in range(len(quants_pred)))
                #     else:
                #         quants[fname] = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]
                # else:
                #     quants = None

            visualize(results, leaders, config.testing, self.evaldir, dance_names, epoch_tested, None)
    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        leaders = []
        dance_names = []
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # if hasattr(config, 'demo') and config.demo:
                #     music_seq = batch_eval['music'].to(self.device)
                #     quants = ([torch.ones(1, 1,).to(self.device).long() * 423], [torch.ones(1, 1,).to(self.device).long() * 12])
                # else:
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
        if hasattr(config.structure, 'name'): 
            print(f'using {config.structure.name}')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            # model_class2 = getattr(models, config.structure_generate.name)
            # model2 = model_class2(config.structure_generate)
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model.float())
        # model2 = nn.DataParallel(model2.float())
        # self.model2 = model2.cuda()
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

        
        
    def _build_optimizer(self):
        #model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model.module.parameters(),
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



 


