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
from datasets.dd100_motion import DD100M
# from models.vqvae import VQVAE

from utils.log import Logger
from utils.visualize import visualize
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
warnings.filterwarnings('ignore')

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
            np.random.seed(config.seed + epoch_i)
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):
                # LR Scheduler missing
                # pose_seq = map(lambda x: x.to(self.device), batch)
                trans = None
                # if data.train.dtype == 'both':
                pose_seq, rot_seq = batch['pos3d'].float(), batch['rotmat'].float()
                pose_seq = pose_seq.to(self.device)
                rot_seq = rot_seq.to(self.device)

                pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                global_vel = pose_seq[:, :, :3].clone().detach()

                # print(pose_seq, flush=True)
                # print(rot_seq, flush=True)
                pose_seq[:, :, :3] = 0

                optimizer.zero_grad()
                # if not config.hybrid:
                output, output_rot, output_shift, loss, metrics = model(pose_seq, rot_seq, global_vel)
                
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
                        rot_seq_eval = batch_eval['rotmat'].to(self.device)
                        fname = batch_eval['fname'][0]
                        # print(fname)
                        dance_names.append(fname)

                        src_pos_eval = pose_seq_eval[:, :] #
                        src_pos_eval[:, :, :3] = 0

                        pose_seq_out, rot_seq_out, shift_out, loss, _ = model(src_pos_eval, rot_seq_eval, src_pos_eval[:, :, :3])  


                        global_vel = shift_out.clone()
                        pose_seq_out[:, 0, :3] = 0
                        for iii in range(1, pose_seq_out.size(1)):
                            pose_seq_out[:, iii, :3] = pose_seq_out[:, iii-1, :3] + global_vel[:, iii-1, :]

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
            random_id = 0  # np.random.randint(0, 1e4)
            quants = {}
            dance_names = []
            for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                pose_seq_eval = batch_eval['pos3d'].to(self.device)
                rot_seq_eval = batch_eval['rotmat'].to(self.device)
                fname = batch_eval['fname'][0]
                # print(fname)
                dance_names.append(fname)

                src_pos_eval = pose_seq_eval[:, :] #
                global_trans_gt = src_pos_eval[:, :, :3].clone().detach()
                src_pos_eval[:, :-1, :3] = src_pos_eval[:, 1:, :3] - src_pos_eval[:, :-1, :3]
                src_pos_eval[:, -1, :3] = src_pos_eval[:, -2, :3]
                # src_pos_eval[:, :, :3] = 0

                pose_seq_out, rot_seq_out, shift_out, loss, _ = model(src_pos_eval, rot_seq_eval, src_pos_eval[:, :, :3])  


                global_vel = shift_out.clone()
                global_vel = src_pos_eval[:, :, :3].clone()
                pose_seq_out[:, 0, :3] = 0

                for iii in range(1, pose_seq_out.size(1)):
                    pose_seq_out[:, iii, :3] = pose_seq_out[:, iii-1, :3] + global_vel[:, iii-1, :]

                pose_seq_out = pose_seq_out.cpu().data.numpy()

                    # the root of left hand
                left_twist = pose_seq_out[:, :, 60:63]
                # 25,40
                pose_seq_out[:, :, 75:120] = pose_seq_out[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

                # the root of right hand
                right_twist = pose_seq_out[:, :, 63:66]
                # 40,55
                pose_seq_out[:, :, 120:165] = pose_seq_out[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

                root = pose_seq_out[:, :, :3]  # the root
                pose_seq_out = pose_seq_out + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
                pose_seq_out[:, :, :3] = root
                
                results.append(pose_seq_out)

                if config.structure.use_bottleneck:
                    quants_pred = model.module.encode(src_pos_eval)
                    # print(quants_pred, flush=True)
                    if isinstance(quants_pred, tuple):
                        quants[fname] = tuple(quants_pred[ii][0].cpu().data.numpy()[0] for ii in range(len(quants_pred)))
                    else:
                        quants[fname] = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]
                else:
                    quants = None

            
            visualize(results, config.testing, self.evaldir, dance_names, epoch_tested, quants)

            # metrics = quantized_metrics()
            # print(metrics)

    def visgt(self,):
        config = self.config
        print("Visualizing ground truth")

        results = []
        dance_names = []
        random_id = 0  # np.random.randint(0, 1e4)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            pose_seq_out = batch_eval['pos3d'].to(self.device).cpu().data.numpy()
            fname = batch_eval['fname'][0]
            # print(fname)
            dance_names.append(fname)
            pose_seq_out = pose_seq_out
            root = pose_seq_out[:, :, :3]  # the root
            pose_seq_out = pose_seq_out + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root

            left_twist = pose_seq_out[:, :, 60:63]
            # 25,40
            pose_seq_out[:, :, 75:120] = pose_seq_out[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))

            # the root of right hand
            right_twist = pose_seq_out[:, :, 63:66]
            # 40,55
            pose_seq_out[:, :, 120:165] = pose_seq_out[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))

            root = pose_seq_out[:, :, :3]  # the root
            pose_seq_out = pose_seq_out + np.tile(root, (1, 1, 55))  # Calculate relative offset with respect to root
            pose_seq_out[:, :, :3] = root

            results.append(pose_seq_out)

        visualize(results, config.testing, self.gtdir, dance_names, 0, None)

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
            pose_seq_eval = batch_eval.to(self.device).cpu().data.numpy()

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        print(all_quants)
                    # exit()
        # visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)
        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])

        #图片的显示及存储
        #plt.show()   #这个是图片显示
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
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                if config.global_vel:
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)

        elif hasattr(config, 'analysis_sequence') and config.analysis_sequence is not None:
            # print(config.analysis_array)
            names = ['-'.join([str(jj) for jj in ii]) + '-rate' + str(config.sample_code_rate) for ii in config.analysis_sequence]
            print(names)
            for ii in config.analysis_sequence:
                print(ii)

                zs =  [torch.tensor(np.array(ii).repeat(self.config.sample_code_rate), device='cuda')[None].long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants['-'.join([str(jj) for jj in ii]) + '-rate' + str(config.sample_code_rate) ] = (zs[0].cpu().data.numpy()[0], zs[0].cpu().data.numpy()[0])

                if False:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)

        elif hasattr(config, 'analysis_pair') and config.analysis_pair is not None:
            print(config.analysis_pair)
            names = ['-'.join([str(jj) for jj in ii])  for ii in config.analysis_pair]
            print(names)
            for ii in config.analysis_pair:
                print(ii)
                zs =  ([torch.tensor(np.array(ii[:1]).repeat(self.config.sample_code_rate), device='cuda')[None].long()], [torch.tensor(np.array(ii[1:2]).repeat(self.config.sample_code_rate), device='cuda')[None].long()])
                print(zs[0][0].size())
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants['-'.join([str(jj) for jj in ii]) ] = (zs[0][0].cpu().data.numpy()[0], zs[1][0].cpu().data.numpy()[0])

                if False:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length), device='cuda')]
                pose_sample = model.module.decode(zs)
                if config.rotmat:
                    pose_sample = torch.cat([torch.zeros(pose_sample.size(0), pose_sample.size(1), 3).cuda(), pose_sample], dim=2)
                quants[str(ii)] = zs[0].cpu().data.numpy()[0]
                quants['rand_seq_' + str(ii)] = (zs[0].cpu().data.numpy()[0], zs[0].cpu().data.numpy()[0])

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
        else:
            raise NotImplementedError("Wrong Model Selection")
        
        model = nn.DataParallel(model)
        self.model = model.cuda()

    def _build_train_loader(self):

        data = self.config.data.train
        trainset = DD100M(data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move)
        
        self.training_data = torch.utils.data.DataLoader(
            trainset,
            num_workers=8,
            batch_size=data.batch_size,
            pin_memory=True,
            shuffle=True, 
            drop_last=True,
        )

    def _build_test_loader(self):
        config = self.config
        data = self.config.data.test
        
        testset = DD100M(data.data_root, split=data.split, interval=data.interval, dtype=data.dtype, move=data.move)
        #pdb.set_trace()

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False
            # collate_fn=paired_collate_fn,
        )
        # self.dance_names = dance_names
        #pdb.set_trace()
        #self.training_data = self.test_loader

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
        if not os.path.exists(os.path.join(self.gtdir, 'videos')):
            os.mkdir(os.path.join(self.gtdir, 'videos'))

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

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)










# def train_m2d(cfg):
#     """ Main function """
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--train_dir', type=str, default='data/train_1min',
#                         help='the directory of dance data')
#     parser.add_argument('--test_dir', type=str, default='data/test_1min',
#                         help='the directory of music feature data')
#     parser.add_argument('--data_type', type=str, default='2D',
#                         help='the type of training data')
#     parser.add_argument('--output_dir', metavar='PATH',
#                         default='checkpoints/layers2_win100_schedule100_condition10_detach')

#     parser.add_argument('--epoch', type=int, default=300000)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--save_per_epochs', type=int, metavar='N', default=50)
#     parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
#                         help='log model loss per x updates (mini-batches).')
#     parser.add_argument('--seed', type=int, default=1234,
#                         help='random seed for data shuffling, dropout, etc.')
#     parser.add_argument('--tensorboard', action='store_false')

#     parser.add_argument('--d_frame_vec', type=int, default=438)
#     parser.add_argument('--frame_emb_size', type=int, default=800)
#     parser.add_argument('--d_pose_vec', type=int, default=24*3)
#     parser.add_argument('--pose_emb_size', type=int, default=800)

#     parser.add_argument('--d_inner', type=int, default=1024)
#     parser.add_argument('--d_k', type=int, default=80)
#     parser.add_argument('--d_v', type=int, default=80)
#     parser.add_argument('--n_head', type=int, default=10)
#     parser.add_argument('--n_layers', type=int, default=2)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--dropout', type=float, default=0.1)

#     parser.add_argument('--seq_len', type=int, default=240)
#     parser.add_argument('--max_seq_len', type=int, default=4500)
#     parser.add_argument('--condition_step', type=int, default=10)
#     parser.add_argument('--sliding_windown_size', type=int, default=100)
#     parser.add_argument('--lambda_v', type=float, default=0.01)

#     parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL', const=True,
#                         default=torch.cuda.is_available(),
#                         help='whether to use GPU acceleration.')
#     parser.add_argument('--aist', action='store_true', help='train on AIST++')
#     parser.add_argument('--rotmat', action='store_true', help='train rotation matrix')

#     args = parser.parse_args()
#     args.d_model = args.frame_emb_size




#     args_data = args.data
#     args_structure = args.structure



 


