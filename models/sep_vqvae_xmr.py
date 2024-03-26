import numpy as np
import torch
import torch.nn as nn

# from .encdec import Encoder, Decoder, assert_shape
# from .bottleneck import NoBottleneck, Bottleneck
# from .utils.logger import average_metrics
# from .utils.audio_utils import  audio_postprocess

from .vqvae_mix import VQVAEmix
from .vqvae_root_mix import VQVAERmix

SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3', 
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]

smpl_down = [0, 1, 2, 4,  5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
smpl_lhand = list(range(25, 40))
smpl_rhand = list(range(40, 55))
# def dont_update(params):
#     for param in params:
#         param.requires_grad = False

# def update(params):
#     for param in params:
#         param.requires_grad = True

# def calculate_strides(strides, downs):
#     return [stride ** down for stride, down in zip(strides, downs)]

# # def _loss_fn(loss_fn, x_target, x_pred, hps):
#     if loss_fn == 'l1':
#         return torch.mean(torch.abs(x_pred - x_target)) / hps.bandwidth['l1']
#     elif loss_fn == 'l2':
#         return torch.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
#     elif loss_fn == 'linf':
#         residual = ((x_pred - x_target) ** 2).reshape(x_targetorch.shape[0], -1)
#         values, _ = torch.topk(residual, hps.linf_k, dim=1)
#         return torch.mean(values) / hps.bandwidth['l2']
#     elif loss_fn == 'lmix':
#         loss = 0.0
#         if hps.lmix_l1:
#             loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
#         if hps.lmix_l2:
#             loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
#         if hps.lmix_linf:
#             loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
#         return loss
#     else:
#         assert False, f"Unknown loss_fn {loss_fn}"
# def _loss_fn(x_target, x_pred):
#     return torch.mean(torch.abs(x_pred - x_target)) 


class SepVQVAEXM(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # self.cut_dim = hps.up_half_dim
        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        self.chanel_num = hps.joint_channel
        self.rot_chanel_num = hps.rot_joint_channel
        self.vqvae_up = VQVAEmix(hps.up_half)
        self.vqvae_down = VQVAERmix(hps.down_half)
        self.vqvae_lhand = VQVAEmix(hps.lhand)
        self.vqvae_rhand = VQVAEmix(hps.rhand)
        self.vqvae_vel = VQVAEmix(hps.vel)
        # self.vqvae_shift = VQVAEmix(hps.shift)

        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        # self.chanel_num = 9 if self.use_rotmat else 3


    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        """
        zs are list with two elements: z for up and z for down
        """
        if isinstance(zs, tuple):
            zup = zs[0]
            zdown = zs[1]
            zlhand = zs[2]
            zrhand = zs[3]
            zvel = zs[4]
        else:
            zup = zs
            zdown = zs
            zlhand = zs
            zrhand = zs
            zvel = zs[4]

        xup, xup_rot = self.vqvae_up.decode(zup)
        xdown, xdown_rot, xshift = self.vqvae_down.decode(zdown)
        xlhand, xlhand_rot = self.vqvae_lhand.decode(zlhand)
        xrhand, xrhand_rot = self.vqvae_rhand.decode(zrhand)
        xvel = self.vqvae_vel.decode(zvel)


        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        _, _, clh = xlhand.size()
        _, _, crh = xrhand.size()
        _, _, cvel = xvel.size()

        b, t, cup_rot = xup_rot.size()
        _, _, cdown_rot = xdown_rot.size()
        _, _, clh_rot = xlhand_rot.size()
        _, _, crh_rot = xrhand_rot.size()


        x = torch.zeros(b, t, (cup+cdown+clh+crh)//self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num)
        x[:, :, smpl_lhand] = xlhand.view(b, t, clh//self.chanel_num, self.chanel_num)
        x[:, :, smpl_rhand] = xrhand.view(b, t, crh//self.chanel_num, self.chanel_num)
        
        x_rot = torch.zeros(b, t, (cup_rot+cdown_rot+clh_rot+crh_rot)//self.rot_chanel_num, self.rot_chanel_num).cuda()
        x_rot[:, :, smpl_up] = xup_rot.view(b, t, cup_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_down] = xdown_rot.view(b, t, cdown_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_lhand] = xlhand_rot.view(b, t, clh_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_rhand] = xrhand_rot.view(b, t, crh_rot//self.rot_chanel_num, self.rot_chanel_num)

        return x.view(b, t, -1), x_rot.view(b, t, -1), xvel


        # z_chunks = [torch.chunk(z, bs_chunks, dim=0) for z in zs]
        # x_outs = []
        # for i in range(bs_chunks):
        #     zs_i = [z_chunk[i] for z_chunk in z_chunks]
        #     x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
        #     x_outs.append(x_out)

        # return torch.cat(x_outs, dim=0)

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1), start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level, bs_chunks)
        zlhand = self.vqvae_lhand.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_lhand].view(b, t, -1), start_level, end_level, bs_chunks)
        zrhand = self.vqvae_rhand.encode(x.view(b, t, c//self.chanel_num, self.chanel_num)[:, :, smpl_rhand].view(b, t, -1), start_level, end_level, bs_chunks)
        return (zup, zdown, zlhand, zrhand)

    def sample(self, n_samples):
        # zs = [torch.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        xup = self.vqvae_up.sample(n_samples)
        xdown = self.vqvae_up.sample(n_samples)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        _, _, clh = xlhand.size()
        _, _, crh = xrhand.size()

        x = torch.zeros(b, t, (cup+cdown+clh+crh)//self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num)
        x[:, :, smpl_lhand] = xlhand.view(b, t, clh//self.chanel_num, self.chanel_num)
        x[:, :, smpl_rhand] = xrhand.view(b, t, crh//self.chanel_num, self.chanel_num)
        return x

    def forward(self, x, xrot, xshift):
        b, t, c = x.size()
        _, _, crot = xrot.size()
        
        x, xrot = x.view(b, t, c//self.chanel_num, self.chanel_num), xrot.view(b, t, crot//self.rot_chanel_num, self.rot_chanel_num)

        xup = x[:, :, smpl_up, :].view(b, t, -1)
        xdown = x[:, :, smpl_down, :].view(b, t, -1)
        xlhand = x[:, :, smpl_lhand, :].view(b, t, -1)
        xrhand = x[:, :, smpl_rhand, :].view(b, t, -1)

        xuprot = xrot[:, :, smpl_up, :].view(b, t, -1)
        xdownrot = xrot[:, :, smpl_down, :].view(b, t, -1)
        xlhandrot = xrot[:, :, smpl_lhand, :].view(b, t, -1)
        xrhandrot = xrot[:, :, smpl_rhand, :].view(b, t, -1)

        x_out_up, x_out_up_rot, loss_up, metrics_up = self.vqvae_up(xup, xuprot)
        x_out_lhand, x_out_lhand_rot, loss_lhand, metrics_lhand = self.vqvae_lhand(xlhand, xlhandrot)
        x_out_rhand, x_out_rhand_rot, loss_rhand, metrics_rhand = self.vqvae_rhand(xrhand, xrhandrot)
        x_out_down, x_out_down_rot, x_shift, loss_down, metrics_down  = self.vqvae_down(xdown, xdownrot, xshift)

        _, _, cup = x_out_up.size()
        _, _, cdown = x_out_down.size()
        _, _, clh = x_out_lhand.size()
        _, _, crh = x_out_rhand.size()

        _, _, cup_rot = x_out_up_rot.size()
        _, _, cdown_rot = x_out_down_rot.size()
        _, _, clh_rot = x_out_lhand_rot.size()
        _, _, crh_rot = x_out_rhand_rot.size()

        xout = torch.zeros(b, t, (cup+cdown+clh+crh)//self.chanel_num, self.chanel_num).cuda().float()
        xout[:, :, smpl_up] = x_out_up.view(b, t, cup//self.chanel_num, self.chanel_num)
        xout[:, :, smpl_down] = x_out_down.view(b, t, cdown//self.chanel_num, self.chanel_num)
        xout[:, :, smpl_lhand] = xlhand.view(b, t, clh//self.chanel_num, self.chanel_num)
        xout[:, :, smpl_rhand] = xrhand.view(b, t, crh//self.chanel_num, self.chanel_num)

        x_rot = torch.zeros(b, t, (cup_rot+cdown_rot+clh_rot+crh_rot)//self.rot_chanel_num, self.rot_chanel_num).cuda()
        x_rot[:, :, smpl_up] = x_out_up_rot.view(b, t, cup_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_down] = x_out_down_rot.view(b, t, cdown_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_lhand] = x_out_lhand_rot.view(b, t, clh_rot//self.rot_chanel_num, self.rot_chanel_num)
        x_rot[:, :, smpl_rhand] = x_out_rhand_rot.view(b, t, crh_rot//self.rot_chanel_num, self.rot_chanel_num)
        
        return xout.view(b, t, -1), x_rot.view(b, t, -1), x_shift, loss_up + loss_down + loss_lhand + loss_rhand, [metrics_up, metrics_down] 
