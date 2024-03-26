import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .encdec import Encoder, Decoder, assert_shape
from .bottleneck import NoBottleneck, Bottleneck
from .utils.logger import average_metrics
# from .utils.audio_utils import  audio_postprocess
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
# from .utils.audio_utils import  audio_postprocess
from smplx import SMPL


def rotation_6d_to_matrix(d6: t.Tensor) -> t.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = t.cross(b1, b2, dim=-1)
    return t.stack((b1, b2, b3), dim=-2)



def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def _loss_fn(x_target, x_pred, weighted=False):
    return t.mean(t.abs(x_pred - x_target)) 

# To train rotmat vqvae from pos3d
class VQVAEmix(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
    
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        commit = hps.commit
        # spectral = hps.spectral
        # multispectral = hps.multispectral
        multipliers = hps.hvqvae_multipliers 
        use_bottleneck = hps.use_bottleneck
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                        dilation_growth_rate=hps.dilation_growth_rate, \
                        dilation_cycle=hps.dilation_cycle, \
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = hps.sample_length

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(hps.sample_length // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(hps.pos_channel, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        decoder = lambda level: Decoder(hps.pos_channel, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        decoder_rot = lambda level: Decoder(hps.rot_channel, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        # decoder_root = lambda level: Decoder(hps.joint_channel, emb_width, level + 1,
                                        # downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoders_rot = nn.ModuleList()
        # self.decoders_root = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))
            self.decoders_rot.append(decoder_rot(level))
            # self.decoders_root.append(decoder_root(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        if self.reg is 0:
            print('No motion regularization!')
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x






    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder_pos, decoder, x_quantised = self.decoders[start_level], self.decoders_rot[start_level], xs_quantised[0:1]

        x_out_pos = decoder_pos(x_quantised, all_levels=False)
        x_out = decoder(x_quantised, all_levels=False)
        # x_vel_out = decoder_root(x_quantised, all_levels=False)
        x_out_pos = self.postprocess(x_out_pos)
        x_out = self.postprocess(x_out)
        # x_vel_out = self.postprocess(x_vel_out)
        
        # _, _, cc = x_vel_out.size()
        # x_out[:, :, :cc] = x_vel_out.clone()
        return x_out_pos, x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        # print(zs, flush=True)
        # print(len(zs), flush=True)
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        x_pos_outs = []
        # x_vel_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_pos, x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
            x_pos_outs.append(x_pos)
            # x_vel_outs.append(x_vel)
        return t.cat(x_pos_outs, dim=0), t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x[:, :, :self.hps.joint_channel] = 0
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, x_rot):
        # print(x, x_rot, x.shape,flush=True)
        # Here, the decoder is transformed from 

        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        # self.bottleneck.eval()
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        
        x_outs = []
        # x_outs_vel = []
        x_outs_pos = []

        for level in range(self.levels):
            decoder_pos = self.decoders[level]
            decoder = self.decoders_rot[level]
            # decoder_root = self.decoders_root[level].eval()
            x_pos_out = decoder_pos(xs_quantised[level:level+1], all_levels=False)
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            # x_vel_out = decoder_root(xs_quantised[level:level+1], all_levels=False)
            assert_shape(x_out, self.preprocess(x_rot).shape)
            
            x_outs.append(x_out)
            x_outs_pos.append(x_pos_out)
            # x_outs_vel.append(x_vel_out)

        # Loss
        # def _spectral_loss(x_target, x_out, self.hps):
        #     if hps.use_nonrelative_specloss:
        #         sl = spectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     else:
        #         sl = spectral_convergence(x_target, x_out, self.hps)
        #     sl = t.mean(sl)
        #     return sl

        # def _multispectral_loss(x_target, x_out, self.hps):
        #     sl = multispectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     sl = t.mean(sl)
        #     return sl

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        # spec_loss = t.zeros(()).to(x.device)
        # multispec_loss = t.zeros(()).to(x.device)
        # x_target = audio_postprocess(x.float(), self.hps)
        x_target = x_rot.float()
        x_target_pos = x.float()


        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            # x_out_vel = self.postprocess(x_outs_vel[level])
            x_out_pos = self.postprocess(x_outs_pos[level])
            # x_out = audio_postprocess(x_out, self.hps)

            # this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_recons_loss = _loss_fn(x_target_pos, x_out_pos) + _loss_fn(x_target, x_out)
            # this_spec_loss = _spectral_loss(x_target, x_out, hps)
            # this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            # metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            # metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            # spec_loss += this_spec_loss
            # multispec_loss += this_multispec_loss
            regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1])**2)

            velocity_loss += _loss_fn(x_out_pos[:, 1:] - x_out_pos[:, :-1], x_target_pos[:, 1:] - x_target_pos[:, :-1]) \
                + _loss_fn( x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            
            acceleration_loss += _loss_fn(x_out_pos[:, 2:] + x_out_pos[:, :-2] - 2 * x_out_pos[:, 1:-1], x_target_pos[:, 2:] + x_target_pos[:, :-2] - 2 * x_target_pos[:, 1:-1]) \
                + _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1]) 
        
            # if self.smpl_weight > 0:
            #     n, tt, c = x_target.size()
            #     if c == 6*9:
            #         x_target = x_target.view(n, tt, 9, 6)
            #         x_out = x_out.view(n, tt, 9, 6)
            #         rotmat_target = rotation_6d_to_matrix(x_target)
            #         rotmat_pred = rotation_6d_to_matrix(x_out)
            #     else:
            #         x_target = x_target.view(n, tt, 9, 3, 3)
            #         x_out = x_out.view(n, tt, 9, 3, 3)
            #         rotmat_target = x_target
            #         rotmat_pred = x_out
                
            #     aa_pred = matrix_to_axis_angle(rotmat_pred)
            #     aa_target = matrix_to_axis_angle(rotmat_target)

            #     pos3d_pred = self.smpl1.eval().forward(
            #         global_orient=aa_pred[:, 0:1].float(),
            #         body_pose=aa_pred[:, 1:].float(),
            #     ).joints[:, 0:24, :]
            #     pos3d_target = self.smpl2.eval().forward(
            #         global_orient=aa_target[:, 0:1].float(),
            #         body_pose=aa_target[:, 1:].float(),
            #     ).joints[:, 0:24, :]
            #     smpl_loss += _loss_fn(pos3d_pred, pos3d_target)
        # if not hasattr(self.)
        commit_loss = sum(commit_losses)
        # loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        loss = recons_loss + commit_loss * self.commit + self.vel * velocity_loss + self.acc * acceleration_loss
        # if self.smpl_weight > 0:
        #     loss += smpl_loss


        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn(x_target_pos, x_out_pos)
            
            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            # spectral_loss=spec_loss,
            # multispectral_loss=multispec_loss,
            # spectral_convergence=sc,
            # l2_loss=l2_loss,
            l1_loss=l1_loss,
            # linf_loss=linf_loss,
            commit_loss=l1_loss*0,
            regularization=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out_pos, x_out, loss, metrics
