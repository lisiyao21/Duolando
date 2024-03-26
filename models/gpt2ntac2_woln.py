"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from torch.autograd import Function

class ClipGradient(Function):
    @staticmethod
    def forward(ctx, input, clip_value):
        ctx.save_for_backward(input)
        ctx.clip_value = clip_value
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_output[input > ctx.clip_value] = 0
        grad_output[input < -ctx.clip_value] = 0
        return grad_output, None

def clip_gradient(input, clip_value):
    return ClipGradient.apply(input, clip_value)

def get_subsequent_mask(seq_len, sliding_windown_size):
    """ For masking out the subsequent info. """
    # batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len)).float()
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    # mask = 1 - mask
    # print(mask)
    return mask #.bool()
def get_sudo_triu_mask(seq_len, look_forward):
    """ For masking out the subsequent info. """
    # batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len + look_forward)).float()
    mask = 1 - torch.triu(mask, diagonal=seq_len)
    # mask = 1 - mask
    # print(mask)
    return mask #.bool()
class MaskAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.register_buffer("mask", get_subsequent_mask( (config.block_size + config.look_forward) * config.downsample_rate, config.look_forward * config.downsample_rate)
                                     [None, None])
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T 
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print(self.mask.size())
        # print(att.size())
        # print(t)
        att = att.masked_fill(self.mask[:,:,:t,:t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class MusicBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MaskAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MusicTrans(nn.Module):
    """ mix up the neiboring music features before auto-regressively inference"""
    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        # self.requires_head = config.requires_head
        # self.requires_tail = config.requires_tail

        # if config.requires_head:
        # dance_
        self.config = config
        self.pos_emb = nn.Parameter(torch.zeros(1, (config.block_size+config.look_forward)*config.downsample_rate, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[MusicBlock(config) for _ in range(config.n_layer)])

        self.downsample = nn.Linear(config.n_embd * config.downsample_rate, config.n_embd)
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, music):
        b, t, c = music.size()
        assert t <= (self.config.block_size + self.config.look_forward) * self.config.downsample_rate, "Cannot forward, model block size is exhausted."
      
        # forward the GPT model
        # if self.requires_head:
        # token_embeddings_up = self.tok_emb_up(idx_up) # each index maps to a (learnable) vector
        # token_embeddings_down = self.tok_emb_down(idx_down) # each index maps to a (learnable) vector
        # print('Line 143 music size', music.size())
        token_embeddings = self.cond_emb(music)

        x = self.drop(token_embeddings)
        x = self.blocks(x)

        # print('Line 149 x size', x.size())
        # print(x.size())

        # print('L153 xd size', xd.size())
        b, t, c = x.size()
        x = self.downsample(x.view(b, t // self.config.downsample_rate, c * self.config.downsample_rate))
        xd = x[:,:-self.config.look_forward, :].contiguous()
        # print('L156 x size', x.size())
        # x = self.ln_f(x)
        # print(xd.size())

        return xd
    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False

def remove_dataparallel_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix from the keys
        new_state_dict[new_key] = value
    return new_state_dict

class GPT2ntAC2(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.alpha = config.alpha
        self.music_trans = MusicTrans(config.music_trans)
        
        self.leader_up_trans = MusicTrans(config.leader_up_trans)
        self.leader_down_trans = MusicTrans(config.leader_down_trans)
        self.leader_lhand_trans = MusicTrans(config.leader_lhand_trans)
        self.leader_rhand_trans = MusicTrans(config.leader_rhand_trans)

        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd)
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        self.tok_emb_lhand = nn.Embedding(config.vocab_size_lhand, config.n_embd)
        self.tok_emb_rhand = nn.Embedding(config.vocab_size_rhand, config.n_embd)
        self.tok_emb_transl = nn.Embedding(config.vocab_size_transl, config.n_embd)

        self.gpt_base = GPTBase(config.base)
        self.gpt_head = GPTHead(config.head)
        # self.gpt_scale = GPTScale(config.scale)

        self.alpha = config.alpha
        
        # self.down_half_gpt = CrossCondGPTHead(config.down_half_gpt)
        # if hasattr(config, 'critic_net'):
        #     self.critic_net = CrossCondGPTHead(config.critic_net)
        self.block_size = config.block_size
        self.music_sample_size = config.music_trans.downsample_rate
        self.look_forward = config.music_trans.look_forward

        if hasattr(config, 'init_weight_q'):
            print('Using pretrained Scaling Net')
            model = torch.load(config.init_weight_q)
            # self.gpt_scale.load_state_dict(remove_dataparallel_prefix(model['model']), strict=False)
    #     # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    # def get_block_size(self):
    #     return self.block_size

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    def get_block_size(self):
        return self.block_size
    # def sample(self, xs, cond):
    #     x_up, x_down = xs
    #     return (self.up_half_gpt.sample(x_up, cond), self.down_half_gpt.sample(x_down, cond))
    def sample(self, xs, cond, shift=None):
        
        block_size = self.get_block_size() - 1
        music_sample_rate = self.music_sample_size
        look_forward = self.look_forward

        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down, x_lhand, x_rhand, x_transl = xs

        # 59 59 59 59 59 30 30 30 30 --> 30
        cond_music, cond_up, cond_down, cond_lhand, cond_rhand = cond
        # print(len(cond_up))
        # print(cond_music.size(), cond_up[0].size(), cond_down[0].size(), cond_lhand[0].size(), cond_rhand[0].size(), flush=True)

        for k in range(cond_music.size(1) // music_sample_rate - look_forward):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            x_cond_lhand = x_lhand if x_lhand.size(1) <= block_size else x_lhand[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_rhand = x_rhand if x_rhand.size(1) <= block_size else x_rhand[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            x_cond_transl = x_transl if x_transl.size(1) <= block_size else x_transl[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            

            cond_input_music = cond_music[:, :(k + 1 + look_forward) * music_sample_rate] if k < block_size  else cond_music[:, (k + 1) * music_sample_rate - (block_shift+(k-block_size-1)%(block_size-block_shift+1)) * music_sample_rate : (k + 1 + look_forward) * music_sample_rate]
            cond_input_up = cond_up[:, :(k+1+look_forward)] if k < block_size  else cond_up[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_down = cond_down[:, :(k+1+look_forward)] if k < block_size  else cond_down[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_lhand = cond_lhand[:, :(k+1+look_forward)] if k < block_size  else cond_lhand[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_rhand = cond_rhand[:, :(k+1+look_forward)] if k < block_size  else cond_rhand[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            
            
            cond_input = (cond_input_music, cond_input_up, cond_input_down, cond_input_lhand, cond_input_rhand)

            logits, _ = self.forward((x_cond_up, x_cond_down, x_cond_lhand, x_cond_rhand, x_cond_transl), cond_input)

            logit_up, logit_down, logit_lhand, logit_rhand, logit_transl = logits

            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]
            logit_lhand = logit_lhand[:, -1, :]
            logit_rhand = logit_rhand[:, -1, :]
            logit_transl = logit_transl[:, -1, :]
            
            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)
            probs_lhand = F.softmax(logit_lhand, dim=-1)
            probs_rhand = F.softmax(logit_rhand, dim=-1)
            probs_transl = F.softmax(logit_transl, dim=-1)

            _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            _, ix_down = torch.topk(probs_down, k=1, dim=-1)
            _, ix_lhand = torch.topk(probs_lhand, k=1, dim=-1)
            _, ix_rhand = torch.topk(probs_rhand, k=1, dim=-1)
            _, ix_transl = torch.topk(probs_transl, k=1, dim=-1)

            # append to the sequence and continue
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)
            x_lhand = torch.cat((x_lhand, ix_lhand), dim=1)
            x_rhand = torch.cat((x_rhand, ix_rhand), dim=1)
            x_transl = torch.cat((x_transl, ix_transl), dim=1)

        return ([x_up], [x_down], [x_lhand], [x_rhand]), [x_transl]

    def samplek(self, xs, cond, topk, shift=None):
        
        block_size = self.get_block_size() - 1
        music_sample_rate = self.music_sample_size
        look_forward = self.look_forward

        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down, x_lhand, x_rhand, x_transl = xs

        # 59 59 59 59 59 30 30 30 30 --> 30
        cond_music, cond_up, cond_down, cond_lhand, cond_rhand = cond
        # print(len(cond_up))
        # print(cond_music.size(), cond_up[0].size(), cond_down[0].size(), cond_lhand[0].size(), cond_rhand[0].size(), flush=True)

        for k in range(cond_music.size(1) // music_sample_rate - look_forward):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            x_cond_lhand = x_lhand if x_lhand.size(1) <= block_size else x_lhand[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_rhand = x_rhand if x_rhand.size(1) <= block_size else x_rhand[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            x_cond_transl = x_transl if x_transl.size(1) <= block_size else x_transl[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            

            cond_input_music = cond_music[:, :(k + 1 + look_forward) * music_sample_rate] if k < block_size  else cond_music[:, (k + 1) * music_sample_rate - (block_shift+(k-block_size-1)%(block_size-block_shift+1)) * music_sample_rate : (k + 1 + look_forward) * music_sample_rate]
            cond_input_up = cond_up[:, :(k+1+look_forward)] if k < block_size  else cond_up[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_down = cond_down[:, :(k+1+look_forward)] if k < block_size  else cond_down[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_lhand = cond_lhand[:, :(k+1+look_forward)] if k < block_size  else cond_lhand[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            cond_input_rhand = cond_rhand[:, :(k+1+look_forward)] if k < block_size  else cond_rhand[:, (k + 1) - (block_shift+(k-block_size-1)%(block_size-block_shift+1)):(k + 1 + look_forward)]
            
            
            cond_input = (cond_input_music, cond_input_up, cond_input_down, cond_input_lhand, cond_input_rhand)

            logits, _ = self.forward((x_cond_up, x_cond_down, x_cond_lhand, x_cond_rhand, x_cond_transl), cond_input)

            logit_up, logit_down, logit_lhand, logit_rhand, logit_transl = logits

            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]
            logit_lhand = logit_lhand[:, -1, :]
            logit_rhand = logit_rhand[:, -1, :]
            logit_transl = logit_transl[:, -1, :]
            
            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)
            probs_lhand = F.softmax(logit_lhand, dim=-1)
            probs_rhand = F.softmax(logit_rhand, dim=-1)
            probs_transl = F.softmax(logit_transl, dim=-1)

            _, ix_up = torch.topk(probs_up, k=topk, dim=-1)
            _, ix_down = torch.topk(probs_down, k=topk, dim=-1)
            _, ix_lhand = torch.topk(probs_lhand, k=topk, dim=-1)
            _, ix_rhand = torch.topk(probs_rhand, k=topk, dim=-1)
            _, ix_transl = torch.topk(probs_transl, k=topk, dim=-1)


            rows = torch.arange(ix_up.shape[0])

            random_indices_up = torch.randint(0, topk, (ix_up.size(0),)).cuda()
            random_indices_down  = torch.randint(0, topk, (ix_down.size(0), )).cuda()
            random_indices_lhand = torch.randint(0, topk, (ix_lhand.size(0), )).cuda()
            random_indices_rhand  = torch.randint(0, topk, (ix_rhand.size(0),)).cuda()
            random_indices_transl = torch.randint(0, topk, (ix_transl.size(0),)).cuda()
            # use indices to select one element from each C channel
            # print(probs_up.size(),÷ ix_up.size(), rows.size(), cols.size(), random_indices_down.size())
            ix_up =  ix_up[rows, random_indices_up][:, None]
            ix_down = ix_down[rows, random_indices_down][:,  None]
            ix_lhand = ix_lhand[rows,random_indices_lhand][:, None]
            ix_rhand = ix_rhand[rows, random_indices_rhand][:, None]
            ix_transl = ix_transl[rows,  random_indices_transl][:, None]

            # append to the sequence and continue
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)
            x_lhand = torch.cat((x_lhand, ix_lhand), dim=1)
            x_rhand = torch.cat((x_rhand, ix_rhand), dim=1)
            x_transl = torch.cat((x_transl, ix_transl), dim=1)

        return ([x_up], [x_down], [x_lhand], [x_rhand]), [x_transl]

    def forward(self, idxs, cond, rewards=None, targets=None):
        """
            supervise/behavior clone: rewards is None and targets are not None
            RL: both rewards and targets are not None 
        """
        stage = 'supervise'
        if rewards is not None and targets is not None:
            stage = 'reinforce'
        
        idx_up, idx_down, idx_lhand, idx_rhand, idx_transl = idxs
        
        targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = None, None, None, None, None
        if targets is not None:
            targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
        
        cond_music, cond_up, cond_down, cond_lhand, cond_rhand = cond
        
        # print('L238', cond.size())
        cond_music = self.music_trans(cond_music)
        cond_up = self.leader_up_trans(self.tok_emb_up(cond_up))
        cond_down = self.leader_down_trans(self.tok_emb_down(cond_down))
        cond_lhand = self.leader_lhand_trans(self.tok_emb_lhand(cond_lhand))
        cond_rhand = self.leader_rhand_trans(self.tok_emb_rhand(cond_rhand))

        # print(cond_up.size(), cond_down.size(), cond_lhand.size(), cond_rhand.size(), flush=True)
        
        # print(cond.size())

        feat = self.gpt_base(self.tok_emb_up(idx_up), self.tok_emb_down(idx_down), self.tok_emb_lhand(idx_lhand), self.tok_emb_lhand(idx_rhand), self.tok_emb_transl(idx_transl), (cond_music, cond_up, cond_down, cond_lhand, cond_rhand))
        
        
        # logits_down, loss_down = self.down_half_gpt(feat, targets_down)
        
        if stage == 'reinforce':
            logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, _, _, _, _, _, = self.gpt_head(feat, None)
        
            
            reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = rewards
            reward_up[:, :-1] += reward_up[:, 1:] * 0.2
            reward_down[:, :-1] += reward_down[:, 1:] * 0.2
            reward_lhand[:, :-1] += reward_lhand[:, 1:] * 0.2
            reward_rhand[:, :-1] += reward_rhand[:, 1:] * 0.2
            reward_transl[:, :-1] += reward_transl[:, 1:] * 0.2

            y_up = torch.sigmoid(self.alpha * reward_up).clone().detach()
            y_down = torch.sigmoid(self.alpha * reward_down).clone().detach()
            y_lhand = torch.sigmoid(self.alpha * reward_lhand).clone().detach()
            y_rhand = torch.sigmoid(self.alpha * reward_rhand).clone().detach()
            y_transl = torch.sigmoid(self.alpha * reward_transl).clone().detach()

            probs_pi_up = torch.softmax(logits_up, dim=-1)
            probs_pi_down = torch.softmax(logits_down, dim=-1)
            probs_pi_lhand = torch.softmax(logits_lhand, dim=-1)
            probs_pi_rhand = torch.softmax(logits_rhand, dim=-1)
            probs_pi_transl = torch.softmax(logits_transl, dim=-1)
            
            probs_up_q = qindex(probs_pi_up, targets_up)
            probs_down_q = qindex(probs_pi_down, targets_down)
            probs_lhand_q = qindex(probs_pi_lhand, targets_lhand)
            probs_rhand_q = qindex(probs_pi_rhand, targets_rhand)
            probs_transl_q = qindex(probs_pi_transl, targets_transl)

            loss_up = - torch.log(1.001 - torch.abs(probs_up_q - y_up)).mean()
            loss_down = - torch.log(1.001 - torch.abs(probs_down_q - y_down)).mean()
            loss_lhand = - torch.log(1.001 - torch.abs(probs_lhand_q - y_lhand)).mean()
            loss_rhand = - torch.log(1.001 - torch.abs(probs_rhand_q - y_rhand)).mean()
            loss_transl = - torch.log(1.001 - torch.abs(probs_transl_q - y_transl)).mean()
        else:
            logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = self.gpt_head(feat, targets)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl
        else:
            loss = None
        # print(logits_up, logits_down, loss_lhand, loss_rhand, flush=True)
        return (logits_up, logits_down, logits_lhand, logits_rhand, logits_transl), loss

    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False


class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.block_size = config.block_size
        self.look_forward = config.look_forward
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # self.register_buffer("mask_cond2cond", get_subsequent_mask( config.block_size + config.look_forward, config.look_forward))
        # self.register_buffer("mask_cond2x", get_sudo_triu_mask(config.block_size, config.look_forward))
        # self.register_buffer("mask_x2x", torch.tril(torch.ones(config.block_size, config.block_size)))
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        t = T // 10
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        mask = torch.zeros(T, T)
        l = self.look_forward

        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, 10, 10) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # # input embedding stem
        # self.requires_head = config.requires_head
        # self.requires_tail = config.requires_tail

        # if config.requires_head:
        # dance_
        # self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd  )
        # self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        # self.tok_emb_lhand = nn.Embedding(config.vocab_size_up, config.n_embd  )
        # self.tok_emb_rhand = nn.Embedding(config.vocab_size_down, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*10, config.n_embd))

        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.cond_up_emb = nn.Linear(config.n_embd, config.n_embd)
        self.cond_down_emb = nn.Linear(config.n_embd, config.n_embd)
        self.cond_lhand_emb = nn.Linear(config.n_embd, config.n_embd)
        self.cond_rhand_emb = nn.Linear(config.n_embd, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.look_forward = config.look_forward


        self.apply(self._init_weights)


        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, idx_lhand, idx_rhand, idx_transl, cond):
        b, t, _ = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t, _ = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        l = self.look_forward
        T_plus_l = self.block_size + self.look_forward
        T = self.block_size

        # forward the GPT model
        # if self.requires_head:

        cond_music, cond_up, cond_down, cond_lhand, cond_rhand = cond

        token_embeddings_up = idx_up # each index maps to a (learnable) vector
        token_embeddings_down = idx_down # each index maps to a (learnable) vector
        token_embeddings_lhand = idx_lhand # each index maps to a (learnable) vector
        token_embeddings_rhand = idx_rhand # each index maps to a (learnable) vector
        token_embeddings_transl = idx_transl

        token_embeddings_cond_up = cond_up # each index maps to a (learnable) vector
        token_embeddings_cond_down = cond_down # each index maps to a (learnable) vector
        token_embeddings_cond_lhand = cond_lhand # each index maps to a (learnable) vector
        token_embeddings_cond_rhand = cond_rhand # each index maps to a (learnable) vector


        token_embeddings = torch.cat([cond_music, token_embeddings_cond_up, token_embeddings_cond_down, token_embeddings_cond_lhand, token_embeddings_cond_rhand, token_embeddings_up, token_embeddings_down, token_embeddings_lhand, token_embeddings_rhand, token_embeddings_transl], dim=1)

        position_embeddings = torch.cat([self.pos_emb[:, :t, :], \
                                         self.pos_emb[:, T:T+t, :], \
                                         self.pos_emb[:, T*2:T*2+t, :], \
                                         self.pos_emb[:, T*3:T*3+t, :], \
                                         self.pos_emb[:, T*4:T*4+t, :], \
                                         self.pos_emb[:, T*5:T*5+t, :], \
                                         self.pos_emb[:, T*6:T*6+t, :], \
                                         self.pos_emb[:, T*7:T*7+t, :], \
                                         self.pos_emb[:, T*8:T*8+t, :], \
                                         self.pos_emb[:, T*9:T*9+t, :]], dim=1) # each position maps to a (learnable) vector
        
        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)
        # x = self.ln_f(x)

        return x
    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False

class GPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

        self.look_forward = config.look_forward
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, targets=None):

        x = self.blocks(x)
        # x = self.ln_f(x)
        N, T, C = x.size()
        t = T // 10
        l = self.look_forward
        logits_up = self.head_up(x[:, t*5:t*6, :])
        logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
        logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
        logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
        logits_transl = self.head_transl(x[:, t*9:t*10, :])
        

        # if we are given some desired targets also calculate the loss
        loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None

        if targets is not None:
            targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
            # print(logits_up.size(), targets_up.size())

            # if len(targets_up.size()) == 2:
            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))
            loss_lhand = F.cross_entropy(logits_lhand.view(-1, logits_lhand.size(-1)), targets_lhand.view(-1))
            loss_rhand = F.cross_entropy(logits_rhand.view(-1, logits_rhand.size(-1)), targets_rhand.view(-1))
            loss_transl = F.cross_entropy(logits_transl.view(-1, logits_transl.size(-1)), targets_transl.view(-1))
            # else:
            #     # print(logits_down, logits_transl, logits_down.min(), logits_down.max(), logits_transl.min(), logits_transl.max(), flush=True)
            #     # print(logits_up.view(-1, logits_up.size(-1)).size(), targets_up.view(-1, targets_up.size(-1)).size(), targets_up.view(-1, targets_up.size(-1)), flush=True)
            #     loss_up = -torch.mean(torch.log(torch.softmax(logits_up.view(-1, logits_up.size(-1)), dim=-1) + 1e-9) * targets_up.view(-1, targets_up.size(-1)))
            #     loss_down = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-9) * targets_down.view(-1, targets_down.size(-1)))
            #     loss_lhand = -torch.mean(torch.log(torch.softmax(logits_lhand.view(-1, logits_lhand.size(-1)), dim=-1) + 1e-9) * targets_lhand.view(-1, targets_lhand.size(-1)))
            #     loss_rhand = -torch.mean(torch.log(torch.softmax(logits_rhand.view(-1, logits_rhand.size(-1)), dim=-1) + 1e-9) * targets_rhand.view(-1, targets_rhand.size(-1)))
            #     loss_transl = -torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-9) * targets_transl.view(-1, targets_transl.size(-1)))

        # if loss_up is np.nan
        return logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, loss_up, loss_down, loss_lhand, loss_rhand, loss_transl
    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False



def qindex(A, B):
    # Get the batch and sequence dimensions
    N, T = B.size()
    # Create a meshgrid for the batch and sequence dimensions
    n_range = torch.arange(N)[:, None].expand(-1, T)
    t_range = torch.arange(T)[None, :].expand(N, -1)
    # Index A with the meshgrid and B to get the desired output
    C = A[n_range, t_range, B]
    return C

