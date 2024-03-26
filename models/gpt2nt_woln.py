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
# from .gpt3p import condGPT3Part
# logger = logging.getLogger(__name__)



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


class GPT2nt(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
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
        
        # self.down_half_gpt = CrossCondGPTHead(config.down_half_gpt)
        # if hasattr(config, 'critic_net'):
        #     self.critic_net = CrossCondGPTHead(config.critic_net)
        self.block_size = config.block_size
        self.music_sample_size = config.music_trans.downsample_rate
        self.look_forward = config.music_trans.look_forward
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

    def samplek(self, xs, cond, topk=5, shift=None):
        
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

    def forward(self, idxs, cond, targets=None):
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
        logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = self.gpt_head(feat, targets)
        # logits_down, loss_down = self.down_half_gpt(feat, targets_down)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl
        else:
            loss = None
        # print(logits_up, logits_down, loss_lhand, loss_rhand, flush=True)
        return (logits_up, logits_down, logits_lhand, logits_rhand, logits_transl), loss
    
    def state(self, idxs, cond):
        self.gpt_base.eval()
        idx_up, idx_down, idx_lhand, idx_rhand, idx_transl = idxs

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
       
        return feat
    
    def actor(self, state, target=None):
        logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = self.gpt_head(state, target)
        # logits_down, loss_down = self.down_half_gpt(feat, targets_down)
        
        loss = None
        if target is not None:
            loss = loss_down + loss_transl
            loss_copy = loss.clone().detach()
            if loss_copy.mean().cpu().data.numpy() == np.nan or loss_copy.mean().cpu().data.numpy() == np.inf:
                np.save('outlier.npy', (logits_up, logits_down, logits_lhand, logits_rhand, logits_transl, loss_up, loss_down, loss_lhand, loss_rhand, loss_transl, target))


        probs_up = F.softmax(logits_up, dim=-1)
        probs_down = F.softmax(logits_down, dim=-1)
        probs_lhand = F.softmax(logits_lhand, dim=-1)
        probs_rhand = F.softmax(logits_rhand, dim=-1)
        probs_transl = F.softmax(logits_transl, dim=-1)

        # print(logits_up, logits_down, loss_lhand, loss_rhand, flush=True)
        return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss
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

            if len(targets_up.size()) == 2:
                loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
                loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))
                loss_lhand = F.cross_entropy(logits_lhand.view(-1, logits_lhand.size(-1)), targets_lhand.view(-1))
                loss_rhand = F.cross_entropy(logits_rhand.view(-1, logits_rhand.size(-1)), targets_rhand.view(-1))
                loss_transl = F.cross_entropy(logits_transl.view(-1, logits_transl.size(-1)), targets_transl.view(-1))
            else:
                # print(logits_down, logits_transl, logits_down.min(), logits_down.max(), logits_transl.min(), logits_transl.max(), flush=True)
                # print(logits_up.view(-1, logits_up.size(-1)).size(), targets_up.view(-1, targets_up.size(-1)).size(), targets_up.view(-1, targets_up.size(-1)), flush=True)
                loss_up = -torch.mean(torch.log(torch.softmax(logits_up.view(-1, logits_up.size(-1)), dim=-1) + 1e-9) * targets_up.view(-1, targets_up.size(-1)))
                loss_down = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-9) * targets_down.view(-1, targets_down.size(-1)))
                loss_lhand = -torch.mean(torch.log(torch.softmax(logits_lhand.view(-1, logits_lhand.size(-1)), dim=-1) + 1e-9) * targets_lhand.view(-1, targets_lhand.size(-1)))
                loss_rhand = -torch.mean(torch.log(torch.softmax(logits_rhand.view(-1, logits_rhand.size(-1)), dim=-1) + 1e-9) * targets_rhand.view(-1, targets_rhand.size(-1)))
                loss_transl = -torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-9) * targets_transl.view(-1, targets_transl.size(-1)))

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

# class QNet(nn.Module):
#     """  the full GPT language model, with a context size of block_size """

#     def __init__(self, config):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
#         # decoder head
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.alpha = config.alpha

#         self.block_size = config.block_size
#         self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

#         self.look_forward = config.look_forward
#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def configure_optimizers(self, train_config):
#         """
#         This long function is unfortunately doing something very simple and is being very defensive:
#         We are separating out all parameters of the model into two buckets: those that will experience
#         weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
#         We are then returning the PyTorch optimizer object.
#         """

#         # separate out all parameters to those that will and won't experience regularizing weight decay
#         decay = set()
#         no_decay = set()
#         whitelist_weight_modules = (torch.nn.Linear, )
#         blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#                 if pn.endswith('bias'):
#                     # all biases will not be decayed
#                     no_decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                     # weights of whitelist modules will be weight decayed
#                     decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                     # weights of blacklist modules will NOT be weight decayed
#                     no_decay.add(fpn)

#         # special case the position embedding parameter in the root GPT module as not decayed
#         no_decay.add('pos_emb')

#         # validate that we considered every parameter
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         inter_params = decay & no_decay
#         union_params = decay | no_decay
#         assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

#         # create the pytorch optimizer object
#         optim_groups = [
#             {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
#             {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#         ]
#         optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
#         return optimizer

#     def forward(self, x, targets=None, reward=None, probs=None):

#         x = self.blocks(x)
#         x = self.ln_f(x)
#         N, T, C = x.size()
#         t = T // 10
#         l = self.look_forward

#         logits_up = self.head_up(x[:, t*5:t*6, :])
#         logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
#         logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
#         logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
#         logits_transl = self.head_transl(x[:, t*9:t*10, :])
        
#         probs_up = F.softmax(logits_up, dim=-1)
#         probs_down = F.softmax(logits_down, dim=-1)
#         probs_lhand = F.softmax(logits_lhand, dim=-1)
#         probs_rhand = F.softmax(logits_rhand, dim=-1)
#         probs_transl = F.softmax(logits_transl, dim=-1)

#         # if we are given some desired targets also calculate the loss
#         # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
#         loss = None
#         if targets is not None:
#             targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
#             reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
#             probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
#             # print(logits_up.size(), targets_up.size())

#             logits_up_q = qindex(logits_up, targets_up)
#             logits_down_q = qindex(logits_down, targets_down)
#             logits_lhand_q = qindex(logits_lhand, targets_lhand)
#             logits_rhand_q = qindex(logits_rhand, targets_rhand)
#             logits_transl_q = qindex(logits_transl, targets_transl)

#             v_up = torch.mean(logits_up * probs_pi_up, dim=-1)
#             v_down = torch.mean(logits_down * probs_pi_down, dim=-1)
#             v_lhand = torch.mean(logits_lhand * probs_pi_lhand, dim=-1)
#             v_rhand = torch.mean(logits_rhand * probs_pi_rhand, dim=-1)
#             v_transl = torch.mean(logits_transl * probs_pi_transl, dim=-1)


#             # print(logits_transl, flush=True)
#             # # for ii in range(len(logits_transl[0])):
#             # print(targets_transl, flush=True)
#             # print(torch.topk(logits_transl, k=5, dim=-1))
#             # print(reward_transl, flush=True)

#             probs_pi_up_q = qindex(probs_pi_up, targets_up)
#             probs_pi_down_q = qindex(probs_pi_up, targets_down)
#             probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
#             probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
#             probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

#             y_up =  (0.8 * v_up[:, 1:] + reward_up[: ,:-1] - self.alpha * torch.log(probs_pi_up_q[:, 1:])).clone().detach()
#             y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
#             y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
#             y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
#             y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

#             # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
#             loss_up = F.l1_loss(logits_up_q[:, :-1], y_up)
#             loss_down = F.l1_loss(logits_down_q[:, :-1], y_down)
#             loss_lhand = F.l1_loss(logits_lhand_q[:, :-1], y_lhand)
#             loss_rhand = F.l1_loss(logits_rhand_q[:, :-1], y_rhand)
#             loss_transl = F.l1_loss(logits_transl_q[:, :-1], y_transl)

#             loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

#         return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss
#     def freeze_drop(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.eval()
#                 m.requires_grad = False

# class QNet2(nn.Module):
#     """  the full GPT language model, with a context size of block_size """

#     def __init__(self, config):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
#         # decoder head
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.alpha = config.alpha

#         self.block_size = config.block_size
#         self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

#         self.look_forward = config.look_forward
#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def configure_optimizers(self, train_config):
#         """
#         This long function is unfortunately doing something very simple and is being very defensive:
#         We are separating out all parameters of the model into two buckets: those that will experience
#         weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
#         We are then returning the PyTorch optimizer object.
#         """

#         # separate out all parameters to those that will and won't experience regularizing weight decay
#         decay = set()
#         no_decay = set()
#         whitelist_weight_modules = (torch.nn.Linear, )
#         blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#                 if pn.endswith('bias'):
#                     # all biases will not be decayed
#                     no_decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                     # weights of whitelist modules will be weight decayed
#                     decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                     # weights of blacklist modules will NOT be weight decayed
#                     no_decay.add(fpn)

#         # special case the position embedding parameter in the root GPT module as not decayed
#         no_decay.add('pos_emb')

#         # validate that we considered every parameter
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         inter_params = decay & no_decay
#         union_params = decay | no_decay
#         assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

#         # create the pytorch optimizer object
#         optim_groups = [
#             {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
#             {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#         ]
#         optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
#         return optimizer

#     def forward(self, x, targets=None, reward=None, probs=None):

#         x = self.blocks(x)
#         x = self.ln_f(x)
#         N, T, C = x.size()
#         t = T // 10
#         l = self.look_forward

#         logits_up = self.head_up(x[:, t*5:t*6, :])
#         logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
#         logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
#         logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
#         logits_transl = self.head_transl(x[:, t*9:t*10, :])
        
#         probs_up = F.softmax(logits_up, dim=-1)
#         probs_down = F.softmax(logits_down, dim=-1)
#         probs_lhand = F.softmax(logits_lhand, dim=-1)
#         probs_rhand = F.softmax(logits_rhand, dim=-1)
#         probs_transl = F.softmax(logits_transl, dim=-1)

#         # if we are given some desired targets also calculate the loss
#         # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
#         loss = None
#         if targets is not None:
#             targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
#             reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
#             probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
#             # print(logits_up.size(), targets_up.size())

#             logits_up_q = qindex(logits_up, targets_up)
#             logits_down_q = qindex(logits_down, targets_down)
#             logits_lhand_q = qindex(logits_lhand, targets_lhand)
#             logits_rhand_q = qindex(logits_rhand, targets_rhand)
#             logits_transl_q = qindex(logits_transl, targets_transl)

#             v_up = torch.mean(logits_up * probs_pi_up, dim=-1)
#             v_down = torch.mean(logits_down * probs_pi_down, dim=-1)
#             v_lhand = torch.mean(logits_lhand * probs_pi_lhand, dim=-1)
#             v_rhand = torch.mean(logits_rhand * probs_pi_rhand, dim=-1)
#             v_transl = torch.mean(logits_transl * probs_pi_transl, dim=-1)


#             # print(logits_transl, flush=True)
#             # # for ii in range(len(logits_transl[0])):
#             # print(targets_transl, flush=True)
#             # print(torch.topk(logits_transl, k=5, dim=-1))
#             # print(reward_transl, flush=True)

#             probs_pi_up_q = qindex(probs_pi_up, targets_up)
#             probs_pi_down_q = qindex(probs_pi_up, targets_down)
#             probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
#             probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
#             probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

#             loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1), reduction='none') * (reward_up[: ,:] - self.alpha * torch.log(probs_pi_up_q[:, :])).view(-1).clone().detach()
#             loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1), reduction='none') * (reward_down[:, :] - self.alpha * torch.log(probs_pi_down_q[:, :])).view(-1).clone().detach()
#             loss_lhand = F.cross_entropy(logits_lhand.view(-1, logits_lhand.size(-1)), targets_lhand.view(-1), reduction='none') *  (reward_lhand[:, :] - self.alpha * torch.log(probs_pi_lhand_q[:, :])).view(-1).clone().detach()
#             loss_rhand = F.cross_entropy(logits_rhand.view(-1, logits_rhand.size(-1)), targets_rhand.view(-1), reduction='none') * (reward_rhand[:, :] - self.alpha * torch.log(probs_pi_rhand_q[:, :])).view(-1).clone().detach() 
#             loss_transl = F.cross_entropy(logits_transl.view(-1, logits_transl.size(-1)), targets_transl.view(-1), reduction='none') * (reward_transl[:, :] - self.alpha * torch.log(probs_pi_transl_q[:, :])).view(-1).clone().detach()

#             # y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
#             # y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
#             # y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
#             # y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

#             # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
#             # loss_up = F.l1_loss(logits_up_q[:, :-1], y_up)
#             # loss_down = F.l1_loss(logits_down_q[:, :-1], y_down)
#             # loss_lhand = F.l1_loss(logits_lhand_q[:, :-1], y_lhand)
#             # loss_rhand = F.l1_loss(logits_rhand_q[:, :-1], y_rhand)
#             # loss_transl = F.l1_loss(logits_transl_q[:, :-1], y_transl)

#             loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

#         return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss
#     def freeze_drop(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.eval()
#                 m.requires_grad = False



# class QNet3(nn.Module):
#     """  the full GPT language model, with a context size of block_size """

#     def __init__(self, config):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
#         # decoder head
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.alpha = config.alpha

#         self.block_size = config.block_size
#         self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

#         self.look_forward = config.look_forward
#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         # if isinstance(module, (nn.Linear, nn.Embedding)):
#         #     module.weight.data.normal_(mean=0.0, std=0.02)
#         #     if isinstance(module, nn.Linear) and module.bias is not None:
#         #         module.bias.data.zero_()
#         if isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def configure_optimizers(self, train_config):
#         """
#         This long function is unfortunately doing something very simple and is being very defensive:
#         We are separating out all parameters of the model into two buckets: those that will experience
#         weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
#         We are then returning the PyTorch optimizer object.
#         """

#         # separate out all parameters to those that will and won't experience regularizing weight decay
#         decay = set()
#         no_decay = set()
#         whitelist_weight_modules = (torch.nn.Linear, )
#         blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#                 if pn.endswith('bias'):
#                     # all biases will not be decayed
#                     no_decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                     # weights of whitelist modules will be weight decayed
#                     decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                     # weights of blacklist modules will NOT be weight decayed
#                     no_decay.add(fpn)

#         # special case the position embedding parameter in the root GPT module as not decayed
#         no_decay.add('pos_emb')

#         # validate that we considered every parameter
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         inter_params = decay & no_decay
#         union_params = decay | no_decay
#         assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

#         # create the pytorch optimizer object
#         optim_groups = [
#             {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
#             {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#         ]
#         optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
#         return optimizer

#     def forward(self, x, targets=None, reward=None, probs=None, pretrain=False):

#         x = self.blocks(x)
#         x = self.ln_f(x)
#         N, T, C = x.size()
#         t = T // 10
#         l = self.look_forward

#         logits_up = self.head_up(x[:, t*5:t*6, :])
#         logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
#         logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
#         logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
#         logits_transl = self.head_transl(x[:, t*9:t*10, :])
        
#         probs_up = F.softmax(logits_up, dim=-1)
#         probs_down = F.softmax(logits_down, dim=-1)
#         probs_lhand = F.softmax(logits_lhand, dim=-1)
#         probs_rhand = F.softmax(logits_rhand, dim=-1)
#         probs_transl = F.softmax(logits_transl, dim=-1)

#         # if we are given some desired targets also calculate the loss
#         # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
#         loss = None
#         ce_loss = None
#         if targets is not None:
#             targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
#             reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
#             probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
#             # print(logits_up.size(), targets_up.size())

#             logits_up_q = qindex(logits_up, targets_up)
#             logits_down_q = qindex(logits_down, targets_down)
#             logits_lhand_q = qindex(logits_lhand, targets_lhand)
#             logits_rhand_q = qindex(logits_rhand, targets_rhand)
#             logits_transl_q = qindex(logits_transl, targets_transl)

#             # NxT
#             reward_up = reward_up + 1
#             reward_lhand = reward_lhand + 1
#             reward_rhand = reward_rhand + 1
#             # v_up = torch.mean(logits_up * probs_pi_up, dim=-1)
#             # v_down = torch.mean(logits_down * probs_pi_down, dim=-1)
#             # v_lhand = torch.mean(logits_lhand * probs_pi_lhand, dim=-1)
#             # v_rhand = torch.mean(logits_rhand * probs_pi_rhand, dim=-1)
#             # v_transl = torch.mean(logits_transl * probs_pi_transl, dim=-1)


#             # print(logits_transl, flush=True)
#             # # for ii in range(len(logits_transl[0])):
#             # print(targets_transl, flush=True)
#             # print(torch.topk(logits_transl, k=5, dim=-1))
#             # print(reward_transl, flush=True)

#             probs_pi_up_q = qindex(probs_pi_up, targets_up)
#             probs_pi_down_q = qindex(probs_pi_down, targets_down)
#             probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
#             probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
#             probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

#             y_up = reward_up.unsqueeze(-1) + torch.log(probs_pi_up/(probs_pi_up_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_down = reward_down.unsqueeze(-1) + torch.log(probs_pi_down/(probs_pi_down_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_lhand = reward_lhand.unsqueeze(-1) + torch.log(probs_pi_lhand/(probs_pi_lhand_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_rhand = reward_rhand.unsqueeze(-1) + torch.log(probs_pi_rhand/(probs_pi_rhand_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_transl = reward_transl.unsqueeze(-1) + torch.log(probs_pi_transl/(probs_pi_transl_q.unsqueeze(-1) + 1e-9)+1e-9)
#             # y_up =  (0.8 * v_up[:, 1:] + reward_up[: ,:-1] - self.alpha * torch.log(probs_pi_up_q[:, 1:])).clone().detach()
#             # y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
#             # y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
#             # y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
#             # y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

#             # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
#             loss_up = F.l1_loss(logits_up, y_up.clone().detach())
#             loss_down = F.l1_loss(logits_down, y_down.clone().detach())
#             loss_lhand = F.l1_loss(logits_lhand, y_lhand.clone().detach())
#             loss_rhand = F.l1_loss(logits_rhand, y_rhand.clone().detach())
#             loss_transl = F.l1_loss(logits_transl, y_transl.clone().detach())

#             loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

#             ce_loss = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-9) * probs_pi_down.view(-1, probs_pi_down.size(-1))) - torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-9) * probs_pi_transl.view(-1, probs_pi_transl.size(-1)))

#         return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss, ce_loss 
#     def freeze_drop(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.eval()
#                 m.requires_grad = False

# class QNet4(nn.Module):
#     """  the full GPT language model, with a context size of block_size """

#     def __init__(self, config):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
#         # decoder head
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.alpha = config.alpha

#         self.block_size = config.block_size
#         self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

#         self.look_forward = config.look_forward
#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         if isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def configure_optimizers(self, train_config):
#         """
#         This long function is unfortunately doing something very simple and is being very defensive:
#         We are separating out all parameters of the model into two buckets: those that will experience
#         weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
#         We are then returning the PyTorch optimizer object.
#         """

#         # separate out all parameters to those that will and won't experience regularizing weight decay
#         decay = set()
#         no_decay = set()
#         whitelist_weight_modules = (torch.nn.Linear, )
#         blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#                 if pn.endswith('bias'):
#                     # all biases will not be decayed
#                     no_decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                     # weights of whitelist modules will be weight decayed
#                     decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                     # weights of blacklist modules will NOT be weight decayed
#                     no_decay.add(fpn)

#         # special case the position embedding parameter in the root GPT module as not decayed
#         no_decay.add('pos_emb')

#         # validate that we considered every parameter
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         inter_params = decay & no_decay
#         union_params = decay | no_decay
#         assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

#         # create the pytorch optimizer object
#         optim_groups = [
#             {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
#             {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#         ]
#         optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
#         return optimizer

#     def forward(self, x, targets=None, reward=None, probs=None, pretrain=False):

#         x = self.blocks(x)
#         x = self.ln_f(x)
#         N, T, C = x.size()
#         t = T // 10
#         l = self.look_forward

#         logits_up = self.head_up(x[:, t*5:t*6, :])
#         logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
#         logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
#         logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
#         logits_transl = self.head_transl(x[:, t*9:t*10, :])
        
#         probs_up = F.softmax(logits_up, dim=-1)
#         probs_down = F.softmax(logits_down, dim=-1)
#         probs_lhand = F.softmax(logits_lhand, dim=-1)
#         probs_rhand = F.softmax(logits_rhand, dim=-1)
#         probs_transl = F.softmax(logits_transl, dim=-1)

#         # if we are given some desired targets also calculate the loss
#         # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
#         loss = None
#         ce_loss = None
#         if targets is not None:
#             targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
#             reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
#             probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
#             # print(logits_up.size(), targets_up.size())

#             logits_up_q = qindex(logits_up, targets_up)
#             logits_down_q = qindex(logits_down, targets_down)
#             logits_lhand_q = qindex(logits_lhand, targets_lhand)
#             logits_rhand_q = qindex(logits_rhand, targets_rhand)
#             logits_transl_q = qindex(logits_transl, targets_transl)

#             # NxT
#             reward_up = reward_up + 1
#             reward_lhand = reward_lhand + 1
#             reward_rhand = reward_rhand + 1
#             # v_up = torch.mean(logits_up * probs_pi_up, dim=-1)
#             # v_down = torch.mean(logits_down * probs_pi_down, dim=-1)
#             # v_lhand = torch.mean(logits_lhand * probs_pi_lhand, dim=-1)
#             # v_rhand = torch.mean(logits_rhand * probs_pi_rhand, dim=-1)
#             # v_transl = torch.mean(logits_transl * probs_pi_transl, dim=-1)


#             # print(logits_transl, flush=True)
#             # # for ii in range(len(logits_transl[0])):
#             # print(targets_transl, flush=True)
#             # print(torch.topk(logits_transl, k=5, dim=-1))
#             # print(reward_transl, flush=True)

#             probs_pi_up_q = qindex(probs_pi_up, targets_up)
#             probs_pi_down_q = qindex(probs_pi_down, targets_down)
#             probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
#             probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
#             probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

#             y_up = reward_up.unsqueeze(-1) + torch.log(probs_pi_up/(probs_pi_up_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_down = reward_down.unsqueeze(-1) + torch.log(probs_pi_down/(probs_pi_down_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_lhand = reward_lhand.unsqueeze(-1) + torch.log(probs_pi_lhand/(probs_pi_lhand_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_rhand = reward_rhand.unsqueeze(-1) + torch.log(probs_pi_rhand/(probs_pi_rhand_q.unsqueeze(-1) + 1e-9)+1e-9)
#             y_transl = reward_transl.unsqueeze(-1) + torch.log(probs_pi_transl/(probs_pi_transl_q.unsqueeze(-1) + 1e-9)+1e-9)
#             # y_up =  (0.8 * v_up[:, 1:] + reward_up[: ,:-1] - self.alpha * torch.log(probs_pi_up_q[:, 1:])).clone().detach()
#             # y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
#             # y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
#             # y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
#             # y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

#             # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
#             loss_up = F.l1_loss(logits_up, y_up.clone().detach())
#             loss_down = F.l1_loss(logits_down, y_down.clone().detach())
#             loss_lhand = F.l1_loss(logits_lhand, y_lhand.clone().detach())
#             loss_rhand = F.l1_loss(logits_rhand, y_rhand.clone().detach())
#             loss_transl = F.l1_loss(logits_transl, y_transl.clone().detach())

#             loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

#             ce_loss = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-9) * probs_pi_down.view(-1, probs_pi_down.size(-1))) - torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-9) * probs_pi_transl.view(-1, probs_pi_transl.size(-1)))

#         return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss, ce_loss 
#     def freeze_drop(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.eval()
#                 m.requires_grad = False
    
# class QNet5(nn.Module):
#     """  the full GPT language model, with a context size of block_size """

#     def __init__(self, config):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
#         self.blocks_rescale = nn.Sequential(*[Block(config) for _ in range(config.n_layer_rescale)])

#         # decoder head
#         self.ln_f = nn.LayerNorm(config.n_embd)
#         self.alpha = config.alpha

#         self.block_size = config.block_size
#         self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
#         self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
#         self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

#         self.head_up_rescale = nn.Linear(config.n_embd, 1, bias=True)
#         self.head_down_rescale = nn.Linear(config.n_embd, 1, bias=True)
#         self.head_lhand_rescale = nn.Linear(config.n_embd, 1, bias=True)
#         self.head_rhand_rescale = nn.Linear(config.n_embd, 1, bias=True)
#         self.head_transl_rescale = nn.Linear(config.n_embd, 1, bias=True)


#         self.look_forward = config.look_forward
#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         if isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def configure_optimizers(self, train_config):
#         """
#         This long function is unfortunately doing something very simple and is being very defensive:
#         We are separating out all parameters of the model into two buckets: those that will experience
#         weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
#         We are then returning the PyTorch optimizer object.
#         """

#         # separate out all parameters to those that will and won't experience regularizing weight decay
#         decay = set()
#         no_decay = set()
#         whitelist_weight_modules = (torch.nn.Linear, )
#         blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

#                 if pn.endswith('bias'):
#                     # all biases will not be decayed
#                     no_decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
#                     # weights of whitelist modules will be weight decayed
#                     decay.add(fpn)
#                 elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
#                     # weights of blacklist modules will NOT be weight decayed
#                     no_decay.add(fpn)

#         # special case the position embedding parameter in the root GPT module as not decayed
#         no_decay.add('pos_emb')

#         # validate that we considered every parameter
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         inter_params = decay & no_decay
#         union_params = decay | no_decay
#         assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

#         # create the pytorch optimizer object
#         optim_groups = [
#             {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
#             {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
#         ]
#         optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
#         return optimizer

#     def forward(self, x, targets=None, reward=None, probs=None, pretrain=False):

#         rescale = self.blocks_rescale(x)
#         with torch.no_grad():
#             self.blocks.eval()
#             self.ln_f.eval()

#             x = self.blocks(x)
#             x = self.ln_f(x)
#             N, T, C = x.size()
#             t = T // 10
#             l = self.look_forward

#             logits_up = self.head_up(x[:, t*5:t*6, :])
#             logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
#             logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
#             logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
#             logits_transl = self.head_transl(x[:, t*9:t*10, :])

#             probs_up = F.softmax(logits_up, dim=-1)
#             probs_down = F.softmax(logits_down, dim=-1)
#             probs_lhand = F.softmax(logits_lhand, dim=-1)
#             probs_rhand = F.softmax(logits_rhand, dim=-1)
#             probs_transl = F.softmax(logits_transl, dim=-1)

#         rescale_up = self.head_up_rescale(rescale[:, t*5:t*6, :])
#         rescale_down = self.head_down_rescale(rescale[:, t*6:t*7, :]) # down half 
#         rescale_lhand = self.head_lhand_rescale(rescale[:, t*7:t*8, :])
#         rescale_rhand = self.head_rhand_rescale(rescale[:, t*8:t*9, :])
#         rescale_transl = self.head_transl_rescale(rescale[:, t*9:t*10, :])
        
        

#         # if we are given some desired targets also calculate the loss
#         # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
#         loss = None
#         ce_loss = None
#         if targets is not None:
#             targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
#             reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
#             probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
#             # print(logits_up.size(), targets_up.size())

#             logits_up_q = qindex(logits_up, targets_up)
#             logits_down_q = qindex(logits_down, targets_down)
#             logits_lhand_q = qindex(logits_lhand, targets_lhand)
#             logits_rhand_q = qindex(logits_rhand, targets_rhand)
#             logits_transl_q = qindex(logits_transl, targets_transl)

#             # NxT
#             reward_up = reward_up + 1
#             reward_lhand = reward_lhand + 1
#             reward_rhand = reward_rhand + 1
#             # v_up = torch.mean(logits_up * probs_pi_up, dim=-1)
#             # v_down = torch.mean(logits_down * probs_pi_down, dim=-1)
#             # v_lhand = torch.mean(logits_lhand * probs_pi_lhand, dim=-1)
#             # v_rhand = torch.mean(logits_rhand * probs_pi_rhand, dim=-1)
#             # v_transl = torch.mean(logits_transl * probs_pi_transl, dim=-1)


#             # print(logits_transl, flush=True)
#             # # for ii in range(len(logits_transl[0])):
#             # print(targets_transl, flush=True)
#             # print(torch.topk(logits_transl, k=5, dim=-1))
#             # print(reward_transl, flush=True)

#             probs_pi_up_q = qindex(probs_pi_up, targets_up)
#             probs_pi_down_q = qindex(probs_pi_down, targets_down)
#             probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
#             probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
#             probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

#             # print(rescale_up.size(), reward_up.size(), logits_up_q.size(), flush=True)
#             y_up = reward_up - logits_up_q
#             y_down = reward_down - logits_down_q
#             y_lhand = reward_lhand - logits_lhand_q
#             y_rhand = reward_rhand - logits_rhand_q
#             y_transl = reward_transl - logits_transl_q
#             # y_up =  (0.8 * v_up[:, 1:] + reward_up[: ,:-1] - self.alpha * torch.log(probs_pi_up_q[:, 1:])).clone().detach()
#             # y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
#             # y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
#             # y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
#             # y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

#             # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
#             loss_up = F.l1_loss(rescale_up.squeeze(-1), y_up.clone().detach())
#             loss_down = F.l1_loss(rescale_down.squeeze(-1), y_down.clone().detach())
#             loss_lhand = F.l1_loss(rescale_lhand.squeeze(-1), y_lhand.clone().detach())
#             loss_rhand = F.l1_loss(rescale_rhand.squeeze(-1), y_rhand.clone().detach())
#             loss_transl = F.l1_loss(rescale_transl.squeeze(-1), y_transl.clone().detach())

#             loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

#             ce_loss = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-10) * probs_pi_down.view(-1, probs_pi_down.size(-1))) - torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-10) * probs_pi_transl.view(-1, probs_pi_transl.size(-1)))

#         return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss, ce_loss 
#     def freeze_drop(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.eval()
#                 m.requires_grad = False


# class QNet6(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks_rescale = nn.Sequential(*[Block(config) for _ in range(config.n_layer_rescale)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.alpha = config.alpha

        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_lhand = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_rhand = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_transl = nn.Linear(config.n_embd, config.vocab_size_transl, bias=False)

        self.head_up_rescale = nn.Linear(config.n_embd, 1, bias=True)
        self.head_down_rescale = nn.Linear(config.n_embd, 1, bias=True)
        self.head_lhand_rescale = nn.Linear(config.n_embd, 1, bias=True)
        self.head_rhand_rescale = nn.Linear(config.n_embd, 1, bias=True)
        self.head_transl_rescale = nn.Linear(config.n_embd, 1, bias=True)


        self.look_forward = config.look_forward
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
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

    def forward(self, x, targets=None, reward=None, probs=None, pretrain=False):


        x = self.blocks(x)
        self.ln_f.eval()
        self.ln_f.requires_grad = False
        x = self.ln_f(x)
        N, T, C = x.size()
        t = T // 10
        l = self.look_forward

        logits_up = self.head_up(x[:, t*5:t*6, :])
        logits_down = self.head_down(x[:, t*6:t*7, :]) # down half 
        logits_lhand = self.head_lhand(x[:, t*7:t*8, :])
        logits_rhand = self.head_rhand(x[:, t*8:t*9, :])
        logits_transl = self.head_transl(x[:, t*9:t*10, :])

        probs_up = F.softmax(logits_up, dim=-1)
        probs_down = F.softmax(logits_down, dim=-1)
        probs_lhand = F.softmax(logits_lhand, dim=-1)
        probs_rhand = F.softmax(logits_rhand, dim=-1)
        probs_transl = F.softmax(logits_transl, dim=-1)

        # with torch.no_grad():
        self.blocks_rescale.eval()
        rescale = self.blocks_rescale(x)

        rescale_up = self.head_up_rescale(rescale[:, t*5:t*6, :])
        rescale_down = self.head_down_rescale(rescale[:, t*6:t*7, :]) # down half 
        rescale_lhand = self.head_lhand_rescale(rescale[:, t*7:t*8, :])
        rescale_rhand = self.head_rhand_rescale(rescale[:, t*8:t*9, :])
        rescale_transl = self.head_transl_rescale(rescale[:, t*9:t*10, :])
            
        

        # if we are given some desired targets also calculate the loss
        # loss_up, loss_down, loss_lhand, loss_rhand, loss_transl = None, None, None, None, None
        loss = None
        ce_loss = None

        N, T, K = logits_up.size()

        if targets is not None:
            targets_up, targets_down, targets_lhand, targets_rhand, targets_transl = targets
            reward_up, reward_down, reward_lhand, reward_rhand, reward_transl = reward
            probs_pi_up, probs_pi_down, probs_pi_lhand, probs_pi_rhand, probs_pi_transl = probs
            
            # print(logits_up.size(), targets_up.size())

            logits_up_q = qindex(logits_up, targets_up)
            logits_down_q = qindex(logits_down, targets_down)
            logits_lhand_q = qindex(logits_lhand, targets_lhand)
            logits_rhand_q = qindex(logits_rhand, targets_rhand)
            logits_transl_q = qindex(logits_transl, targets_transl)

            # NxT
            reward_up = reward_up + 1
            reward_lhand = reward_lhand + 1
            reward_rhand = reward_rhand + 1

            reward_transl = reward_transl + reward_down
            v_up = torch.mean(logits_up.clone().detach() * probs_pi_up, dim=-1)
            v_down = torch.mean(logits_down.clone().detach() * probs_pi_down, dim=-1)
            v_lhand = torch.mean(logits_lhand.clone().detach() * probs_pi_lhand, dim=-1)
            v_rhand = torch.mean(logits_rhand.clone().detach() * probs_pi_rhand, dim=-1)
            v_transl = torch.mean(logits_transl.clone().detach() * probs_pi_transl, dim=-1)


            # print(logits_transl, flush=True)
            # # for ii in range(len(logits_transl[0])):
            # print(targets_transl, flush=True)
            # print(torch.topk(logits_transl, k=5, dim=-1))
            # print(reward_transl, flush=True)

            probs_pi_up_q = qindex(probs_pi_up, targets_up)
            probs_pi_down_q = qindex(probs_pi_up, targets_down)
            probs_pi_lhand_q = qindex(probs_pi_lhand, targets_lhand)
            probs_pi_rhand_q = qindex(probs_pi_rhand, targets_rhand)
            probs_pi_transl_q = qindex(probs_pi_transl, targets_transl)

            # print(reward_up.size(), rescale_up.size(), flush=True)
            y_up =  reward_up - rescale_up.squeeze(-1)
            y_down = reward_down - rescale_down.squeeze(-1)
            y_lhand =  reward_lhand - rescale_lhand.squeeze(-1)
            y_rhand =  reward_rhand - rescale_rhand.squeeze(-1)
            y_transl = reward_transl - rescale_transl.squeeze(-1)

            y_up[:, :-1] +=  (0.5 * v_up[:, 1:]  - self.alpha * torch.log(probs_pi_up_q[:, 1:]))
            y_down[:, :-1] +=  (0.5 * v_down[:, 1:]  - self.alpha * torch.log(probs_pi_down_q[:, 1:]))
            y_lhand[:, :-1] +=  (0.5 * v_lhand[:, 1:]  - self.alpha * torch.log(probs_pi_lhand_q[:, 1:]))
            y_rhand[:, :-1] +=  (0.5 * v_rhand[:, 1:]  - self.alpha * torch.log(probs_pi_rhand_q[:, 1:]))
            y_transl[:, :-1] +=  ( 0.5 *v_transl[:, 1:]  - self.alpha * torch.log(probs_pi_transl_q[:, 1:]))

            # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
            loss_up = F.l1_loss(logits_up_q[:, :], y_up.clone().detach())
            loss_down = F.l1_loss(logits_down_q[:, :], y_down.clone().detach())
            loss_lhand = F.l1_loss(logits_lhand_q[:, :], y_lhand.clone().detach())
            loss_rhand = F.l1_loss(logits_rhand_q[:, :], y_rhand.clone().detach())
            loss_transl = F.l1_loss(logits_transl_q[:, :], y_transl.clone().detach())
            # y_up =  (0.8 * v_up[:, 1:] + reward_up[: ,:-1] - self.alpha * torch.log(probs_pi_up_q[:, 1:])).clone().detach()
            # y_down =  (0.8 * v_down[:, 1:] + reward_down[:, :-1] - self.alpha * torch.log(probs_pi_down_q[:, 1:])).clone().detach()
            # y_lhand =  (0.8 * v_lhand[:, 1:] + reward_lhand[:, :-1] - self.alpha * torch.log(probs_pi_lhand_q[:, 1:])).clone().detach()
            # y_rhand =  (0.8 * v_rhand[:, 1:] + reward_rhand[:, :-1] - self.alpha * torch.log(probs_pi_rhand_q[:, 1:])).clone().detach()
            # y_transl =  ( 0.8 *v_transl[:, 1:] + reward_transl[:, :-1] - self.alpha * torch.log(probs_pi_transl_q[:, 1:])).clone().detach()

            # print(y_up.size(), y_down.size(), y_lhand.size(), y_rhand.size(), y_transl.size(), flush=True)
            loss_up = F.l1_loss(rescale_up.squeeze(-1), y_up.clone().detach()) / K
            loss_down = F.l1_loss(rescale_down.squeeze(-1), y_down.clone().detach()) / K
            loss_lhand = F.l1_loss(rescale_lhand.squeeze(-1), y_lhand.clone().detach()) / K 
            loss_rhand = F.l1_loss(rescale_rhand.squeeze(-1), y_rhand.clone().detach()) / K 
            loss_transl = F.l1_loss(rescale_transl.squeeze(-1), y_transl.clone().detach()) / K

            loss = loss_up + loss_down + loss_lhand + loss_rhand + loss_transl

            ce_loss = -torch.mean(torch.log(torch.softmax(logits_down.view(-1, logits_down.size(-1)), dim=-1) + 1e-10) * probs_pi_down.view(-1, probs_pi_down.size(-1))) - torch.mean(torch.log(torch.softmax(logits_transl.view(-1, logits_transl.size(-1)), dim=-1) + 1e-10) * probs_pi_transl.view(-1, probs_pi_transl.size(-1)))

        return (probs_up, probs_down, probs_lhand, probs_rhand, probs_transl), loss, ce_loss 
    def freeze_drop(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
                m.requires_grad = False