import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os


def paired_collate_fn(insts):
    # for src in insts:
    #     for s in src:
    #         print(s.shape)

    # print()

    mo_seq = list(zip(*insts))
    print(mo_seq.shape)
    mo_seq = torch.FloatTensor(mo_seq)
    print('Here!')
    print(mo_seq.size())

    return mo_seq


class AClf(Dataset):
    def __init__(self, sequences, interval, look_forward, music_code_rate=4, expand_rate=100):

        self.quants_l = []
        self.quants_f_in = []
        self.quants_f_target = []
        self.music_seqs = []
        self.rewards = []
        self.expand_rate = expand_rate

        # 101*4 
        for seq in sequences:
            music_seq, zseql, zseqf, rewards = seq
            # print(len(music_seq), len(zseql), len(zseqf), len(rewards), flush=True)
            for ii in range(len(zseql[0]) - look_forward - interval - 1):
                
                this_quant_l = tuple(zseql[kk][ii:ii+interval+look_forward] for kk in range(len(zseql)))
                this_quant_f_in = tuple(zseqf[kk][ii:ii+interval] for kk in range(len(zseqf)))
                this_quant_f_target = tuple(zseqf[kk][ii+1:ii+interval+1] for kk in range(len(zseqf)))
                this_music_seq = music_seq[(ii+1)*music_code_rate:(ii+interval+look_forward+1)*music_code_rate]
                this_reward = tuple(rewards[kk][ii+1:ii+interval+1] for kk in range(len(rewards)))

                # print(len(this_music_seq), len(this_quant_l[0]), len(this_quant_f_in[0]), len(this_quant_f_target[0]), len(this_reward[0]))
                # print(interval, len(this_reward[0]), flush=True)
                if len(this_music_seq) == (interval+look_forward)*music_code_rate:
                    self.quants_l.append(this_quant_l)
                    self.quants_f_in.append(this_quant_f_in)
                    self.quants_f_target.append(this_quant_f_target)
                    self.music_seqs.append(this_music_seq)
                    self.rewards.append(this_reward)

    def __len__(self):
        return len(self.quants_l) * self.expand_rate

    def __getitem__(self, index):
        index = index // self.expand_rate
        return {'music_feat':self.music_seqs[index], 'quants_l':self.quants_l[index], 'quants_input':self.quants_f_in[index], 'quants_target':self.quants_f_target[index], 'rewards':self.rewards[index]}

if __name__ == '__main__':
    dd100 = DD100lf('/mnt/lustre/syli/duet/data/music', '/mnt/lustre/syli/duet/data/motion', split='half_test', interval=None, dtype='both')
    print(len(dd100.dances['pos3dl']), len(dd100.dances['rotmatl']), len(dd100.dances['pos3df']), len(dd100.dances['rotmatf']), len(dd100.dances['music']), len(dd100.names))
    for ii in range(len(dd100.dances['pos3dl'])):
        # if len(dd100.dances['pos3d'][ii]) != len(dd100.dances['music'][ii]):
        print(len(dd100.dances['pos3dl'][ii]), len(dd100.dances['rotmatl'][ii]), len(dd100.dances['pos3df'][ii]), len(dd100.dances['rotmatf'][ii]), len(dd100.dances['music'][ii]), dd100.names[ii], flush=True)
        # print(dd100.dances['rotmatl'][ii][55], dd100.dances['rotmatf'][ii][55])