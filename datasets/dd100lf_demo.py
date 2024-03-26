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


class DD100lfDemo(Dataset):
    def __init__(self, music_root, motion_root, split='train', interval=None, dtype='pos3d', move=8, music_dance_rate=1, expansion=1):
        self.dances = {'rotmatl':[],  'pos3dl':[], 'music':[]}
        dtypes = ['rotmat', 'pos3d']
        self.dtype = dtype
        self.expansion = expansion
        self.names = []

        music_files = {}
        agent_files = {'leader':{}}
        
        music_seqs = {}

        for subfolder in os.listdir(os.path.join(music_root, 'feature')):
            for mname in os.listdir(os.path.join(music_root, 'feature', subfolder)):
                path = os.path.join(music_root, 'feature', subfolder, mname)
                music_files[mname[:-4]] = path

        for fname in os.listdir(os.path.join(motion_root, 'pos3d', split)): 
            path = os.path.join(motion_root, 'pos3d', split, fname)
            # if path.endswith('_00.npy'):
            #     agent_files['follower'][fname[:-7]] = path
            # elif path.endswith('_01.npy'):
            agent_files['leader'][fname[:-4]] = path
            
        for take in agent_files['leader']:
            # print(take)
            # if take not in agent_files['leader'] or take[:-4] not in music_files:
            #     continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            len_this_music = len(np_music) // int(music_dance_rate) * int(music_dance_rate)
            # print(len_this_music)
            
            for dtype_folder in dtypes:
                this_pair = {}
                for agent in agent_files:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    np_dance = np.load(dance_path)

                    if dtype_folder == 'pos3d':
                        root = np_dance[:, :3]  # the root
                        np_dance = np_dance - np.tile(root, (1, 55))  # Calculate relative offset with respect to root
                        np_dance[:, :3] = root
                        # the root of left hand
                        left_twist = np_dance[:, 60:63]
                        # 25,40
                        np_dance[:, 75:120] = (np_dance[:, 75:120] - np.tile(left_twist, (1, 15))) * 10
                        # the root of right hand
                        right_twist = np_dance[:, 63:66]
                        # 40,55
                        np_dance[:, 120:165] = (np_dance[:, 120:165] - np.tile(right_twist, (1, 15))) * 10
                    if dtype_folder == 'rotmat':
                        np_dance = np_dance[:, 3:]
                    this_pair[agent] = np_dance

                ldance = this_pair['leader']
                lenf, dim = ldance.shape
                # lenl, dim = fdance.shape
                seq_len = lenf
                # print(lenf, lenl)

                if (interval is not None) and ( interval != 'None' ):
                    for i in range(0, seq_len, move):

                        np_dance_sub_seq_l = ldance[i: i + interval]
                        # np_dance_sub_seq_f = fdance[i: i + interval]
                        
                        np_music_sub_seq = np_music[i//music_dance_rate:i//music_dance_rate + interval//music_dance_rate]

                        if len(np_dance_sub_seq_l) != interval or len(np_music_sub_seq) != interval // music_dance_rate:
                            # print(len(np_dance_sub_seq), len(np_music_sub_seq))
                            continue
                        self.dances[dtype_folder+'l'].append(np_dance_sub_seq_l)
                        # self.dances[dtype_folder+'f'].append(np_dance_sub_seq_f)
                        if dtype_folder != 'rotmat':
                            self.dances['music'].append(np_music_sub_seq)
                            self.names.append(take)
                else:
                    self.dances[dtype_folder+'l'].append(ldance[:seq_len//move*move])
                    # self.dances[dtype_folder+'f'].append(fdance[:seq_len//move*move])
                    if dtype_folder != 'rotmat':
                        self.dances['music'].append(np_music[:seq_len//move*move])
                        self.names.append(take)

    def __len__(self):
        # if len(self.dances['pos3dl']) < 20:
        # return 3
        # else:
        #     return 33
            # return len(self.dances['pos3dl'])
        return len(self.dances['pos3dl'])*self.expansion

    def __getitem__(self, index):
        # if len(self.dances['pos3dl']) > 20:
        index = index // self.expansion
        return {'pos3dl':self.dances['pos3dl'][index],  'rotmatl':self.dances['rotmatl'][index], 'music':self.dances['music'][index], 'fname':self.names[index]}

if __name__ == '__main__':
    dd100 = DD100lfDemo('/mnt/lustre/syli/duet/demo_data/music', '/mnt/lustre/syli/duet/demo_data/motion', split='demo', interval=None, dtype='both')
    print(len(dd100.dances['pos3dl']), len(dd100.dances['rotmatl']), len(dd100.dances['music']), len(dd100.names))
    for ii in range(len(dd100.dances['pos3dl'])):
        # if len(dd100.dances['pos3d'][ii]) != len(dd100.dances['music'][ii]):
        print(len(dd100.dances['pos3dl'][ii]), len(dd100.dances['rotmatl'][ii]), len(dd100.dances['music'][ii]), dd100.names[ii], flush=True)
        # print(dd100.dances['rotmatl'][ii][55], dd100.dances['rotmatf'][ii][55])