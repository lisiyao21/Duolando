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


class DD100M(Dataset):
    def __init__(self, data_root, split='train', interval=None, dtype='pos3d', move=8):
        self.dances = {'rotmat':[], 'pos3d':[]}
        dtypes = ['rotmat', 'pos3d']
        self.dtype = dtype

        self.names = []

        for dtype_folder in dtypes:
            for fname in os.listdir(os.path.join(data_root, dtype_folder, split)):
                path = os.path.join(data_root, dtype_folder, split, fname)
                np_dance = np.load(path)

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
                if (interval is not None) and ( interval != 'None' ):
                    seq_len, dim = np_dance.shape
                    for i in range(0, seq_len, move):

                        np_dance_sub_seq = np_dance[i: i + interval]
                        if len(np_dance_sub_seq) != interval:
                            continue
                        self.dances[dtype_folder].append(np_dance_sub_seq)
                        self.names.append(fname)
                else:
                    self.dances[dtype_folder].append(np_dance[:len(np_dance)//move*move])
                    self.names.append(fname)

    def __len__(self):
        return len(self.dances['pos3d'])

    def __getitem__(self, index):
        # print(self.dances[index].shape)
        return {'pos3d':self.dances['pos3d'][index], 'rotmat':self.dances['rotmat'][index], 'fname':self.names[index]}

if __name__ == '__main__':
    dd100m = DD100M('/mnt/lustre/syli/duet/data/motion', split='first_part', interval=240, dtype='both')
    aa  = dd100m[2]
    print(len(dd100m), aa, flush=True)