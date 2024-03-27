# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse

import numpy as np

from smplx import SMPLX
import torch
import torch
from scipy.spatial.transform import Rotation as R



def smplx_to_pos3d(data):
    smplx = None
    # print(data['betas'][0][:10].shape,flush=True)
    smplx = SMPLX(model_path='/mnt/sfs-common/syli/duet_final/smplx', betas=data['betas'][:, :10], gender=data['meta']['gender'], \
        batch_size=len(data['betas']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)
    data['poses'] = data['poses'].reshape(len(data['poses']), -1)
    print(data['betas'].shape, data['poses'].shape,data['global_orient'].shape, data['transl'].shape)
    keypoints3d = smplx.forward(
        global_orient=torch.from_numpy(data['global_orient']).float(),
        body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
        transl=torch.from_numpy(data['transl']).float(),
        betas=torch.from_numpy(data['betas'][:, :10]).float()
        ).joints.detach().numpy()[:, :55]

    nframes = keypoints3d.shape[0]
    return keypoints3d.reshape(nframes, -1)

def smplx_to_rotmat(data):
    smpl_poses, smpl_trans = data['poses'], data['transl']
        
    nframes = smpl_poses.shape[0]
    njoints = 55

    r = R.from_rotvec(smpl_poses.reshape([nframes*njoints, 3])) 
    rotmat = r.as_matrix().reshape([nframes, njoints, 3, 3])

    rotmat = np.concatenate([
        smpl_trans,
        rotmat.reshape([nframes, njoints * 3 * 3])
    ], axis=-1)
    nframes = rotmat.shape[0]
    return rotmat.reshape(nframes, -1)


if __name__ == '__main__':

    motion_root =  '/mnt/sfs-common/syli/duet_final/demo_data/motion'
    smpl_root =    '/mnt/sfs-common/syli/duet_final/demo_data/motion/smplx'
    pos3d_root =   '/mnt/sfs-common/syli/duet_final/demo_data/motion/pos3d'
    rotmat_root =  '/mnt/sfs-common/syli/duet_final/demo_data/motion/rotmat'

    os.makedirs(pos3d_root, exist_ok=True)
    os.makedirs(rotmat_root, exist_ok=True)

    for folder in os.listdir(smpl_root):
        print(folder)

        smplx_folder = os.path.join(smpl_root, folder)
        pos3d_folder = os.path.join(pos3d_root, folder)
        rotmat_folder = os.path.join(rotmat_root, folder)

        if not os.path.exists(pos3d_folder):
            os.mkdir(pos3d_folder)
        if not os.path.exists(rotmat_folder):
            os.mkdir(rotmat_folder)
        
        for smplx_file in os.listdir(smplx_folder):
            if not smplx_file.endswith('.npy'):
                print(smplx_file)
                continue

            data = np.load(os.path.join(smplx_folder, smplx_file), allow_pickle=True, encoding='bytes').item()

            keypoints3d = smplx_to_pos3d(data)
            rotmat = smplx_to_rotmat(data)
            
            print(rotmat.shape, flush=True)

            np.save(os.path.join(pos3d_folder, smplx_file), keypoints3d)
            np.save(os.path.join(rotmat_folder, smplx_file), rotmat)




