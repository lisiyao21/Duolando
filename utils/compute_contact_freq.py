from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import time

import argparse

import torch

import numpy as np
from tqdm import tqdm

import trimesh
import smplx
from detect_inter_collisions import numpy2set, detect_collision


import numpy as np

def compute_vertex_normals_efficient(vertices, faces):
    
    face_vertices = vertices[:, faces.flatten(), :].reshape(vertices.shape[0], faces.shape[0], 3, 3)

    v1, v2, v3 = np.split(face_vertices, 3, axis=2)
    edge1 = v2 - v1
    edge2 = v3 - v1

    edge1 = edge1.squeeze(axis=2)  
    edge2 = edge2.squeeze(axis=2)  
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.linalg.norm(face_normals, axis=2, keepdims=True) + 1e-5

    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for j in range(3):  
            vertex_normals[:, face[j], :] += face_normals[:, i, :]

    vertex_normals /= np.linalg.norm(vertex_normals, axis=2, keepdims=True) + 1e-5
    return vertex_normals

def inflate_mesh_efficient(vertices, vertex_normals, distance=0.01):
    
    inflated_vertices = vertices + vertex_normals * distance
    return inflated_vertices




def get_smplx_mesh(fn):
    # read smplx params from file
    data = np.load(fn, allow_pickle=True).item()
    transl = torch.from_numpy(data['transl']).float()

    betas = torch.from_numpy(data['betas']).float() if 'betas' in data else None
    global_orient = data['global_orient']
    if len(global_orient.shape) == 3:
        global_orient = global_orient[:, 0]
    poses = data['poses']
    gender = data['meta']['gender']

    nframe = poses.shape[0]
    #  = betas.shape

    
    poses = torch.from_numpy(poses).float().reshape(-1, 55, 3)
    global_orient = poses[:, 0, :3]

    # global_orient = torch.from_numpy(global_orient).float()
    body_pose = poses[:, 1:22, :]
    jaw_pose = poses[:, 22:23, :]
    leye_pose = poses[:, 23:24, :]
    reye_pose = poses[:, 24:25, :]
    left_hand_pose = poses[:, 25:40, :]
    right_hand_pose = poses[:, 40:55, :]
    expression = torch.zeros(nframe, 10)

    # create smplx body model
    model_folder = '/mnt/sfs-common/syli/duet_final/'
    model_type = 'smplx'

    if betas != None:
        num_betas = betas.shape[1]
        model = smplx.create(model_folder,
                            model_type=model_type,
                            gender=gender,
                            use_face_contour=True,
                            use_pca=False,
                            num_betas=num_betas,
                            num_expression_coeffs=10,
                            ext='npz')
    else:
        model = smplx.create(model_folder,
                            model_type=model_type,
                            gender=gender,
                            use_face_contour=True,
                            use_pca=False,
                            num_betas=10,
                            num_expression_coeffs=10,
                            ext='npz')

    if betas != None:
        output = model(transl=transl,
                    betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    return_verts=True)
    else:
        output = model(transl=transl,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    return vertices, model.faces


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        help='bla')
    # parser.add_argument('--fn2',
    #                     type=str,
    #                     help='A mesh file to be checked for self-collisions')
    # parser.add_argument('--ofn',
    #                     type=str,
    #                     help='A npz file to save the collision results')

    # pred_roots = [
    #     '/mnt/lustre/syli/duet/Bailando/experiments/fgptn_9t_half_bsz256_transl_bailando/eval/npy/pos3d/ep0330',
    #     '/mnt/lustre/syli/duet/Duelando/experiments/ac_new_full_tp_again2/eval/npy/pos3d/ep0050',
    #     '/mnt/lustre/syli/duet/data/motion/pos3d/test4metric',
    #     '/mnt/lustre/syli/duet/Duelando/experiments/fgptn_9t_full_bsz128_transl_beta0.9_tp_trytry/eval/npy/pos3d/ep0500',
    #     '/mnt/lustre/syli/duet/Duelando/experiments/fgpt_full/eval/npy/pos3d/ep0260',
    #     '/mnt/lustre/syli/duet/Duelando/experiments/gpt_full/eval/npy/pos3d/ep0500'
    # ]


    parser.add_argument('--max_collisions',
                        default=20,
                        type=int,
                        help='The maximum number of bounding box collisions')

    args, _ = parser.parse_known_args()

    max_collisions = args.max_collisions
    root = args.root

    tot_frame = 0
    col_frame = 0
    for pkl in os.listdir(args.root ):
        if not pkl.endswith('_00.npy'):
            continue
        pkll = pkl.replace('_00.npy', '_01.npy')
        # print(pkl, flush=True)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        if not os.path.exists(os.path.join(root, pkll)):
            continue

        vertices1, faces1 = get_smplx_mesh(os.path.join(root, pkl))
        vertices2, faces2 = get_smplx_mesh(os.path.join(root, pkll))
        # print(vertices1.shape, vertices1.min(), vertices1.types)
        # print(vertices1.shape, vertices2.shape, flush=True)
        
        normals1 = compute_vertex_normals_efficient(vertices1, faces1.copy()).astype(float)
        # print(normals1[0])
        vertices1 = inflate_mesh_efficient(vertices1, normals1)
        # print(normals1.shape)
        normals2 = compute_vertex_normals_efficient(vertices2, faces2.copy()).astype(float)
        vertices2 = inflate_mesh_efficient(vertices2, normals2)
        # print(vertices1.shape, vertices2.shape,  flush=True)


        # assert vertices1.shape == vertices2.shape
        # assert (faces1 == faces2).all()
        # del faces2

        nframe = min(vertices1.shape[0], vertices2.shape[0])
        tot_frame += nframe
        col_lst = {'frame': [], 'n_col': [], 'r_col': []}

        t0 = time.time()
        for i in tqdm(range(nframe)):
            input_mesh1 = trimesh.Trimesh(vertices1[i, :], faces1)
            input_mesh2 = trimesh.Trimesh(vertices2[i, :], faces2)
            col1, _ = detect_collision(input_mesh1,
                                    max_collisions,
                                    device,
                                    verbose=False)
            col2, _ = detect_collision(input_mesh2,
                                    max_collisions,
                                    device,
                                    verbose=False)

            input_mesh = trimesh.util.concatenate([input_mesh1, input_mesh2])
            col12, input_mesh = detect_collision(input_mesh,
                                                max_collisions,
                                                device,
                                                verbose=False)

            col2 += input_mesh1.faces.shape[0]
            self_collisions = np.concatenate([col1, col2])

            self_col = numpy2set(self_collisions)
            all_col = numpy2set(col12)

            collisions = np.array(list(all_col - self_col))
            n_collisions = len(collisions)
            # print(n_collisions, len(all_col), len(col1), len(col2), flush=True)
            assert n_collisions == (len(all_col) - len(col1) - len(col2))

            if n_collisions > 0:
                ratio = n_collisions / float(input_mesh.faces.shape[0]) * 100
                col_lst['frame'].append(i)
                col_lst['n_col'].append(n_collisions)
                col_lst['r_col'].append(ratio)
            # print(f'[{i}] Number of collisions = {n_collisions}')
            # print(f'[{i}] Percentage of collisions (%): {ratio}')
        col_frame += len(col_lst['frame'])
    t1 = time.time()
    nframe = tot_frame
    n = col_frame
    # print(f'Process {nframe} frames: {t1 - t0} s')
    print('\n')
    print(f'{n} frames from {nframe} frames have collision between two person', flush=True)
   
    print(f'avg collision ratio of {n} frames is: {n*1.0/nframe*100:.4f}%')