# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl' # use headless rendering

import time

import argparse

import cv2

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm

import trimesh
import pyrender
from mesh_intersection.bvh_search_tree import BVH


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        print(len(scene_or_mesh.geometry.values()))
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def numpy2set(arr):
    return set([tuple(x) for x in arr])


def detect_collision(mesh_fn, max_collisions, device, verbose=True):
    if isinstance(mesh_fn, str):
        print(f'Mesh file: {mesh_fn}')
        input_mesh = trimesh.load(mesh_fn)
        input_mesh = as_mesh(input_mesh)
    elif isinstance(mesh_fn, trimesh.base.Trimesh):
        input_mesh = mesh_fn
    else:
        raise TypeError

    if verbose:
        print('Number of triangles = ', input_mesh.faces.shape[0])

    vertices = torch.tensor(input_mesh.vertices,
                            dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=max_collisions)

    torch.cuda.synchronize()
    start = time.time()
    outputs = m(triangles)
    torch.cuda.synchronize()
    if verbose:
        print(f'Elapsed time: {(time.time() - start) * 1000} ms')

    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]

    if verbose:
        print('Number of collisions = ', collisions.shape[0])
        print('Percentage of collisions (%)',
            collisions.shape[0] / float(triangles.shape[1]) * 100)

    return collisions, input_mesh


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--fn1', type=str,
                        help='A mesh file to be checked for self-collisions')
    parser.add_argument('--fn2', type=str,
                        help='A mesh file to be checked for self-collisions')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')

    args, _ = parser.parse_known_args()

    max_collisions = args.max_collisions

    col1, input_mesh1 = detect_collision(args.fn1, max_collisions)
    col2, input_mesh2 = detect_collision(args.fn2, max_collisions)

    input_mesh = trimesh.util.concatenate([input_mesh1, input_mesh2])
    col12, input_mesh = detect_collision(input_mesh, max_collisions)

    col2 += input_mesh1.faces.shape[0]
    self_collisions = np.concatenate([col1, col2])

    self_col = numpy2set(self_collisions)
    all_col = numpy2set(col12)

    collisions = np.array(list(all_col - self_col))
    n_collisions = len(collisions)
    assert n_collisions == (len(all_col) - len(col1) - len(col2))

    recv_faces = input_mesh.faces[collisions[:, 0]]
    intr_faces = input_mesh.faces[collisions[:, 1]]

    print('[All] Number of collisions = ', collisions.shape[0])
    print('[All] Percentage of collisions (%)',
          collisions.shape[0] / float(input_mesh.faces.shape[0]) * 100)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 0.99])
    recv_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.0, 0.9, 0.0, 1.0])
    intr_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.9, 0.0, 0.0, 1.0])

    main_mesh = pyrender.Mesh.from_trimesh(input_mesh, material=material)

    recv_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(input_mesh.vertices, recv_faces),
        material=recv_material)
    intr_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(input_mesh.vertices, intr_faces),
        material=intr_material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(main_mesh)
    scene.add(recv_mesh)
    scene.add(intr_mesh)

    pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)