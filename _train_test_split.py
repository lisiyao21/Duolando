import numpy as np 
import os

import shutil

roots = ['./data/motion/smplx', './data/music/mp3']

for root in roots:
    source_root = root + '/all'
    test_root = root + '/test'
    train_root = root + '/train'

    if not os.path.exists(test_root):
        os.mkdir(test_root)
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    for ff in os.listdir(source_root):
        if not ff.endswith('npy') and not ff.endswith('mp3'):
            continue
        
        index = int(ff.split('_')[1])
        if 'Ballet' not in ff:
            if index % 5 == 0:
                shutil.copyfile(os.path.join(source_root, ff), os.path.join(test_root, ff))
            else:
                shutil.copyfile(os.path.join(source_root, ff), os.path.join(train_root, ff))
        else:
            if index == 9 or index == 10:
                shutil.copyfile(os.path.join(source_root, ff), os.path.join(test_root, ff))
            else:
                shutil.copyfile(os.path.join(source_root, ff), os.path.join(train_root, ff))