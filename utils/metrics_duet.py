import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import os 
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import json

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPLX_FEAT_POINT = [0, 7, 8, 10, 11, 15, 16, 17, 20, 21]


SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', 
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', 
    'left_foot', # 10
    'right_foot', # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]

def duet_feature(posef, posel):
    """
        posef: Tx55x3
        posel: Tx55x3
    """
    # Tx10x3
    Tf, _, _ = posef.shape
    Tl, _, _ = posel.shape
    T = np.min([Tf, Tl])
    posef = posef.copy()[:T, :]
    posel = posel.copy()[:T, :]
    feat = np.sqrt(np.sum((posef[:, SMPLX_FEAT_POINT][:, :, None, :] - posel[:, SMPLX_FEAT_POINT][:, None, :, :])**2, axis=-1)).reshape(T, -1)
    feat = np.mean(feat, axis=0)

    return feat




def quantized_metrics(predicted_pkl_root, gt_pkl_root):


    pred_features = []
    gt_freatures = []


    pred_features = [np.load(os.path.join(predicted_pkl_root, 'duet_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'duet_features'))]
    gt_freatures = [np.load(os.path.join(gt_pkl_root, 'duet_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'duet_features'))]
    
    
    pred_features = np.stack(pred_features)  # Nx72 p40
    gt_freatures = np.stack(gt_freatures) # N' x 72 N' >> N
        
    gt_freatures, pred_features = normalize(gt_freatures, pred_features)

    print('Calculating metrics')

    fid = calc_fid(pred_features, gt_freatures)
    div_gt = calculate_avg_distance(gt_freatures)
    div = calculate_avg_distance(pred_features)

    metrics = {'fid_k': fid, 'div' : div, 'div_gt': div_gt}
    return metrics


def calc_fid(kps_gen, kps_gt):


    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats_duet(root):
    if not os.path.exists(os.path.join(root, 'duet_features')):
        os.mkdir(os.path.join(root, 'duet_features'))
    # if not os.path.exists(os.path.join(root, 'manual_features_new')):
    #     os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []
    pen_rates = []

    for pkl in os.listdir(root):
        if not pkl.endswith('_00.npy'):
            continue
        pkll = pkl.replace('_00.npy', '_01.npy')
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        if not os.path.exists(os.path.join(root, pkll)):
            continue
        
        # print(joint3d.shape)
        joint3df = np.load(os.path.join(root, pkl)).reshape([-1, 55, 3])
        joint3dl = np.load(os.path.join(root, pkll)).reshape([-1, 55, 3])

        np.save(os.path.join(root, 'duet_features', pkl), duet_feature(joint3df, joint3dl))






def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 55, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    # print(kinetic_vel.shape)
    kinetic_vel = G(kinetic_vel, 5)
    # print(kinetic_vel.shape)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats[0]:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats[0]))


def calc_duet_be_score(root):

    # gt_list = []
    ba_scores = []

    for pklf in os.listdir(root):
        if not pklf.endswith('_00.npy'):
            continue
        pkll = pklf.replace('_00.npy', '_01.npy')
        # print(pkl)
        if os.path.isdir(os.path.join(root, pklf)):
            continue
        joint3df = np.load(os.path.join(root, pklf))
        joint3dl = np.load(os.path.join(root, pkll))

        # print(joint3df.shape, joint3dl.shape, flush=True)

        dance_beatsf, lengthf = calc_db(joint3df)  
        dance_beatsl, lengthl = calc_db(joint3dl) 
        # print(dance_beatsf.shape)       
        
        ba_scores.append(BA(dance_beatsl, dance_beatsf))
        
    return np.mean(ba_scores)






if __name__ == '__main__':


    gt_root = 'data/motion/pos3d/all'
    music_root = 'data/music/feature/test'
    pred_roots = [
        'experiments/rl/eval/npy/pos3d/ep0050',
    ]

    # run this line for only once
    calc_and_save_feats_duet(gt_root)
    for pred_root in pred_roots:
        print(pred_root, flush=True)
        calc_and_save_feats_duet(pred_root)
        print(calc_duet_be_score(pred_root))
        print(quantized_metrics(pred_root, gt_root), flush=True)