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

def cross_penetration(posef, posel):
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
    cross_pen = np.sqrt(np.sum((posef[:, :, None, :] - posel[:, None, :, :])**2, axis=-1)).reshape(T, -1)
    cross_pen = np.mean((np.min(cross_pen, axis=-1) < 0.01).astype(float))

    return cross_pen


def quantized_metrics(predicted_pkl_root, gt_pkl_root):


    pred_features = []
    gt_freatures = []



    # for pkl in os.listdir(predicted_pkl_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))

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

    # print(kps_gen.shape)
    # print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

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
        # joint3d24 = np.zeros([joint3d.shape[0], 24, 3])
        # joint3d24[:, :22] = joint3d[:, :22]
        # joint3d24[:, 22] = (joint3d[:, 25] + joint3d[:, 28] + joint3d[:, 31] + joint3d[:, 34] + joint3d[:, 37])/5.0
        # joint3d24[:, 23] = (joint3d[:, 40] + joint3d[:, 43] + joint3d[:, 46] + joint3d[:, 49] + joint3d[:, 52])/5.0
        # joint3d = joint3d24.copy().reshape([-1, 72])
        # # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # # print(roott)
        # joint3d = joint3d - np.tile(roott, (1, 24))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # np_dance[:, :3] = root
        np.save(os.path.join(root, 'duet_features', pkl), duet_feature(joint3df, joint3dl))

        pen_rates.append(cross_penetration(joint3df, joint3dl))
    print('CFP: ', np.mean(pen_rates))
        # np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))




# def get_mb(music_root, key, length=None):
#     path = os.path.join(music_root, key)
#     with open(path) as f:
#         #print(path)
#         # sample_dict = json.loads(f.read())
#         if length is not None:
#             beats = np.load(path)[:, 53][:][:length]
#         else:
#             beats = np.load(path)[:, 53]


#         beats = beats.astype(bool)
#         beat_axis = np.arange(len(beats))
#         beat_axis = beat_axis[beats]
        
#         # fig, ax = plt.subplots()
#         # ax.set_xticks(beat_axis, minor=True)
#         # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
#         # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
#         # ax.xaxis.grid(True, which='minor')


#         # print(len(beats))
#         return beat_axis


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


    gt_root = '/mnt/sfs-common/syli/duet_final/data/motion/pos3d/all'
    music_root = '/mnt/sfs-common/syli/duet_final/data/music2/feature/test'
    pred_roots = [
        '/mnt/sfs-common/syli/duet_final/Duelando/experiments/rl_final_debug_reward3_lr1e-5_random_5mem_3e-5/eval/npy/pos3d/ep0050',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/rl_final_debug_reward3_lr1e-5_random_5mem_3e-5/eval/npy/pos3d/ep0040',
        '/mnt/sfs-common/syli/duet_final/data/motion/pos3d/test',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/gpt1_final/eval/npy/pos3d/ep0250',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/gpt2_final/eval/npy/pos3d/ep0120',
        '/mnt/sfs-common/syli/duet_final/Duelando/experiments/follower_gpt_full_bsz128_transl_beta0.9_final/eval/npy/pos3d/ep0250',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/rl_final/eval/npy/pos3d/ep0050',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/rl_final/eval_normal/npy/pos3d/ep0050',
        # '/mnt/sfs-common/syli/duet_final/Duelando/experiments/rl_final/eval_normal/npy/pos3d/ep0010'
    ]
    # print(calc_duet_ba_score('/mnt/lustre/syli/duet/Duelando/experiments/fgptn_9t_full_bsz128_transl_beta0.9_tp_trytry/eval/npy/pos3d/ep0500'), flush=True)
    # pred_root = '/mnt/lustre/syli/dance/Bailando/experiments/cc_gpt_music_trans/eval/pkl/ep000500'
    # pred_roots = '/mnt/lustre/syli/dance/Bailando/experiments/cc_motion_gpt_again/vis/pkl'
    # calc_and_save_feats_duet(gt_root)
    for pred_root in pred_roots:
        print(pred_root, flush=True)
        pen_rate = calc_and_save_feats_duet(pred_root)


    # print('Calculating metrics')
    # print(gt_root)
    # print(pred_root)
        print(calc_duet_be_score(pred_root))
        print(quantized_metrics(pred_root, gt_root), flush=True)