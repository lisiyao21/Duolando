import os
import glob
import numpy as np
# from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d


def detect_footskating_dur(
        keypoints3d,
        pelvis_idx=0,
        left_ankle_idx=7,
        right_ankle_idx=8,
        pelvis_threshold=0.03,  # m/s
        leg_velocity_threshold=0.01,  # m/s
):
    pelvis_coord = keypoints3d[:, pelvis_idx, :]
    left_ankle_coord = keypoints3d[:, left_ankle_idx, :]
    right_ankle_coord = keypoints3d[:, right_ankle_idx, :]
    # pelvis_coord[:, 0] = 0
    # pelvis_velocity
    
    # print('pelvis_velocity', pelvis_velocity.mean(), pelvis_velocity.std())

    # compute left-right relative velocity
    leg_distance1 = (left_ankle_coord - pelvis_coord).copy()
    leg_distance2 = (right_ankle_coord - pelvis_coord).copy()

    pelvis_coord[:, 2] = 0
    pelvis_velocity = np.linalg.norm(np.diff(pelvis_coord, axis=0), axis=1)
    
    leg_velocity1 = np.linalg.norm(np.diff(leg_distance1, axis=0), axis=1)
    leg_velocity2 = np.linalg.norm(np.diff(leg_distance2, axis=0), axis=1)
    # print('leg_velocity', leg_velocity.mean(), leg_velocity.std())

    n_frame = len(pelvis_velocity)
    footskating_dur = 0

    for i in range(1, n_frame):
        if pelvis_velocity[i] > pelvis_threshold and leg_velocity1[
                i] < leg_velocity_threshold and leg_velocity2[i] < leg_velocity_threshold:
            footskating_dur += 1
            # print(i)

    return footskating_dur, n_frame


if __name__ == '__main__':

    pred_roots = [
        'experiments/rl/eval/npy/pos3d/ep0050',
    ]
    for root in pred_roots:

        pelvis_threshold = 0.03  # m/s
        leg_velocity_threshold = 0.01  # m/s

        tot_dur = 0
        tot_frame = 0
        path_lst = glob.glob(os.path.join(root, '*.npy'))
        for path in path_lst:
            data = np.load(path, allow_pickle=True)
            keypoints3d = data.reshape(-1, 55, 3)
            # visualize_kp3d(keypoints3d, output_path='out.mp4', data_source='smpl')

            dur, n_frame = detect_footskating_dur(
                keypoints3d,
                pelvis_threshold=pelvis_threshold,
                leg_velocity_threshold=leg_velocity_threshold)
            # print('footskating_duration:', dur)

            tot_dur += dur
            tot_frame += n_frame

        print(f'root folder: {root}')
        print(f'pelvis_threshold={pelvis_threshold}, leg_velocity_threshold={leg_velocity_threshold}')
        print(f'footskating_duration={tot_dur}')
        print(f'total_frames={tot_frame}')
        print(f'ratio={tot_dur / tot_frame * 100.:.4f} %')
