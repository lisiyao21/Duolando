## visualize skeleton of two agents


import numpy as np
import cv2
import os 
from os.path import isfile, join
from os import listdir
from moviepy.editor import *


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



# Define edges to connect joints
edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [9, 13],
    [9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 25], [25, 26], [26, 27],
    [20, 28], [28, 29], [29, 30],
    [20, 31], [20, 32], [20, 33],
    [20, 34], [20, 35], [20, 36],
    [20, 37], [20, 38], [20, 39],

    [21, 40], [40, 41], [41, 42],
    [21, 43], [21, 44], [21, 45],
    [21, 46], [46, 47], [47, 48],
    [21, 49], [49, 50], [50, 51],
    [21, 52], [52, 53], [53, 54]
]

# Create figure and scene
def visualize(joints, config, evaldir, dance_names, epoch_tested, quants):
    save_folder = os.path.join(evaldir, 'videos', 'ep'+str(epoch_tested).zfill(4))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    # joints = joints.reshape((len(joints), 55, 3))

    # Define colors
    colors = [(255, 80, 80)]

    for ii in range(len(joints)):
        this_joints = joints[ii][0]
        this_joints = this_joints.reshape(len(this_joints), 55, 3)
        # Define video settings
        video_name = os.path.join(save_folder, dance_names[ii]+'.mp4')
        frame_rate = 30.0
        frame_size = (config.width, config.height)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

        # Generate frames
        for i in range(len(this_joints)):
            # Create blank frame
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Draw edges
            for j in range(len(edges)):
                pt1 = tuple((this_joints[i][edges[j][0]] * 100).astype(int))
                pt2 = tuple((this_joints[i][edges[j][1]] * 100).astype(int))
                color = colors[0]
                
                cv2.line(frame, [pt1[0] + 480, -pt1[2] + 270], [pt2[0] + 480, -pt2[2] + 270], color, thickness=2)
            
            if quants is not None:
                # print(quants)
                # print(len(quants), flush=True)
                cv2.putText(frame, str(tuple(quants[dance_names[ii]][jj][i//4] for jj in range(len(quants[dance_names[ii]])))), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
           
            # Write frame to video
            out.write(frame)
        out.release()

        # music_source = config.music_source
        # if '_' in dance_names[ii]:
        #     folder = dance_names[ii].split('_')[0]
        #     mp3 = dance_names[ii].split('_')[1]
        #     music_name = os.path.join(music_source, folder, mp3[:-2] + '0' + mp3[-2:] + '.mp3')
        #     print(music_name,flush=True)
        #     if os.path.exists(music_name):
        #         video = VideoFileClip(video_name)
        #         audio = AudioFileClip(music_name)
        #         # bg_music = audio.set_duration(video.duration)
        #         video_with_music = video.set_audio(audio)
        #         video_with_music.write_videofile(video_name[:-4] + '_music.mp4', codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)

def visualize2(joints, joints_leader, config, evaldir, dance_names, epoch_tested, quants):
    save_folder = os.path.join(evaldir, 'videos', 'ep'+str(epoch_tested).zfill(4))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    # joints = joints.reshape((len(joints), 55, 3))

    # Define colors
    colors = [(80, 80, 255)]
    colors_l = [(255, 80, 80)] 

    print(len(joints), len(joints_leader), flush=True)
    len_joints = min(len(joints), len(joints_leader))

    for ii in range(len_joints):
        this_joints = joints[ii][0]
        this_joints_leader = joints_leader[ii][0]
        this_joints_leader = this_joints_leader.reshape(len(this_joints_leader), 55, 3)
        this_joints = this_joints.reshape(len(this_joints), 55, 3)
        # print(len(joints), len(this_joints), ' fdsafdsafdsa', flush=True)
        # Define video settings
        video_name = os.path.join(save_folder, dance_names[ii]+'.mp4')
        frame_rate = 30.0
        frame_size = (config.width, config.height)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

        # Generate frames
        for i in range(len(this_joints)):
            # Create blank frame
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Draw edges
            for j in range(len(edges)):
                pt1 = tuple((this_joints[i][edges[j][0]] * 100).astype(int))
                pt2 = tuple((this_joints[i][edges[j][1]] * 100).astype(int))
                color = colors[0]
                cv2.line(frame, [pt1[1] + 480, -pt1[2] + 270], [pt2[1] + 480, -pt2[2] + 270], color, thickness=2)

            for j in range(len(edges)):
                pt1 = tuple((this_joints_leader[i][edges[j][0]] * 100).astype(int))
                pt2 = tuple((this_joints_leader[i][edges[j][1]] * 100).astype(int))
                color = colors_l[0]
                cv2.line(frame, [pt1[1] + 480, -pt1[2] + 270], [pt2[1] + 480, -pt2[2] + 270], color, thickness=2)

            # Write frame to video
            # quant = quants[ii]
            if quants is not None:
                cv2.putText(frame, str(tuple(quants[ii][jj][i//4] for jj in range(len(quants[ii])))), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            out.write(frame)
        out.release()

        # if find music
        music_source = config.music_source
        if '2023' not in dance_names[ii]:
            if '_' in dance_names[ii]:
                mp3 = dance_names[ii].split('_')[0] + '_' + dance_names[ii].split('_')[1] + '_' +  dance_names[ii].split('_')[2]
                music_name = os.path.join(music_source, mp3 + '.mp3')
            else:
                music_name = os.path.join(music_source, dance_names[ii] + '.mp3')
        else:
            folder = dance_names[ii].split('_')[0]
            mp3 = dance_names[ii].split('_')[1]
            music_name = os.path.join(music_source, folder, mp3[:-2] + '0' + mp3[-2:] + '.mp3')
        
        print(music_name, flush=True)
        if os.path.exists(music_name):
            video = VideoFileClip(video_name)
            audio = AudioFileClip(music_name)
            # bg_music = audio.set_duration(video.duration)
            video_with_music = video.set_audio(audio)
            video_with_music.write_videofile(video_name[:-4] + '_music.mp4', codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)

def save_rot(smplxs, smplxs_leader, transl, transl_leader, config, evaldir, dance_names, epoch_tested):
    """
        input: 
            smplxs: list of [1xTx55*9]
            smplxs_leader: list of [1xTx55x9] 
    """
    for smplxf, smplxl in zip(smplxs, smplxs_leader):
        smplxf, smplxl = smplxf[0], smplxl[0]

    

def visualize_smplx(joints, joints_leader, config, evaldir, dance_names, epoch_tested, quants):
    pass

