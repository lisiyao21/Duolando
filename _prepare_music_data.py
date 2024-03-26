# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor




# parser = argparse.ArgumentParser()
# parser.add_argument('--input_video_dir', type=str, default='aist_plusplus_final/all_musics')
# parser.add_argument('--store_dir', type=str, default='data/aistpp_music_feat_7.5fps')
# parser.add_argument('--sampling_rate', type=int, default=15360*2/8)
# args = parser.parse_args()
# store_dir = args.store_dir
extractor = FeatureExtractor()

# if not os.path.exists(args.store_dir):
#     os.mkdir(args.store_dir)


def signal_to_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr)
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr,  octave=7 if sr==15360 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc, # 20
        mfcc_delta, # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
    ], axis=0)

            # mfcc, #20
            # mfcc_delta, #20

            # chroma_cqt, #12
            # onset_env, # 1
            # onset_beat, #1

    feature = feature.transpose(1, 0)
    print(f'acoustic feature -> {feature.shape}')

    return feature





if __name__ == '__main__':
    # musics, dances, fnames = make_music_dance_set(args.input_video_dir, args.input_annotation_dir) 

    # musics, dances, musics_raw = align(musics, dances)
    # save(args, musics, dances, fnames, musics_raw)
    
    mp3_root = './data/music/mp3'
    feature_root = './data/music/feature'
    os.makedirs(feature_root, exist_ok=True)
    # mp3_root = '/mnt/lustre/syli/duet/data/music/mp3'
    # feature_root = '/mnt/lustre/syli/duet/data/music/feature_bailando'
    sampling_rate = 15360

    for folder in os.listdir(mp3_root):
        print(folder)
        # if  folder != 'train' and folder != 'test':
        #     continue
        mp3_folder = os.path.join(mp3_root, folder)
        feature_folder = os.path.join(feature_root, folder)

            
        if not os.path.exists(mp3_folder):
            os.mkdir(mp3_folder)
        if not os.path.exists(feature_folder):
            os.mkdir(feature_folder)
        
        for mp3 in os.listdir(mp3_folder):
            if not mp3.endswith('.mp3'):
                continue

            print('Processing ', os.path.join(mp3_folder, mp3), flush=True)
            loader = essentia.standard.MonoLoader(filename=os.path.join(mp3_folder, mp3), sampleRate=sampling_rate)

            audio = loader()
            audio = np.array(audio).T

            feature = signal_to_feature(audio, sampling_rate)
            print(feature.shape)
            np.save(os.path.join(feature_folder, mp3[:-4]), feature)
