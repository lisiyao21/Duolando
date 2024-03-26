# Duolando
Code for ICLR 2024 paper "Duolando: Follower GPT with Off-Policy Reinforcement Learning for Dance Accompaniment."

[[Paper]] | [[Project Page]] |  [[Video Demo]]

✨ Please do not hesitate to give a star! ✨

<!-- <p float="left">
	<img src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif1.gif" width="150" /> <img src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif2.gif" width="360" /> <img width="280" src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif3.gif"/>
	</p> -->

> We introduce a novel task within the field of human motion generation, termed dance accompaniment, which necessitates the generation of responsive movements from a dance partner, the "follower", synchronized with the lead dancer’s movements and the underlying musical rhythm. Unlike existing solo or group dance generation tasks, a duet dance scenario entails a heightened degree of interaction between the two participants, requiring delicate coordination in both pose and position. To support this task, we first build a large-scale and diverse duet interactive dance dataset, **DD100**, by recording about 117 minutes of professional dancers’ performances. To address the challenges inherent in this task, we propose a GPT based model, **Duolando**, which autoregressively predicts the subsequent tokenized motion conditioned on the coordinated information of the music, the leader’s and the follower’s movements. To further enhance the GPT’s capabilities of generating stable results on unseen conditions (music and leader motions), we devise an off-policy reinforcement learning strategy that allows the model to explore viable trajectories from out-of-distribution samplings, guided by human-defined rewards. Based on the collected dataset and proposed method, we establish a benchmark with several carefully designed metrics.

# Code

## Environment setup
    conda create -n duet python=3.8
    conda activate duet
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
    pip install -r requirement.txt


Besides, thie project needs the SMPLX models, please download (after register) from [here](https://smpl-x.is.tue.mpg.de/). Please decompress it to a specific path. If you also wish to visualize the SMPLX models, please download Blender from [here](https://www.blender.org/download/) (we use version 3.4.1) and decompress it to './software'; please download SMPLX Add-on for Blender from [here](https://github.com/Meshcapade/SMPL_blender_addon) as a zip file.


## DD100 Data 

In this work, we collect a duet dance dataset, named DD100. Please visit [here] to download it and decompress to the ./data folder. The dataset will be look like

    data
      |-- motion
      |     |_smplx
      |          |_all
      |             |-- [GENRE]_[CHOREOGRAPH]_[TAKE]_00.npy
      |             |-- [GENRE]_[CHOREOGRAPH]_[TAKE]_01.npy
      |             ... 
      |
      |__ music
            |_mp3
               |_all
                  |-- [GENRE]_[CHOREOGRAPH]_[TAKE].mp3
                  ...

Here, [GENRE] is the type of dance, like Ballet, Waltz, Tango, ... [CHOREOGRAPH] is the choreography index amd [TAKE] represent the take index. For 67 dance sequences we recorded the movement twice. Since the details of the movement may change, we keep them in two different takes. The xxx_00.npy and xxx_01.npy are the SMPLX sequences of the lady (follower dacer) and the the man (leader), respectively.

In our experiment, we need first transfer SMPLX format into ratation matrix and 3d positions of the joints, and transfer the .mp3 music to specific features. Run the following scripts to do so. Before that, please change the 'model_path' in Line 19 of _prepare_motion_data.py to path/to/downloaded/smplx/models.
    
    python _train_test_split.py
    python _prepare_motion_data.py
    python _prepre_music_data.py

Then you will see the split of 'train' and 'test' folder in subfolders under 'motion' and 'music', and see the extracted features. The 'pos3d' sequences are in shape of Tx55x3, where T is the frame number, 55 is the SMPLX joint number and 3 is (x,y,z) dimensions, while for 'rotmat', they are Tx55x9, where 9 is the 3x3 rotation matrix transformed from the axis angles of SMPLX.

If you want to visualize these sequences, please run

    bash visualize_01.sh 

Before that, please change Line 12 of tools/vis/vis_smplx_w_blender_01.py to path/to/smplx/add-on/zip, change Line 4 of visualize_01.sh to path/to/blender.

If you are not willing to do the preprocessing by yourself, you can directly download our preprocessed feature from [here] into ./data folder.

## Training

The training of Duolando comprises of 4 steps in the following sequence. If you are using the slurm workload manager, you can directly run the corresponding shell. Otherwise, please remove the 'srun' parts in 'srun_xxx.sh'. A kind note here: if you do not have multiple gpus, you may need to lower the batch size in related config when training follower GPT and reinforcement learning part.


### Step 1: Train pose motion VQ-VAE

    bash srun_mix.sh configs/sep_vqvaexm_full_final.yaml train [node name] 1

### Step 2: Train translation VQ-VAE
    bash srun_transl.sh configs/transl_vqvaex_final.yaml train [node name] 1

### Step 3: Train motion GPT

    bash srun_gpt2t.sh configs/follower_gpt_beta0.9_final.yaml train [node name] 8

### Step 4: Actor-Critic finetuning on target music 

    bash srun_rl_new_.sh configs/rl_final_debug_reward3_random_5mem_lr3e-5.yaml train [node name] 8

## Evaluation

To test with our pretrained models, please download the weights from [here] (Google Drive) or [here] (OneDrive0 into ./experiments folder.

### 1. Generate dancing results

To test follower GPT:

    bash srun_gpt2t.sh configs/follower_gpt_beta0.9_final.yaml eval [your node name] 1
   
To test follower GPT w. RL:
    
    bash srun_rl_new_.sh configs/rl_final_debug_reward3_random_5mem_lr3e-5.yaml eval [node name] 1

### 2. Dance quality evaluations

a. Solo metrics

Run 

    python utils/metrics_new.py

To fasten the computation, comment Line 184 of utils/metrics_new.py after computed the ground-truth feature once. To test another folder, change Line 182 to your destination,

b. Interactive metrics

    python utils/metrics_duet.py

c. Contact frequency

This metric is computed based on [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect). We first inflate the smplx models vertices toward their normals by 1cm and then compute the intersection between the meshes. To do so, please first build it

    cd utils/torch-mesh-isect
    python setup.py install

Then, run 

    python examples/compute_contact_freq.py

Before that, please change Line 80 and 81 to path/to/downloaded/smplx/models. This computation is slow.

The values should be exactly the same as reported in the paper. 

## Accompaniment in the wild

TBD

I wish you could enjoy it. 

### Citation

    @inproceedings{siyao2022bailando,
	    title={Bailando: 3D dance generation via Actor-Critic GPT with Choreographic Memory,
	    author={Siyao, Li and Yu, Weijiang and Gu, Tianpei and Lin, Chunze and Wang, Quan and Qian, Chen and Loy, Chen Change and Liu, Ziwei },
	    booktitle={CVPR},
	    year={2022}
    }

### License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/lisiyao21/Bailando/blob/main/LICENSE). Redistribution and use should follow this license.

