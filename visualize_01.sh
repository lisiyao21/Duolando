#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PATH:/mnt/sfs-common/syli/duet_final/software/blender-3.4.1-linux-x64/

echo $LD_LIBRARY_PATH

a1='/mnt/sfs-common/syli/duet_final/github/Duelando/data/motion/smplx/all'
a2='/mnt/sfs-common/syli/duet_final/github/Duelando/data/motion/smplx/all_video/'

mkdir -p $a2


        export a1
        export a2
        
 ls $a1 | grep _00.npy | xargs -I {} -P 20 bash -c 'file="{}";
        prefix=$a1/"${file%_00.npy}";
        echo $prefix;
        save=$a2/"${file%_00.npy}";
        if [ -e "${prefix}_01.npy" ]; then
            echo "Processing ${prefix}_00.npy and ${prefix}_01.npy...";
            srun -p gpu --cpus-per-task=1  blender -b -P tools/vis/vis_smplx_w_blender_01.py -- --npy0_path ${prefix}_00.npy --npy1_path ${prefix}_01.npy --output_video_path ${save}.mp4;
        fi
        '
# python add_frame_number.py -- ${save}.mp4 ${save}_with_franem_number.mp4;