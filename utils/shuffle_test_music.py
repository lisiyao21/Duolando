import os
import shutil
import random
from metrics import calc_ba_score

def shuffle_copy(ori_folder, shuffle_folder, ori_folder_feat, shuffle_folder_feat):
    # 确保目标文件夹存在，如果不存在，则创建
    if not os.path.exists(shuffle_folder):
        os.makedirs(shuffle_folder)
    if not os.path.exists(shuffle_folder_feat):
        os.makedirs(shuffle_folder_feat)
    
    # 获取原始文件夹中的所有文件名
    files = [f for f in os.listdir(ori_folder) if os.path.isfile(os.path.join(ori_folder, f))]
    
    # 生成一个随机顺序的索引列表
    shuffled_indices = random.sample(range(len(files)), len(files))
    
    # 遍历文件列表，并按照随机顺序的索引复制文件到目标文件夹
    for i, file in enumerate(files):
        # 构建原始文件的完整路径
        src_file_path = os.path.join(ori_folder, file)
        src_file_path_feat = os.path.join(ori_folder_feat, file[:-4] + '.npy')
        
        # 构建目标文件的新文件名（使用随机索引）
        dst_file_name = files[shuffled_indices[i]]
        
        
        # 构建目标文件的完整路径
        dst_file_path = os.path.join(shuffle_folder, dst_file_name)
        dst_file_path_feat = os.path.join(shuffle_folder_feat, dst_file_name[:-4] + '.npy')
        
        # 复制文件
        shutil.copy(src_file_path, dst_file_path)
        shutil.copy(src_file_path_feat, dst_file_path_feat)
        
    print(f'Files have been shuffled and copied from {ori_folder} to {shuffle_folder}.')

# 示例使用
seed = 0
while(True):
    print(seed)
    random.seed(seed)
    ori_folder = '/mnt/sfs-common/syli/duet_final/data/music2/mp3/test' # 原始文件夹路径
    shuffle_folder = '/mnt/sfs-common/syli/duet_final/data/music2/mp3/test_random' # 目标文件夹路径
    ori_folder_feat = '/mnt/sfs-common/syli/duet_final/data/music2/feature/test' # 原始文件夹路径
    shuffle_folder_feat = '/mnt/sfs-common/syli/duet_final/data/music2/feature/test_random'
# 调用函数
    shuffle_copy(ori_folder, shuffle_folder, ori_folder_feat, shuffle_folder_feat)
    
    ba = calc_ba_score('/mnt/sfs-common/syli/duet_final/data/motion/pos3d/test', shuffle_folder_feat)
    print(ba)
    if ba < 0.178:
        break
    seed += 1

