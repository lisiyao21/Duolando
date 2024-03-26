import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os


def paired_collate_fn(insts):
    # for src in insts:
    #     for s in src:
    #         print(s.shape)

    # print()

    mo_seq = list(zip(*insts))
    print(mo_seq.shape)
    mo_seq = torch.FloatTensor(mo_seq)
    print('Here!')
    print(mo_seq.size())

    return mo_seq


class DD100lfAll(Dataset):
    def __init__(self, music_root, motion_root, split='train', interval=None, dtype='pos3d', move=8, music_dance_rate=1, expansion=1):
        self.dances = {'rotmatl':[], 'rotmatf':[], 'pos3dl':[], 'pos3df':[], 'music':[]}
        dtypes = ['rotmat', 'pos3d']
        self.dtype = dtype
        self.expansion = expansion
        self.names = []

        music_files = {}
        agent_files = {'leader':{}, 'follower':{}}
        
        music_seqs = {}

        for mname in os.listdir(os.path.join(music_root,  'feature', split)):
            path = os.path.join(music_root, 'feature', split, mname)
            music_files[mname[:-4]] = path

        for fname in os.listdir(os.path.join(motion_root, 'pos3d', split)): 
            path = os.path.join(motion_root, 'pos3d', split, fname)
            if path.endswith('_00.npy'):
                agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path
            
        for take in agent_files['follower']:
            # print(take)
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            len_this_music = len(np_music) // int(music_dance_rate) * int(music_dance_rate)
            # print(len_this_music)
            
            for dtype_folder in dtypes:
                this_pair = {}
                for agent in agent_files:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    np_dance = np.load(dance_path)

                    if dtype_folder == 'pos3d':
                        root = np_dance[:, :3]  # the root
                        np_dance = np_dance - np.tile(root, (1, 55))  # Calculate relative offset with respect to root
                        np_dance[:, :3] = root
                        # the root of left hand
                        left_twist = np_dance[:, 60:63]
                        # 25,40
                        np_dance[:, 75:120] = (np_dance[:, 75:120] - np.tile(left_twist, (1, 15))) * 10
                        # the root of right hand
                        right_twist = np_dance[:, 63:66]
                        # 40,55
                        np_dance[:, 120:165] = (np_dance[:, 120:165] - np.tile(right_twist, (1, 15))) * 10
                    if dtype_folder == 'rotmat':
                        np_dance = np_dance[:, 3:]
                    this_pair[agent] = np_dance

                ldance, fdance = this_pair['leader'], this_pair['follower']
                lenf, dim = ldance.shape
                lenl, dim = fdance.shape
                seq_len = min(lenf, lenl)
                # print(lenf, lenl)

                if (interval is not None) and ( interval != 'None' ):
                    for i in range(0, seq_len, move):

                        np_dance_sub_seq_l = ldance[i: i + interval]
                        np_dance_sub_seq_f = fdance[i: i + interval]
                        
                        np_music_sub_seq = np_music[i//music_dance_rate:i//music_dance_rate + interval//music_dance_rate]

                        if len(np_dance_sub_seq_l) != interval or len(np_dance_sub_seq_f) != interval or len(np_music_sub_seq) != interval // music_dance_rate:
                            # print(len(np_dance_sub_seq), len(np_music_sub_seq))
                            continue
                        self.dances[dtype_folder+'l'].append(np_dance_sub_seq_l)
                        self.dances[dtype_folder+'f'].append(np_dance_sub_seq_f)
                        if dtype_folder != 'rotmat':
                            self.dances['music'].append(np_music_sub_seq)
                            self.names.append(take)
                else:
                    self.dances[dtype_folder+'l'].append(ldance[:seq_len//move*move])
                    self.dances[dtype_folder+'f'].append(fdance[:seq_len//move*move])
                    if dtype_folder != 'rotmat':
                        self.dances['music'].append(np_music[:seq_len//move*move])
                        self.names.append(take)

    def __len__(self):
        # if len(self.dances['pos3dl']) < 20:
        # return 3
        # else:
        #     return 33
            # return len(self.dances['pos3dl'])
        return len(self.dances['pos3dl'])*self.expansion

    def __getitem__(self, index):
        # if len(self.dances['pos3dl']) > 20:
        index = index // self.expansion
        return {'pos3dl':self.dances['pos3dl'][index], 'pos3df':self.dances['pos3df'][index], 'rotmatl':self.dances['rotmatl'][index], 'rotmatf':self.dances['rotmatf'][index], 'music':self.dances['music'][index], 'fname':self.names[index]}

# if __name__ == '__main__':

#     keywords = ['Fox', 'Jive', 'Paso', 'Lumba', 'Qiaqia', 'Quick', 'Samba', 'Tango', 'Waltz']
#     stat = {}
#     stat_hand = {}
#     stat_foot = {}
#     stat_std = {}

#     dd100 = DD100lfAll('/mnt/lustre/syli/duet/data/music', '/mnt/lustre/syli/duet/data/motion', split='train', interval=None, dtype='both')
#     dd1002 = DD100lfAll('/mnt/lustre/syli/duet/data/music', '/mnt/lustre/syli/duet/data/motion', split='test', interval=None, dtype='both')
    
#     # dd100 += dd1002
#     print(len(dd100.dances['pos3dl']), len(dd100.dances['rotmatl']), len(dd100.dances['pos3df']), len(dd100.dances['rotmatf']), len(dd100.dances['music']), len(dd100.names))
#     for ii in range(len(dd100.dances['pos3dl'])):
#         # print(ii)
#         # if len(dd100.dances['pos3d'][ii]) != len(dd100.dances['music'][ii]):
#         gp = dd100.dances['pos3df'][ii][:, :3]

#         # 10， 11， 25， 40
#         gv = gp[1:, :] - gp[:-1, :]
#         gv_mad = np.sqrt(np.sum(gv**2, axis=1)).mean()
#         gv_std = np.sqrt(np.sum(gv**2, axis=1)).std()

#         gh = dd100.dances['pos3df'][ii][:, 30:33] 
#         gh2 = dd100.dances['pos3df'][ii][:, 33:36] 
#         ghv = gh[1:, :] - gh[:-1, :]
#         ghv2 = gh2[1:, :] - gh2[:-1, :]
#         ghv_mad = np.sqrt(np.sum(ghv**2, axis=1)).mean() * 0.5 +  np.sqrt(np.sum(ghv2**2, axis=1)).mean() * 0.5

#         gf = dd100.dances['pos3df'][ii][:, 60:63] 
#         gf2 = dd100.dances['pos3df'][ii][:, 63:66] 
#         gfv = gf[1:, :] - gf[:-1, :]
#         gfv2 = gf2[1:, :] - gf2[:-1, :]
#         gfv_mad = np.sqrt(np.sum(gfv**2, axis=1)).mean() * 0.5 +  np.sqrt(np.sum(gfv2**2, axis=1)).mean() * 0.5

#         keyword = 'Ballet'
#         for kk in keywords:
#             if kk in dd100.names[ii]:
#                 keyword = kk
#                 break
#         if keyword in stat:
#             stat[keyword].append(gv_mad)
#             stat_hand[keyword].append(ghv_mad)
#             stat_foot[keyword].append(gfv_mad)
#             stat_std[keyword].append(gv_std)
#         else:
#             stat[keyword] = []
#             stat_hand[keyword] = []
#             stat_foot[keyword] = []
#             stat_std[keyword] = []

#     # print(len(dd100.dances['pos3dl']), len(dd100.dances['rotmatl']), len(dd100.dances['pos3df']), len(dd100.dances['rotmatf']), len(dd100.dances['music']), len(dd100.names))
    
#     for ii in range(len(dd1002.dances['pos3dl'])):
#         # print(ii)
#         # if len(dd100.dances['pos3d'][ii]) != len(dd100.dances['music'][ii]):
#         gp = dd1002.dances['pos3df'][ii][:, :3]

#         # 10， 11， 25， 40
#         gv = gp[1:, :]  - gp[:-1, :]
#         gv_mad = np.sqrt(np.sum(gv**2, axis=1)).mean()
#         gv_std = np.sqrt(np.sum(gv**2, axis=1)).std()

#         gh = dd1002.dances['pos3df'][ii][:, 30:33]  
#         gh2 = dd1002.dances['pos3df'][ii][:, 33:36]  
#         ghv = gh[1:, :] - gh[:-1, :]
#         ghv2 = gh2[1:, :] - gh2[:-1, :]
#         ghv_mad = np.sqrt(np.sum(ghv**2, axis=1)).mean() * 0.5 +  np.sqrt(np.sum(ghv2**2, axis=1)).mean() * 0.5

#         gf = dd1002.dances['pos3df'][ii][:, 60:63] 
#         gf2 = dd1002.dances['pos3df'][ii][:, 63:66]
#         gfv = gf[1:, :] - gf[:-1, :]
#         gfv2 = gf2[1:, :] - gf2[:-1, :]
#         gfv_mad = np.sqrt(np.sum(gfv**2, axis=1)).mean() * 0.5 +  np.sqrt(np.sum(gfv2**2, axis=1)).mean() * 0.5

#         keyword = 'Ballet'
#         for kk in keywords:
#             if kk in dd1002.names[ii]:
#                 keyword = kk
#                 break
#         if keyword in stat:
#             stat[keyword].append(gv_mad)
#             stat_hand[keyword].append(ghv_mad)
#             stat_foot[keyword].append(gfv_mad)
#             stat_std[keyword].append(gv_std)
#         else:
#             stat[keyword] = []
#             stat_hand[keyword] = []
#             stat_foot[keyword] = []
    
#     whole = []
#     hand = []
#     foot = []
#     keywords =['Waltz', 'Jive', 'Tango', 'Ballet', 'Samba', 'Rumba', 'Paso Dobal', 'Quickstep', 'Foxtrot', 'Cha cha']


#     for keyword in stat:
#         # print(keyword, np.mean(stat[keyword]) ,np.mean(stat_foot[keyword]), np.mean(stat_hand[keyword]), flush=True)
#         whole.append(np.mean(stat[keyword]) * 100)
#         hand.append(np.mean(stat_foot[keyword]) * 100)
#         foot.append(np.mean(stat_hand[keyword]) * 100)
#     print(len(whole), len(hand), len(foot), flush=True)
    
#     barWidth = 0.3
#     # keywords.append('Ballet')
#     print(whole, '>>>>', flush=True)

#     r1 = np.arange(len(whole))*1.0
#     r2 = r1 + barWidth
#     r3 = r2 + barWidth
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(20, 6))
#     plt.bar(r1, np.array(whole), width=barWidth, color=[0.6, 1, 0.6], edgecolor='black', label='Root')
#     plt.bar(r2, hand, width=barWidth, color=[0.5, 0.8, 1.0], edgecolor='black', label='Hands')
#     plt.bar(r3, foot, width=barWidth, color=[1, 0.9, 0.2], edgecolor='black', label='Feet')

#     plt.xlabel(u'Dance type', fontsize=18)
#     plt.xticks([r + barWidth for r in range(len(whole))], keywords, fontsize=18)

#     plt.legend(fontsize=18)
#     # plt.xticks([1, 10, 20, 30, 40, 50], [1, 10, 20, 30, 40, 50], fontsize=14)
#     plt.yticks([ii for ii in range(0, 7)], [ii for ii in range(0, 7)], fontsize=18)
#     # plt.legend()
  
#     plt.ylabel(u'Speed (cm/frame)', fontsize=18)
#     plt.title('Avg. speed per frame', fontsize=20)
#     # plt.savefig( store_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d') + '_reward.pdf')


#     # plt.legend()
#     plt.savefig('rebuttal_speed.pdf')

    
        # print(keyword, np.mean(stat[keyword]), flush=True)
        # # print(keyword, np.std(stat[keyword]), flush=True)
        # print(keyword, np.mean(stat_hand[keyword]), flush=True)
        # print(keyword, np.mean(stat_foot[keyword]), flush=True) 


    
            #   len(dd100.dances['rotmatl'][ii]), len(dd100.dances['pos3df'][ii]), len(dd100.dances['rotmatf'][ii]), len(dd100.dances['music'][ii]), dd100.names[ii], flush=True)
        # print(dd100.dances['rotmatl'][ii][55], dd100.dances['rotmatf'][ii][55])
if __name__ == '__main__':
    dd100 = DD100lfAll('/mnt/sfs-common/syli/duet_final/data/music2', 
        '/mnt/sfs-common/syli/duet_final/data/motion', split='test', interval=None, dtype='both')
    time = {}
    for sample in dd100:
        pass
        # print(len(sample['pos3dl']), sample['fname'])
        # if sample['fname'][:-4] not in time:
        #     time[sample['fname'][:-4]] = len(sample['pos3dl'])
        # else:
        #     if time[sample['fname'][:-4]]  < len(sample['pos3dl']):
        #         time[sample['fname'][:-4]] = len(sample['pos3dl'])
        # time[sample['fname']] = len(sample['pos3dl'])
    dd1002 = DD100lfAll('/mnt/sfs-common/syli/duet_final/data/music2', 
        '/mnt/sfs-common/syli/duet_final/data/motion', split='train', interval=None, dtype='both')
    # time = {}
    for sample in dd1002:
        # print(len(sample['pos3dl']), sample['fname'])
        if sample['fname'][:-4] not in time:
            time[sample['fname'][:-4]] = len(sample['pos3dl'])
        else:
            if time[sample['fname'][:-4]]  < len(sample['pos3dl']):
                time[sample['fname'][:-4]] = len(sample['pos3dl'])
        # time[sample['fname']] = len(sample['pos3dl'])
        # pass
        
    time_list = []
    for tt in time:
        time_list.append(time[tt])
    print(len(time_list))
    print(np.sum(time_list), np.sum(time_list)*1.0/30, np.min(time_list)*1.0/30, np.max(time_list)*1.0/30)

    # print(len(dd100.dances['pos3dl']), len(dd100.dances['rotmatl']), len(dd100.dances['pos3df']), len(dd100.dances['rotmatf']), len(dd100.dances['music']), len(dd100.names))
    # for ii in range(len(dd100.dances['pos3dl'])):
    #     # if len(dd100.dances['pos3d'][ii]) != len(dd100.dances['music'][ii]):
    #     print(len(dd100.dances['pos3dl'][ii]), len(dd100.dances['rotmatl'][ii]), len(dd100.dances['pos3df'][ii]), len(dd100.dances['rotmatf'][ii]), len(dd100.dances['music'][ii]), dd100.names[ii], flush=True)
    #     # print(dd100.dances['rotmatl'][ii][55], dd100.dances['rotmatf'][ii][55])