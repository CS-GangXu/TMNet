import cv2
import numpy as np
from PIL import Image
import glob
import cv2
import os

mov_files = glob.glob("/home/ubuntu/Dataset/adobe240fps/original_high_fps_videos/*")

def check_if_folder_exist(folder_path='/home/ubuntu/'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

name_list = []
for i, mov_path in enumerate(mov_files):
    # single_list = []
    # if i == 0:
    #     pass
    # else:
    #     break

    # if i <= 1:
    #     continue
    # else:
    #     if i <= 25:
    #         pass
    #     else:
    #         break

    # if i <= 20:
    #     pass
    # else:
    #     break
    
    j = 0
    video = cv2.VideoCapture(mov_path)
    save_folder = os.path.join('/home/ubuntu/Media/hdd1/Dataset/adobe240fps/visualization/', mov_path.split('/')[-1].split('.')[0])
    check_if_folder_exist(save_folder)
    success, frame = video.read()
    while success:
        img_cv2 = np.transpose(frame, (0, 1, 2))
        
        cv2.imwrite(os.path.join(save_folder, str(j) + '.png'), img_cv2)
        # single_list.append(mov_path.split('/')[-1].split('.')[0] + '_' + str(j) + '.png')
        success, frame = video.read()
        print(str(i) + ' ' + str(j))
        j += 1
    
    # end_point = int(len(single_list)/10) * 10
    # single_list = single_list[0:end_point]
    # name_list += single_list
    # np.save('../data/adobe240fps_sub10000_valid_dict.npy', np.array(name_list))
    # np.save('./data/adobe240fps/' + mov_path.split('/')[-1].split('.')[0] + '.npy', np.array(np_dict))