from generate_mod_LR_bic import generate_mod_LR_bic
import os

folder_in = '/home/ubuntu/Dataset/Vimeo-90k/vimeo_septuplet/sequences/train'
folder_out = '/home/ubuntu/Media/hdd0/dataset/Vimeo-90k/vimeo_septuplet/temp' # '/home/ubuntu/Dataset/Vimeo-90k/vimeo_septuplet/sequences_LR'
check_number = 57
up_scale = 4

# ====================
folder_root = folder_in
folder_leaf = []
folder_branch = []
file_leaf = []
index = 0
for dirpath, subdirnames, filenames in os.walk(folder_root):
    print('Processing ' + str(index) + ' Item')
    index += 1

    if len(subdirnames) == 0:
        # print('Leaf Folder: ' + dirpath)
        folder_leaf.append(dirpath)
    else:
        # print('Branch Folder: ' + dirpath)
        folder_branch.append(dirpath)

    for i in range(len(filenames)):
        # print('Leaf file: ' + os.path.join(dirpath, filenames[i]))
        file_leaf.append(os.path.join(dirpath, filenames[i]))

# ====================

for i in range(len(folder_leaf)):
    print('Processing ' + str(i) + ' Flip')
    path_in = folder_leaf[i]
    path_out = os.path.join(folder_out, 'LR', 'x' + str(up_scale),  folder_leaf[i][check_number:])
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    generate_mod_LR_bic(up_scale, path_in, path_out)

exit()
