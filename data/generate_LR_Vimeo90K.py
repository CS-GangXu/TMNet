from generate_mod_LR_bic import generate_mod_LR_bic
import os

# Configure the path =========================================================
folder_in = '/home/ubuntu/Dataset/Vimeo-90k/vimeo_septuplet/sequences/train'
folder_out = '/home/ubuntu/Dataset/Vimeo-90k/vimeo_septuplet/sequences_LR'
up_scale = 4
# ============================================================================

folder_root = folder_in
folder_leaf = []
folder_branch = []
file_leaf = []
index = 0

for dirpath, subdirnames, filenames in os.walk(folder_root):
    print('Processing ' + str(index) + ' Item')
    index += 1

    if len(subdirnames) == 0:
        folder_leaf.append(dirpath)
    else:
        folder_branch.append(dirpath)

    for i in range(len(filenames)):
        file_leaf.append(os.path.join(dirpath, filenames[i]))

for i in range(len(folder_leaf)):
    print('Processing ' + str(i) + ' to Get LR image')
    path_in = folder_leaf[i]
    path_out = os.path.join(folder_out, '/'.join(folder_leaf[i].split('/')[-3:]))
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    generate_mod_LR_bic(up_scale, path_in, path_out)
