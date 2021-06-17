from generate_mod_LR_bic import generate_mod_LR_bic
import os

# Configuration
# ============================

args = [
        {
            'folder_in': '../datasets/adobe240fps/frame',
            'folder_out': '../datasets/adobe240fps/frame_LRx4',
            'up_scale': 8,
        },
        {
            'folder_in': '../datasets/adobe240fps/frame',
            'folder_out': '../datasets/adobe240fps/frame_HR',
            'up_scale': 2,
        },
    ]

# Run
# ============================

for arg in args:
    folder_in = arg['folder_in']
    folder_out = arg['folder_out']
    up_scale = arg['up_scale']

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
        path_out = os.path.join(folder_out, folder_leaf[i][len(folder_in) + 1:])
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        generate_mod_LR_bic(up_scale, path_in, path_out)