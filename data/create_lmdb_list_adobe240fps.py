import shutil
import os

folder_root = '../datasets/adobe240fps/frame/train'
txt_path = '../datasets/adobe240fps/train_lmdb_list.txt'

folder_root = 
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

with open(txt_path, 'w') as f:
    for i in range(len(folder_leaf)):
        folder = folder_leaf[i]
        if i == len(folder_leaf) - 1:
            f.write('/'.join(folder.split('/')[-2:]))
        else:
            f.write('/'.join(folder.split('/')[-2:]) + '\n')

exit()