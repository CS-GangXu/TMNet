# Data Preparation
<!-- We hope that you can get a dataset folder ```$ROOT/datasets``` with the structure: -->
## Vimeo-90K Septuplet Dataset
### 1. Download the original training + test set of `Vimeo-septuplet` (82 GB).
```
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
apt-get install unzip
unzip vimeo_septuplet.zip
```
This will create `train` and `test` folders in the directory of **`vimeo_septuplet/sequences`**. The folder structure is as follows:
```
vimeo_septuplet
├── sequences
    ├── 00001
        ├── 0266
            ├── im1.png
            ├── ...
            └── im7.png
        ├── 0268...
    ├── 00002...
├── readme.txt
├── sep_trainlist.txt
└── sep_testlist.txt
```

### 2. Split the `Vimeo-septuplet` into a training set and a test set (remember to configure the input and output path first!). 
Please make sure you change the dataset's path to your download path in script, also you need to run for the training set and test set separately. The ```sep_*.txt``` (```sep_trainlist.txt```, ```sep_testlist.txt```, ```sep_fast_testset.txt```, ```sep_medium_testset.txt```, ```sep_slow_testset.txt```) can be downloaded from [Link](https://drive.google.com/drive/folders/1PjXClB-S8pyB6y1UWJQnZK7fela5Lcu1?usp=sharing).
```
cd $ROOT/data
python sep_vimeo_list_Vimeo90K.py
```
You also need to split the test folder into fast, medium and slow subsets using ```sep_fast_testset.txt```, ```sep_medium_testset.txt``` and ```sep_slow_testset.txt```.

### 3. Generate low resolution (LR) images (remember to configure the input and output path first!):
```
cd $ROOT/data
python generate_LR_Vimeo90K.py
```

### 4. Create the LMDB files for faster I/O speed. Note that you need to configure your input and output path in the following script (remember to configure the input and output path first!):
```
cd $ROOT/data
python create_lmdb_mp_Vimeo90K_LR.py
python create_lmdb_mp_Vimeo90K_HR.py
```
The structure of generated lmdb folder is as follows:
```
Vimeo7_train.lmdb
├── data.mdb
├── lock.mdb
└── Vimeo7_train_keys.pkl
```
Please copy the ```Vimeo7_train_keys.pkl``` into the folder with higher level. After the aboved operations, we assume that you can get a folder with the following structure:
```
├── fast_of_test # For testing
│   ├── HR
│   └── LR
├── medium_of_test # For testing
│   ├── HR
│   └── LR
├── slow_of_test # For testing
│   ├── HR
│   └── LR
├── vimeo7_train_GT.lmdb # For training
│   ├── data.mdb
│   ├── lock.mdb
│   └── Vimeo7_train_keys.pkl
├── vimeo7_train_LR7.lmdb # For training
│   ├── data.mdb
│   ├── lock.mdb
│   └── Vimeo7_train_keys.pkl
└── Vimeo7_train_keys.pkl
```

## Vid4 Dataset
You can download Vid4 dataset (```vid4.tar```) via [Link](https://drive.google.com/drive/folders/1PjXClB-S8pyB6y1UWJQnZK7fela5Lcu1?usp=sharing). If you download the Vid4 dataset, you can extract the image from ```vid4.tar``` and put them into ```$ROOT/datasets/vid4```. We assume that you can get a folder with the following structure:
```
vid4
├── HR
│   ├── calendar
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── ***.png
│   ├── city
│   ├── foliage
│   └── walk
└── LR
    ├── calendar
    │   ├── 00000000.png
    │   ├── ...
    │   └── ***.png
    ├── city
    ├── foliage
    └── walk
```
We only use this dataset for evaluation.

## Adobe240fps Dataset
We recommend that you use the ```opencv-python``` with the version 4.4.0.46.

### 1. Download Dataset.

You can download Adobe240fps via [Link](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip) and extract the video files in ```$ROOT/data/adobe240fps/video``` folder.

Then, You should download the record files: adobe240fps_folder_*.txt (adobe240fps_folder_train.txt, adobe240fps_folder_valid.txt, adobe240fps_folder_test.txt), which can be downloaded from [Link](https://drive.google.com/drive/folders/1PjXClB-S8pyB6y1UWJQnZK7fela5Lcu1) for following processing operations.
### 2. Extract frames from the video files (remember to configure the input and output path first!).
```
cd $ROOT/data
python generate_frames_from_adobe240fps.py
```
### 3. Create the HR and LR image pairs (remember to configure the input and output path first!).
```
cd $ROOT/data
python generate_data_adobe240fps.py
```
### 4. Create LMDB files.
You should create file for following lmdb processing operation first.
```
cd $ROOT/data
python create_lmdb_list_adobe240fps.py
```
Create LMDB file.
```
cd $ROOT/data
python create_lmdb_mp_adobe240fps_LR.py
python create_lmdb_mp_adobe240fps_HR.py
```
Get ```pkl``` file.
```
cd $ROOT/data
sh get_adobe240fps_pkl.sh
```
### 5. Organize the files for validation and test.

We recommend that you organize the files of ```valid``` and ```test``` into the following structure:
```
adobe240fps
├── valid
│   ├── HR
│   │   ├── IMG_0030
│   │   │   ├── 0.png
│   │   │   ├── ...
│   │   │   └── ***.png
│   │   ├── GOPR9654a
│   │   ├── IMG_0002
│   │   └── IMG_0153
│   └── LR
│       ├── IMG_0030
│       │   ├── 0.png
│       │   ├── ...
│       │   └── ***.png
│       ├── GOPR9654a
│       ├── IMG_0002
│       └── IMG_0153
└── test
    ├── HR
    │   ├── GOPR9653
    │   │   ├── 0.png
    │   │   ├── ...
    │   │   └── ***.png
    │   ├── IMG_0001
    │   ├── IMG_0003
    │   └── IMG_0004a
    └── LR
        ├── GOPR9653
        │   ├── 0.png
        │   ├── ...
        │   └── ***.png
        ├── IMG_0001
        ├── IMG_0003
        └── IMG_0004a
```