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
To be uploaded...

## Adobe240fps Dataset
To be uploaded...