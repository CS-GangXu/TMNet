# Data Preparation
## Vimeo-90K Septuplet Dataset
### 1. Download the original training + test set of `Vimeo-septuplet` (82 GB).
```
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
apt-get install unzip
unzip vimeo_septuplet.zip
```

### 2. Split the `Vimeo-septuplet` into a training set and a test set. 
Please make sure you change the dataset's path to your download path in script, also you need to run for the training set and test set separately.
```
cd $ROOT/data/data_scripts
python sep_vimeo_list.py
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
├──sep_trainlist.txt
├── sep_testlist.txt
```

### 3. Generate low resolution (LR) images (remember to configure the input and output path):
<!-- ```Matlab
# In Matlab Command Window
run $ROOT/data/data_scripts/generate_LR_Vimeo90K.m
``` -->

```
python $ROOT/data/data_scripts/generate_mod_LR_bic.py    
```

### 4. Create the LMDB files for faster I/O speed. Note that you need to configure your input and output path in the following script:
```
python $ROOT/data/data_scripts/create_lmdb_mp.py
```
The structure of generated lmdb folder is as follows:
```
Vimeo7_train.lmdb
├── data.mdb
├── lock.mdb
└── Vimeo7_train_keys.pkl
```
Please put the ```Vimeo7_train_keys.pkl``` into the folder of ``````, After that
```
├── fast_of_test
│   ├── HR
│   └── LR
├── medium_of_test
│   ├── HR
│   └── LR
├── slow_of_test
│   ├── HR
│   └── LR
├── vimeo7_train_GT.lmdb
│   ├── data.mdb
│   ├── lock.mdb
│   ├── meta_info.pkl
│   └── Vimeo7_train_keys.pkl
├── vimeo7_train_LR7.lmdb
│   ├── data.mdb
│   └── lock.mdb
└── Vimeo7_train_keys.pkl

```

## Vid4 Dataset
To be uploaded...

## Adobe240fps Dataset
To be uploaded...