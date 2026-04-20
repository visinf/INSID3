# Data Preparation

We provide instructions to set up the evaluation for **in-context segmentation** and **semantic correspondence**.

## 🚀 In-Context Segmentation

Our setup follows [Matcher](https://github.com/aim-uofa/Matcher), [GF-SAM](https://github.com/ANDYZAQ/GF-SAM), [DR-Adapter](https://github.com/Matt-Su/DR-Adapter), [ABCDFSS](https://github.com/Vision-Kek/ABCDFSS) and [Where's Waldo](https://github.com/dvirsamuel/PDM).  
Create a directory `data/` to store all datasets.

At the end of the preparation, we expect the following structure:

```text
INSID3/
├── data/
│   ├── COCO2014/
│   │   ├── annotations/
│   │   │   ├── train2014/
│   │   │   └── val2014/
│   │   ├── train2014/
│   │   ├── val2014/
│   │   └── splits/
│   │       └── val/
│   │       └── trn/
│   ├── LVIS/
│   │   ├── coco/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   ├── lvis_train.pkl
│   │   └── lvis_val.pkl
│   ├── PACO-Part/
│   │   ├── coco/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   └── paco/
│   │       ├── paco_part_train.pkl
│   │       └── paco_part_val.pkl
│   ├── Pascal-Part/
│   │   └── VOCdevkit/
│   │       └── VOC2010/
│   │           ├── JPEGImages/
│   │           ├── Annotations_Part_json_merged_part_classes/
│   │           └── all_obj_part_to_image.json
│   ├── SUIM/
│   │   ├── images/
│   │   └── masks/
│   │       ├── FV
│   │       ├── HD
│   │       ├── ...
│   ├── ISIC/
│   │   ├── ISIC2018_Task1_Training_GroundTruth/
│   │   └── ISIC2018_Task1-2_Training_Input/
│   │       ├── 1/
│   │       ├── 2/
│   │       └── 3/
│   ├── LungSegmentation/
│   │   ├── CXR_png/
│   │   └── masks/
│   ├── iSAID/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── semantic_png/
│   │   └── splits/
│   │       ├── val/
│   └── PerMIRS/
│       ├── 0/
│       ├── 4/
│       ├── 5/
│       ├── ...

```

## Instructions

Run all commands below from the `INSID3/data/` directory.  
Some datasets require `gdown` for Google Drive downloads (`pip install gdown==5.2.0`). 

### 🥥 COCO-20<sup>i</sup>

Download the COCO 2014 train/val images, annotations, and splits by running:
```bash
mkdir -p COCO2014 && cd COCO2014

# COCO 2014 images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
rm -f *.zip

# Annotations
mkdir -p annotations && cd annotations
gdown --fuzzy https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view
gdown --fuzzy https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view
unzip train2014.zip
unzip val2014.zip
rm -f *.zip
cd ..

# Split files
gdown --fuzzy https://drive.google.com/file/d/1K1UxhUzpjCe1jtgpsEUyrvU8gAlH1ZMy/view
unzip splits.zip
rm -f splits.zip
cd ..
```

As an alternative to downloading [splits.zip](https://drive.google.com/file/d/1K1UxhUzpjCe1jtgpsEUyrvU8gAlH1ZMy/view?usp=sharing), you can retrieve the `splits` folder directly from the original source: [Matcher/datasets/COCO2014/splits](https://github.com/aim-uofa/Matcher/tree/main/datasets/COCO2014/splits).

### 🦉 LVIS-92<sup>i</sup>
Download the COCO 2017 train/val images and the LVIS mask annotations:
```bash
mkdir -p LVIS/coco && cd LVIS/coco

# COCO 2017 images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
rm -f *.zip
cd ..

# LVIS mask annotations
gdown --fuzzy https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view
unzip lvis.zip
mv lvis/* .
rm -rf lvis lvis.zip
cd ..
 ```



### 🧩 PACO-Part
PACO-Part uses the same COCO 2017 images as LVIS. The following commands create a symbolic link to `data/LVIS/coco/` and then download the PACO-Part mask annotations (if you have not prepared LVIS, make sure `data/LVIS/coco/` exists first; see the LVIS instructions above):
```bash
mkdir -p PACO-Part && cd PACO-Part

# Reuse the COCO 2017 images from LVIS
ln -sf ../LVIS/coco .

# PACO-Part mask annotations
gdown --fuzzy https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view
unzip paco.zip
rm -f paco.zip
cd ..
 ```



### 🔩 Pascal-Part
Download the VOC 2010 train/val images together with the Pascal-Part mask annotations:

```bash
mkdir -p Pascal-Part && cd Pascal-Part

# VOC 2010 images
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -xf VOCtrainval_03-May-2010.tar

# Pascal-Part mask annotations
gdown --fuzzy https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view
unzip pascal.zip
mv pascal/* VOCdevkit/VOC2010/
rm -rf pascal.zip pascal VOCtrainval_03-May-2010.tar
cd ..
 ```
This should result in 11321 images in `VOCdevkit/VOC2010/JPEGImages`.


### 🩺 ISIC 2018
Download the ISIC 2018 training images and ground-truth masks, then preprocess the images following [DR-Adapter](https://github.com/Matt-Su/DR-Adapter) so that `ISIC2018_Task1-2_Training_Input/` is reorganized into the `1/`, `2/`, and `3/` subfolders:

```bash
mkdir -p ISIC && cd ISIC

# ISIC 2018 images and ground-truth masks
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip
unzip ISIC2018_Task1-2_Training_Input.zip
unzip ISIC2018_Task1_Training_GroundTruth.zip

# DR-Adapter preprocessing
wget -O ISIC_Split.py https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/ISIC_Split.py
mkdir -p isic
wget -O isic/class_id.csv https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/isic/class_id.csv
sed -i "s|dir = '../data/ISIC/ISIC/ISIC2018_Task1-2_Training_Input/'|dir = 'ISIC2018_Task1-2_Training_Input/'|" ISIC_Split.py
python ISIC_Split.py

rm -f *.zip ISIC_Split.py
rm -rf isic
rm -f ISIC2018_Task1-2_Training_Input/*.jpg
cd ..
```
Note: preparing ISIC additionally requires `pandas` (`pip install pandas`).
This should result in 2594 images and 2594 masks in total, split into 3 folds (1/, 2/, and 3/). The three folds contain 208, 1867, and 519 images.

### 🫁 Chest X-ray
Download the Chest X-ray dataset from [Chest X-ray](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) by running:


```bash
mkdir -p LungSegmentation && cd LungSegmentation
curl -L -o chest-xray-masks-and-labels.zip \
    https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels
unzip chest-xray-masks-and-labels.zip
mv 'Lung Segmentation'/* .
rm -rf 'Lung Segmentation' data *.zip
cd ..
```
This should result in 800 images (`CXR_png/`) and 704 masks (`masks/`).



### 🌊 SUIM

We prepare the SUIM dataset following [ABCDFSS](https://github.com/Vision-Kek/ABCDFSS/blob/main/data/README.md) by running:



```bash
mkdir -p SUIM && cd SUIM
curl -L -o suim-merged.zip \
    https://www.kaggle.com/api/v1/datasets/download/heyoujue/suim-merged
unzip suim-merged.zip
mv suim_merged/* .
rm -rf suim_merged suim-merged.zip
cd ..
```

This should result in 1635 images and 1635 masks in total.


### 🌍 iSAID
Download the iSAID dataset and splits by running:


```bash
mkdir -p iSAID && cd iSAID
gdown --fuzzy https://drive.google.com/file/d/17PQ1iKCbaj2OjwBdCn_VBh09ntI4lxgL/view?usp=sharing -O iSAID_5i.zip
unzip iSAID_5i.zip
mv iSAID_patches/* .

# Split files
gdown --fuzzy https://drive.google.com/file/d/1RBmVHXGhEXak2XB4OeCzQj2yswOmuMB2/view
unzip splits.zip
rm -rf *.zip iSAID_patches
cd ..
```
This should result in 6363 images  (`val/images/`) and 6363 masks (`val/semantic_png/`).

### 🎯 PerMIS

Download the required BURST data and prepare the PerMIS dataset following [Where’s Waldo](https://github.com/dvirsamuel/PDM) by running:

```bash
mkdir -p PerMIRS && cd PerMIRS

# BURST data
wget "https://motchallenge.net/data/3-TAO_TEST.zip"
wget https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip

unzip 3-TAO_TEST.zip
unzip annotations.zip

# PerMIS generation script
git clone https://github.com/dvirsamuel/PDM.git

cd PDM/PerMIRS
sed -i 's|base_path = f"/PerMIRS/{i}"|base_path = f"../../{i}"|' permirs_gen_dataset.py
PYTHONPATH=.:.. python permirs_gen_dataset.py --images_base_dir ../../frames --annotations_file ../../test/all_classes.json

cd ../..
rm -rf PDM *.zip test train val frames info
cd ..
 ```

This should result in 216 folders, each containing 3 frames and a `masks.npz` file.

## 🧹 SPair-71k (semantic correspondence analysis)

As an analysis, we evaluate the effect of positional debiasing on **semantic correspondence** using **SPair-71k**.   
After preparation, we expect the following structure:

```text
data/
├── SPair-71k/
│   ├── JPEGImages/{category}/*.jpg
│   ├── PairAnnotation/{split}/*.json
│   ├── ImageAnnotation/{category}/*.json
│   └── Layout/large/{trn,test}.txt
 ```

Download and extract the dataset with:

```bash
mkdir -p SPair-71k && cd SPair-71k
wget https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xzf SPair-71k.tar.gz --strip-components=1
rm -f SPair-71k.tar.gz
cd ..
 ```