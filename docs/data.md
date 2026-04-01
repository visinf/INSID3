# Data Preparation


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


### 🥥 COCO-20<sup>i</sup>

Download COCO2014 train/val images and annotations:
```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

Download train/val annotations: [train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing), [val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing). Unzip and place both `train2014/` and `val2014/` under `data/COCO2014/annotations/`.
Download [splits.zip](https://drive.google.com/file/d/1K1UxhUzpjCe1jtgpsEUyrvU8gAlH1ZMy/view?usp=sharing) (alternatively, you may download the `splits` folder from [Matcher](https://github.com/aim-uofa/Matcher/tree/main/datasets/COCO2014/splits)). Unzip it and place the `splits/` folder under `data/COCO2014/`.

### 🦉 LVIS-92<sup>i</sup>
Download COCO2017 train/val images: 
 ```
 wget http://images.cocodataset.org/zips/train2017.zip
 wget http://images.cocodataset.org/zips/val2017.zip
 ```
Unzip and place under `data/LVIS/coco`. Download LVIS-92<sup>i</sup> extended mask annotations: [lvis.zip](https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view?usp=sharing).
Unzip and place the `.pkl` files under `data/LVIS/`.


### 🧩 PACO-Part
Download COCO2017 train/val images (same as LVIS): 
 ```
 wget http://images.cocodataset.org/zips/train2017.zip
 wget http://images.cocodataset.org/zips/val2017.zip
 ```

Unzip and place under `data/PACO-Part/coco`. Download PACO-Part extended mask annotations: [paco.zip](https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view?usp=sharing).
Unzip and place the `.pkl` files under `data/PACO-Part/paco/`.

### 🔩 Pascal-Part
 Download VOC2010 train/val images: 
 ```
 wget http://roozbehm.info/pascal-parts/trainval.tar.gz
 wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
 ```
 Download Pascal-Part extended mask annotations: [pascal.zip](https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view?usp=sharing).
Unzip everything into `data/Pascal-Part/VOCdevkit/VOC2010/`.



### 🩺 ISIC 2018
Download the dataset from the [ISIC Challenge 2018](https://challenge.isic-archive.com/data/#2018).
Download the images and annotations:
```
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip 
```
Unzip everything into `data/ISIC/`. This should result in 2594 images and 2594 masks in total. Then, preprocess `ISIC2018_Task1-2_Training_Input` data following [DR-Adapter](https://github.com/Matt-Su/DR-Adapter), running their `./data_util/ISIC_Split.py` script. The script reorganizes the images into three categories: `ISIC2018_Task1-2_Training_Input/` will contain the `1/`, `2/`, and `3/` subfolders after preprocessing.

### 🫁 Chest X-ray
Download the dataset from the [Chest X-ray](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels):

```bash
curl -L -o chest-xray-masks-and-labels.zip https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels
unzip chest-xray-masks-and-labels.zip
mv 'Lung Segmentation' LungSegmentation
```
Place `LungSegmentation/` under `data/`. This should result in 800 images (`CXR_png/`) and 704 masks (`masks/`).



### 🌊 SUIM

We follow [ABCDFSS](https://github.com/Vision-Kek/ABCDFSS/blob/main/data/README.md):

```bash
curl -L -o suim-merged.zip https://www.kaggle.com/api/v1/datasets/download/heyoujue/suim-merged
unzip suim-merged.zip
mv suim_merged SUIM
```
Place `SUIM/` under `data/`.
This should result in 1635 images and 1635 masks in total.


### 🌍 iSAID
Download the dataset:

```bash
gdown --fuzzy https://drive.google.com/file/d/17PQ1iKCbaj2OjwBdCn_VBh09ntI4lxgL/view?usp=sharing -O iSAID_5i.zip
unzip iSAID_5i.zip
mv iSAID_patches iSAID
```
Download [splits.zip](https://drive.google.com/file/d/1RBmVHXGhEXak2XB4OeCzQj2yswOmuMB2/view?usp=sharing). Unzip and place it under `iSAID/`. Place `iSAID/` under `data/`. 

### 🎯 PerMIS

The PerMIS dataset is derived from the [BURST dataset](https://github.com/Ali2500/BURST-benchmark). Download the required BURST data from the official repository:

```
wget "https://motchallenge.net/data/1-TAO_TRAIN.zip" 
wget "https://motchallenge.net/data/2-TAO_VAL.zip" 
wget "https://motchallenge.net/data/3-TAO_TEST.zip" 
wget https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip
 ```

Then follow the preprocessing steps provided in the [Where’s Waldo](https://github.com/dvirsamuel/PDM) repository to prepare the PerMIS dataset.
