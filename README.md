# Automatic flare ribbons segmentation and detection of parallel shapes
 
**This repository contains algorithm used for detecting parallel flare ribbons in SDO AIA images.**
The solution is based on automatic segmentation by CNN and computer vision techniques applied on segmented images.

Repository contains: 
* File **data_preparation.ipynb** has all the steps required to prepare data for detection. There's no need to use this file unless you want to create new files. 
* File **main.ipynb** contains the main algorithm and model training. All used data and trained model are downloaded within this file. 
--------------------------------
* Folder **src** containing model SCSS-Net, used for segmentations, and several functions. Source of most of its contents: https://github.com/space-lab-sk/scss-net.
* Folder **dsepruning** and file **setup.py** are used for pruning skeletons of segmentations. Source: https://github.com/originlake/DSE-skeleton-pruning.
* File **requirements.txt** has a list of all needed libraries
* File **ribbondb_v1.0.csv** is a catalogue of solar flares used for downloading the data. Source: http://solarmuri.ssl.berkeley.edu/~kazachenko/RibbonDB/
