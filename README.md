# Brain-Tumor-Segmentation

Implementation of the BraTS 2019 winning model

BraTS 2018 dataset can be downloaded from this link [here](https://www.med.upenn.edu/sbia/brats2018/data.html)

Move the downloaded files to dataset directory

Move the pretrained weights to weights directory

To train the model or to load pretrained weights run train.py

Link to the paper: 

https://arxiv.org/pdf/1810.11654#:~:text=In%20this%20work%2C%20we%20describe,won%20the%20BraTS%202018%20challenge.&text=In%20particular%2C%20EMMA%20combined%20DeepMedic,and%20ensembled%20their%20segmentation%20predictions.

Model architecture:


![](images/Screenshot%202022-01-31%20225714.png)

Results:



![](images/Screenshot%202022-01-31%20225850.png)

Dice Loss for Enhancing Tumor: 0.8145


Dice Loss for Whole Tumor: 0.9042


Dice Loss for Tumor Core: 0.8596
