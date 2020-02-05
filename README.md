

### Dataset
The [BraTS](http://www.med.upenn.edu/sbia/brats2018.html) data set is used for training and evaluating the model. This dataset contains four modalities for each individual brain, namely, T1, T1c (post-contrast T1), T2, and Flair which were skull-stripped, resampled and coregistered. For more information, please refer to the main site. 

### Pre-processing
For pre-processing the data, firstly, [N4ITK](https://ieeexplore.ieee.org/abstract/document/5445030) algorithm is adopted on each MRI modalities to correct the inhomogeneity of these images. Secondly, 1% of the top and bottom intensities is removed, and then each modality is normalized to zero mean and unit variance.
<br />
### Architecture
<br />

![image](https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/model.png)

The network is based on U-Net architecture with some modifications as follows:
- The minor modifications: adding Residual Units, strided convolution, PReLU activation and Batch Normalization layers to the original U-Net
- The attention mechanism: employing [Squeeze and Excitation Block](https://arxiv.org/abs/1709.01507) (SE) on concatenated multi-level features. This technique prevents confusion for the model by weighting each of the channels adaptively ([our paper](https://ieeexplore.ieee.org/document/8964956)).

<p align="center"><img src="https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/attention.png" width="500" height="250"></p>

### Training Process

Since our proposed network is a 2D architecture, we need to extract 2D slices from 3D volumes of MRI images. To benefit from 3D contextual information of input images, we extract 2D slices from both Axial and Coronal views, and then train a network for each view separately. In the test time, we build the 3D output volume for each model by concatenating the 2D predicted maps. Finally, we fuse the two views by pixel-wise averaging. 

![image](https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/MultiView.png)


### Results
These results are obtained from the [BraTS online evaluation platform](https://ipp.cbica.upenn.edu/) using the BRATS 2018 validation data set

![image](https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation/blob/master/doc/table.png)


### Usage
1- Download the BRATS 2019, 2018 or 2017 data by following the steps described in [BraTS](https://www.med.upenn.edu/cbica/brats2019/registration.html)
2- Perform N4ITK bias correction using [ANTs](https://github.com/ANTsX/ANTs), follow the steps in [this repo](https://github.com/ellisdg/3DUnetCNN) (This step is optional)
3- Set The path to all brain volumes in the config.py (ex:  cfg['data_dir'] ='./BRATS19/MICCAI_BraTS_2019_Data_Training/*/*')
4- Read, preprocess and save all brain volumes into a single table file
```
python prepare_data.py
```
5- Run the training:
```
python train.py
```
The model can be trained from 'axial', 'saggital' or 'coronal' views (set cfg['view'] in the config.py). Moreover, K-fold cross-validation can be used (set cfg['k_fold'] in the config.py)

5- To predict and save mlabel aps:
```
python predict.py
```
The predictions will be written in .nii.gz format and can be uploaded to BraTS online evaluation platform.

