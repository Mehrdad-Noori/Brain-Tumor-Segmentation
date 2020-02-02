
"""

The required configurations for training phase ('prepare_Data.py', 'train.py').

"""

cfg = dict()



"""
The coordinates to crop brain volumes. For example, a brain volume with the 
One can set the x0,x1,... by calculating none zero pixels through dataset. 
Note that the final three shapes must be divisible by the network downscale rate.
"""
cfg['crop_coord']            =  {'x0':42, 'x1':194,
                                 'y0':29, 'y1':221,
                                 'z0':2,  'z1':146}



"""
The path to all brain volumes (ex: suppose we have a folder 'BraTS2019' that 
contains two HGG and LGG folders each of which contains some folders so:
dataset_dir="./BraTS2019/*/*")
"""
cfg['data_dir']              = '/media/meno/Game/DataSets/BRATS19/MICCAI_BraTS_2019_Data_Training/*/*'



"""
The final data shapes of saved table file.
"""
cfg['table_data_shape']      =  (cfg["crop_coord"]['z1']-cfg["crop_coord"]['z0'],
                                 cfg["crop_coord"]['y1']-cfg["crop_coord"]['y0'], 
                                 cfg["crop_coord"]['x1']-cfg["crop_coord"]['x0'])



"""
BraTS datasets contain 4 channels: (FLAIR, T1, T1ce, T2)
"""
cfg['data_channels']         = 4



"""
The path to save table file + folds indexes + models + ...
"""
cfg['save_dir']              = './data/'



"""
k-fold cross-validation
"""
cfg['k_fold']                = 5



"""
The defualt path of saved table.
"""
cfg['hdf5_dir']              = './data/data.hdf5'



"""
The path to brain indexes of specific fold (a numpy file that is saved in ./data/ by default)
"""
cfg['brains_idx_dir']        = './data/fold0_idx.npy'



"""
'axial', 'sagittal' or 'coronal'. The 'view' has no effect in "prepare_data.py".
All 2D slices and the model will be prepared  with respect to 'view'.
"""
cfg['view']                  = 'axial'



"""
The batch size for training and validating the model
"""
cfg['batch_size']            = 32
cfg['val_batch_size']        = 64



"""
The augmentation parameters.
"""
cfg['hor_flip']              = True
cfg['ver_flip']              = True
cfg['rotation_range']        = 0
cfg['zoom_range']            = 0.



"""
The leraning rate and the number of epochs for training the model
"""
cfg['epochs']                = 100
cfg['lr']                    = 0.008



"""
If True, use process-based threading. "https://keras.io/models/model/"
"""
cfg['multiprocessing']       = False 



"""
Maximum number of processes to spin up when using process-based threading. 
If unspecified, workers will default to 1. If 0, will execute the generator 
on the main thread. "https://keras.io/models/model/"
"""
cfg['workers']               = 5



"""
Whether to use the proposed modifid UNet or the original UNet
"""
cfg['modified_unet']         = True 



"""
The depth of the U-structure 
"""
cfg['levels']                = 3



"""
The number of channels of the first conv
"""
cfg['start_chs']             = 64



"""
If specified, before training, the model weights will be loaded from this path otherwise
the model will be trained from scratch.
"""
cfg['load_model_dir']        = None 

 
