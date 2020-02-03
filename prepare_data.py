import os
import tables
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from config import cfg

def read_brain(brain_dir, mode='train', x0=42, x1=194, y0=29, y1=221, z0=2, z1=146):

    """
    A function that reads and crops a brain modalities (nii.gz format)
    
    Parameters
    ----------
    brain_dir : string
        The path to a folder that contains MRI modalities of a specific brain
    mode : string
        'train' or 'validation' mode. The default is 'train'.
    x0, x1, y0, y1, z0, z1 : int
        The coordinates to crop brain volumes. For example, a brain volume with the 
        shape [x,y,z,modalites] is cropped [x0:x1, y0:y1, z0:z1, :] to have the shape
        [x1-x0, y1-y0, z1-z0, modalities]. One can calculate the x0,x1,... by calculating
        none zero pixels through dataset. Note that the final three shapes must be divisible
        by the network downscale rate.
        
    Returns
    -------
    all_modalities : array
        The cropped modalities (+ gt if mode='train')
    brain_affine : array
        The affine matrix of the input brain volume
    brain_name : str
        The name of the input brain volume

    """
    
    brain_dir = os.path.normpath(brain_dir)
    flair     = glob(os.path.join(brain_dir, '*_flair*.nii.gz'))
    t1        = glob(os.path.join(brain_dir, '*_t1*.nii.gz'))
    t1ce      = glob(os.path.join(brain_dir, '*_t1ce*.nii.gz'))
    t2        = glob(os.path.join(brain_dir, '*_t2*.nii.gz'))
    
    if mode=='train':
        gt             = glob( os.path.join(brain_dir, '*_seg*.nii.gz'))
        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0], gt[0]]
        
    elif mode=='validation':
        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0]]   
    
    all_modalities = []    
    for modality in modalities_dir:      
        nifti_file   = nib.load(modality)
        brain_numpy  = np.asarray(nifti_file.dataobj)    
        all_modalities.append(brain_numpy)
        
    # all modalities have the same affine, so we take one of them (the last one in this case),
    # affine is just saved for preparing the predicted nii.gz file in the future.       
    brain_affine   = nifti_file.affine
    all_modalities = np.array(all_modalities)
    all_modalities = np.rint(all_modalities).astype(np.int16)
    all_modalities = all_modalities[:, x0:x1, y0:y1, z0:z1]
    # to fit keras channel last model
    all_modalities = np.transpose(all_modalities) 
    # tumor grade + name
    brain_name     = os.path.basename(os.path.split(brain_dir)[0]) + '_' + os.path.basename(brain_dir) 

    return all_modalities, brain_affine, brain_name
    


def create_table(dataset_dir, table_data_shape, save_dir, crop_coordinates, data_channels, k_fold=None):
    
    """
    Reads and saves all brain volumes into a single table file. 
    
    Parameters
    ----------
    dataset_dir : 
        The path to all brain volumes (ex: suppose we have a folder 'BraTS2019' that 
        contains two HGG and LGG folders each of which contains some folders so:
        dataset_dir="./BraTS2019/*/*")
    table_data_shape : tuple
        A tuple which shows the final brain volume shape in the table
    data_channels : int
        Number of data channels/modalities
    save_dir : str
        The path to save table.
    crop_coordinates : dict
    k_fold : int
        k-fold cross-validation
        if specified, k .npy files will be saved. Each of these files shows the indexes of 
        brain volumes in that fold, which will be used for training the model.

    Returns
    -------
    None

    """
    
    all_brains_dir = glob(dataset_dir)
    all_brains_dir.sort()
    
    hdf5_file    = tables.open_file(os.path.join(save_dir + 'data.hdf5'), mode='w')
    filters      = tables.Filters(complevel=5, complib='blosc')
    data_shape   = tuple([0] + list(table_data_shape) + [data_channels])
    truth_shape  = tuple([0] + list(table_data_shape))
    affine_shape = tuple([0] + [4, 4])
    
    data_storage   = hdf5_file.create_earray(hdf5_file.root, 'data', tables.UInt16Atom(), shape=data_shape,
                                               filters=filters, expectedrows=len(all_brains_dir))
    truth_storage  = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                                filters=filters, expectedrows=len(all_brains_dir))
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=affine_shape,
                                                filters=filters, expectedrows=len(all_brains_dir))
     
    brain_names = []
    for brain_dir in tqdm(all_brains_dir):
        all_modalities, brain_affine, brain_name = read_brain(brain_dir, mode='train', **crop_coordinates)
        brain    = all_modalities[..., :4]
        gt       = all_modalities[..., -1]
        
        # in BraTS 2017, 2018, 2019 there is no '3' label!
        gt[gt==4]  = 3    
        brain_names.append(brain_name)   
        data_storage.append(brain[np.newaxis,...])
        truth_storage.append(gt[np.newaxis,...])
        affine_storage.append(brain_affine[np.newaxis,...])
        
    hdf5_file.create_array(hdf5_file.root, 'brain_names', obj=brain_names)
    hdf5_file.close()
         
    if k_fold:
        validation_split = (1/k_fold) # this equal to 5-fold validation
        all_HGG_names = [i for i in brain_names if 'HGG' in i]
        all_LGG_names = [i for i in brain_names if 'LGG' in i]
              
        np.random.seed(100)
        np.random.shuffle(all_HGG_names)
        np.random.shuffle(all_LGG_names)
              
        HGG_val_size = int(validation_split * len(all_HGG_names))
        LGG_val_size = int(validation_split * len(all_LGG_names))
        
        for fold in range(k_fold):
            chosen_HGG_val = all_HGG_names[fold*HGG_val_size:(fold+1)*HGG_val_size]
            chosen_LGG_val = all_LGG_names[fold*LGG_val_size:(fold+1)*LGG_val_size]
        
            chosen_HGG_train = [i for i in all_HGG_names if i not in chosen_HGG_val]
            chosen_LGG_train = [i for i in all_LGG_names if i not in chosen_LGG_val]
        
            # saving train_idx is enough
            train = chosen_HGG_train + chosen_LGG_train    
            train_idx = [brain_names.index(i) for i in train]    
            train_idx.sort()
        
            np.save(os.path.join(save_dir, 'fold{}_idx.npy'.format(fold)), train_idx)
    
   
    
if __name__ == '__main__':
      
    create_table(cfg['data_dir'], cfg['table_data_shape'], cfg['save_data_dir'], 
                 cfg['crop_coord'], cfg['data_channels'], cfg['k_fold'])
    

