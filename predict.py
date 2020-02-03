import os
import numpy as np
import nibabel as nib
from glob import glob
from tensorflow.keras.models import load_model


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
        The coordinates to crop 3D brain volume. For example, a brain volume with the 
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
    flair     = glob( os.path.join(brain_dir, '*_flair*.nii.gz'))
    t1        = glob( os.path.join(brain_dir, '*_t1*.nii.gz'))
    t1ce      = glob( os.path.join(brain_dir, '*_t1ce*.nii.gz'))
    t2        = glob( os.path.join(brain_dir, '*_t2*.nii.gz'))
    
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
    
    

def normalize_slice(slice):
    
    """
    Removes 1% of the top and bottom intensities and perform
    normalization on the input 2D slice.
    """
    
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    if np.std(slice)==0:
        return slice
    else:
        slice = (slice - np.mean(slice)) / np.std(slice)
        return slice
    

def normalize_volume(input_volume):
    
    """
    Perform a slice-based normalization on each modalities of input volume.
    """
    normalized_slices = np.zeros_like(input_volume).astype(np.float32)
    for slice_ix in range(4):
        normalized_slices[slice_ix] = input_volume[slice_ix]
        for mode_ix in range(input_volume.shape[1]):
            normalized_slices[slice_ix][mode_ix] = normalize_slice(input_volume[slice_ix][mode_ix])

    return normalized_slices    


def save_predicted_results(prediction, brain_affine, view, output_dir,  z_main=155, z0=2, z1=146, y_main=240, y0=29, y1=221, x_main=240, x0=42, x1=194):
    
    """
    Save the segmented results into a .nii.gz file, so that it can be uploaded to the BraTS server.
    Note that to correctly save the segmented brains, it is necessery to set x0, x1, ... correctly.
    
    Parameters
    ----------
    prediction : array
        The predictred brain.
    brain_affine : array
        The affine matrix of the predicted brain volume
    view : str
        'axial', 'sagittal' or 'coronal'. The 'view' is needed to reconstruct output axes.
    output_dir : str
        The path to save .nii.gz file.


    """
    
    prediction = np.argmax(prediction, axis=-1).astype(np.uint16)            
    prediction[prediction==3] = 4
    
    if view=="axial":
        prediction    = np.pad(prediction, ((z0, z_main-z1), (y0, y_main-y1), (x0, x_main-x1)), 'constant')
        prediction    = prediction.transpose(2,1,0)
    elif view=="sagital":
        prediction    = np.pad(prediction, ((x0, x_main-x1), (y0, y_main-y1), (z0 , z_main-z1)), 'constant')
    elif view=="coronal":
        prediction    = np.pad(prediction, ((y0, y_main-y1), (x0, x_main-x1), (z0 , z_main-z1)), 'constant')
        prediction    = prediction.transpose(1,0,2)
    #
    prediction_ni    = nib.Nifti1Image(prediction, brain_affine)
    prediction_ni.to_filename(output_dir+ '.nii.gz')






if __name__ == '__main__':
       
    val_data_dir       = '/path/to/data/*'
    view               = 'axial'
    saved_model_dir    = '/path/to/a/trained/model.hdf5'  #ex './save/axial_fold0/model.hdf5'
    save_pred_dir      = './predict/'
    batch_size         = 32

    
    if not os.path.isdir(save_pred_dir):
        os.mkdir(save_pred_dir)
       
    all_brains_dir = glob(val_data_dir)
    all_brains_dir.sort()
    
    if view == 'axial':
        view_axes = (0, 1, 2, 3)            
    elif view == 'sagittal': 
        view_axes = (2, 1, 0, 3)
    elif view == 'coronal':
        view_axes = (1, 2, 0, 3)            
    else:
        ValueError('unknown input view => {}'.format(view))
    
    
    model        = load_model(saved_model_dir, compile=False)
    for brain_dir in all_brains_dir:    
        if os.path.isdir(brain_dir):
            print("Volume ID: ", os.path.basename(brain_dir))
            all_modalities, brain_affine, _ = read_brain(brain_dir, mode='validation')
            all_modalities                  = all_modalities.transpose(view_axes)
            all_modalities                  = normalize_volume(all_modalities)
            prediction                      = model.predict(all_modalities, batch_size=batch_size, verbose=1)
            output_dir                      = os.path.join(save_pred_dir, os.path.basename(brain_dir))
            save_predicted_results(prediction, brain_affine, view, output_dir)
            
            
