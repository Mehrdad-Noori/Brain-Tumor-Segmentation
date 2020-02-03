"""
Utilities for real-time multi-thread data generator
"""

import scipy
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical



class CustomDataGenerator(Sequence):
    
    def __init__(self, hdf5_file, brain_idx, batch_size=16, view="axial", mode='train', horizontal_flip=False,
                 vertical_flip=False, rotation_range=0, zoom_range=0., shuffle=True):
        """
        Custom data generator based on Keras Sequance class.
        This implementation enables multiprocessing and on-the-fly data augmentation 
        which will speed up training, especially in the task of brain tumor segmentation
        that suffers from time-consuming data processing.
        
        Parameters
        ----------
        hdf5_file : file.File
            An opend hdf5 file that contains all data.
        brain_idx : array
            The brain indexes corresponing to a specific fold. All of these 
            brain indexes will be use for training and the ones which are 
            not in 'brain_idx' will be used for validation
        batch_size : int
            The number of input/output arrays that will be generated each 
            time. The default is 16.
        view : str
            'axial', 'sagittal' or 'coronal'. The generator will extract
            2D slices and perform normalization with respect to the chosen view.
            The defualt is axial.
        mode : str
            Prepare the DataGenerator for 'train' or 'validation' phase. 
            The default is 'train'.
        horizontal_flip : bool
            Whether to use horizontal flip for data augmentation. The default is False.
        vertical_flip : bool
            Whether to use vertical flip for data augmentation. The default is False.
        rotation_range : float
            Random rotation for data augmentation. The default is 0.
        zoom_range : float
            Random zoom for data augmentation. The default is 0.
        shuffle : bool
            Whether to shuffle data. The default is True. Note that if mode='validation' 
            it will not shufflw tha data.

        """
        
        self.data_storage    = hdf5_file.root.data
        self.truth_storage   = hdf5_file.root.truth   
        
        total_brains         = self.data_storage.shape[0]
        self.brain_idx       = self.get_brain_idx(brain_idx, mode, total_brains)
        self.batch_size      = batch_size
        
        if view == 'axial':
            self.view_axes = (0, 1, 2, 3)            
        elif view == 'sagittal': 
            self.view_axes = (2, 1, 0, 3)
        elif view == 'coronal':
            self.view_axes = (1, 2, 0, 3)            
        else:
            ValueError('unknown input view => {}'.format(view))
            
        self.mode            = mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.rotation_range  = rotation_range       
        self.zoom_range      = [1 - zoom_range, 1 + zoom_range]
        self.shuffle         = shuffle
        self.data_shape      = tuple(np.array(self.data_storage.shape[1:])[np.array(self.view_axes)])
        
        print('Using {} out of {} brains'.format(len(self.brain_idx), total_brains), end=' ')
        print('({} out of {} 2D slices)'.format(len(self.brain_idx) * self.data_shape[0], total_brains * self.data_shape[0]))
        print('the generated data shape in "{}" view: {}'.format(view, str(self.data_shape[1:])))
        print('-----'*10)

        self.on_epoch_end()
        
        

    @staticmethod
    def get_brain_idx(brain_idx, mode, total_brains):
        
        """
        Getting the brain indexes that will be used by the generator.
        if mode=='train' => the original indexes will be used (because we built these
        npy files based on training indexes in 'prepare_data.py' for k-fold, remember? :)
        if mode=='validation' => the indexes which are not in the brain_idx will
        be used.
        

        """            
        if mode=='validation':
            brain_idx       = np.array([i for i in np.arange(total_brains) if i not in brain_idx])
            print('DataGenerator is preparing for validation mode ...') 
        elif mode=='train':
            brain_idx       = brain_idx
            print('DataGenerator is preparing for training mode ...')
        else:
            raise ValueError('unknown "{}" mode'.format(mode))
            
        return brain_idx


    def __len__(self):
        return int(np.floor( len(self.indexes) / self.batch_size))
    
    
    def __getitem__(self, index):

        # Generate indexes of the batch
        idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X_batch, Y_batch = self.data_load_and_preprocess(idx)

        return X_batch, Y_batch

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        tmp=[]
        for i in self.brain_idx:
            for j in range(self.data_shape[0]):
                tmp.append((i,j))
        self.indexes = tmp
            
        if self.mode=='train' and self.shuffle:
            np.random.shuffle(self.indexes)
            
            
    def data_load_and_preprocess(self, idx):
        """
        Generates data containing batch_size samples
        """
        slice_batch = []
        label_batch = []

        # Generate data
        for i in idx:
            brain_number     = i[0]
            slice_number     = i[1]
            slice_, label_   = self.read_data(brain_number, slice_number)
            slice_           = self.normalize_modalities(slice_)
            slice_and_label  = np.concatenate((slice_, label_) , axis=-1)
            params           = self.get_random_transform()
            slice_and_label  = self.apply_transform(slice_and_label, params)
            slice_           = slice_and_label[...,:4]
            label_           = slice_and_label[..., 4]
            label_           = to_categorical(label_, 4) 
            
            slice_batch.append(slice_)
            label_batch.append(label_)
            
        return np.array(slice_batch), np.array(label_batch)
    
    
    
    def read_data(self, brain_number, slice_number):
        
        """
        Reads data from table with respect to the 'view'
        
        """
        
        slice_    = self.data_storage[brain_number].transpose(self.view_axes)[slice_number]
        label_    = self.truth_storage[brain_number].transpose(self.view_axes[:3])[slice_number]
        label_    = np.expand_dims(label_, axis=-1)
        
        return slice_, label_ 
        
    
    def normalize_slice(self, slice):
        
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
        
        
    def normalize_modalities(self, Slice): 
        
        """
        Performs normalization on each modalities of input
        """

        normalized_slices = np.zeros_like(Slice).astype(np.float32)
        for slice_ix in range(4):
            normalized_slices[..., slice_ix] = self.normalize_slice(Slice[..., slice_ix])
    
        return normalized_slices  
    

    def flip_axis(self, x, axis):
        
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
    
    def apply_transform(self, x, transform_parameters):
        
        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                           transform_parameters.get('tx', 0),
                           transform_parameters.get('ty', 0),
                           transform_parameters.get('shear', 0),
                           transform_parameters.get('zx', 1),
                           transform_parameters.get('zy', 1),
                           row_axis=0,
                           col_axis=1,
                           channel_axis=2)
        if transform_parameters.get('flip_horizontal', False):
            x = self.flip_axis(x, 1)
        if transform_parameters.get('flip_vertical', False):
            x = self.flip_axis(x, 0)            
        return x
        
    def get_random_transform(self):
    
        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range,self.rotation_range)    
        else:
            theta = 0            
 
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0],self.zoom_range[1], 2)
            
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip    
        flip_vertical   = (np.random.random() < 0.5) * self.vertical_flip
        
        transform_parameters = {'flip_horizontal': flip_horizontal,
                                'flip_vertical':flip_vertical,
                                'theta': theta, 
                                'zx': zx, 
                                'zy': zy}
    
        return transform_parameters        
        
"""
The two following functions are from ImageDataGenerator class of keras.
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""
    
def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x

        

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix



        