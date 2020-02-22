import os
import tables
import numpy as np
from config import cfg
from model import unet_model
from data_generator import CustomDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

def train_model(hdf5_dir, brains_idx_dir, view, modified_unet=True, batch_size=16, val_batch_size=32,
                lr=0.01, epochs=100, hor_flip=False, ver_flip=False, zoom_range=0.0, save_dir='./save/',
                start_chs=64, levels=3, multiprocessing=False, load_model_dir=None):
    """

    The function that builds/loads UNet model, initializes the data generators for training and validation, and finally 
    trains the model.

    """
    # preparing generators
    hdf5_file        = tables.open_file(hdf5_dir, mode='r+')
    brain_idx        = np.load(brains_idx_dir)
    datagen_train    = CustomDataGenerator(hdf5_file, brain_idx, batch_size, view, 'train',
                                    hor_flip, ver_flip, zoom_range, shuffle=True)
    datagen_val      = CustomDataGenerator(hdf5_file, brain_idx, val_batch_size, view, 'validation', shuffle=False)
    
    # add callbacks    
    save_dir     = os.path.join(save_dir, '{}_{}'.format(view, os.path.basename(brains_idx_dir)[:5]))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    logger       = CSVLogger(os.path.join(save_dir, 'log.txt'))
    checkpointer = ModelCheckpoint(filepath = os.path.join(save_dir, 'model.hdf5'), verbose=1, save_best_only=True)
    tensorboard  = TensorBoard(os.path.join(save_dir, 'tensorboard'))
    callbacks    = [logger, checkpointer, tensorboard]        
    
    # building the model
    model_input_shape = datagen_train.data_shape[1:]
    model             = unet_model(model_input_shape, modified_unet, lr, start_chs, levels)
    # training the model
    model.fit_generator(datagen_train, epochs=epochs, use_multiprocessing=multiprocessing, 
                        callbacks=callbacks, validation_data = datagen_val)


   
if __name__ == '__main__':
    
    
    train_model(cfg['hdf5_dir'], cfg['brains_idx_dir'], cfg['view'], cfg['modified_unet'], cfg['batch_size'], 
                cfg['val_batch_size'], cfg['lr'], cfg['epochs'], cfg['hor_flip'], cfg['ver_flip'], cfg['zoom_range'], 
                cfg['save_dir'], cfg['start_chs'], cfg['levels'], cfg['multiprocessing'], 
                cfg['load_model_dir'])
    
    
    
    
    
