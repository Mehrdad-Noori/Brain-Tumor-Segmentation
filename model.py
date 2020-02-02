import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, PReLU, UpSampling2D, concatenate , Reshape, Dense, Permute, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, add, GaussianNoise, BatchNormalization, multiply
from tensorflow.keras.optimizers import SGD
from loss import custom_loss
K.set_image_data_format("channels_last")



def unet_model(input_shape, modified_unet=True, learning_rate=0.01, start_channel=64, 
               number_of_levels=3, inc_rate=2, output_channels=4, saved_model_dir=None):
    """
    Builds UNet model
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (height, width, channel)
    modified_unet : bool
        Whether to use modified UNet or the original UNet
    learning_rate : float
        Learning rate for the model. The default is 0.01.
    start_channel : int
        Number of channels of the first conv. The default is 64.
    number_of_levels : int
        The depth size of the U-structure. The default is 3.
    inc_rate : int
        Rate at which the conv channels will increase. The default is 2.
    output_channels : int
        The number of output layer channels. The default is 4
    saved_model_dir : str
        If spesified, the model weights will be loaded from this path. The default is None.

    Returns
    -------
    model : keras.model
        The created keras model with respect to the input parameters

    """

        
    input_layer = Input(shape=input_shape, name='the_input_layer')

    if modified_unet:
        x = GaussianNoise(0.01, name='Gaussian_Noise')(input_layer)
        x = Conv2D(64, 2, padding='same')(x)
        x = level_block_modified(x, start_channel, number_of_levels, inc_rate)
        x = BatchNormalization(axis = -1)(x)
        x = PReLU(shared_axes=[1, 2])(x)
    else: 
        x = level_block(input_layer, start_channel, number_of_levels, inc_rate)

    x            = Conv2D(output_channels, 1, padding='same')(x)
    output_layer = Activation('softmax')(x)
    
    model        = Model(inputs = input_layer, outputs = output_layer)

    if modified_unet:
        print("The modified UNet was built!")
    else:
        print("The original UNet was built!")

    if saved_model_dir:
        model.load_weights(saved_model_dir)
        print("the model weights were successfully loaded!")
            
    sgd = SGD(lr=learning_rate, momentum=0.9, decay=0)
    model.compile(optimizer=sgd, loss=custom_loss)
    
    return model


def se_block(x, ratio=16):
    
    """
    creates a squeeze and excitation block
    https://arxiv.org/abs/1709.01507
    
    Parameters
    ----------
    x : tensor
        Input keras tensor
    ratio : int
        The reduction ratio. The default is 16.

    Returns
    -------
    x : tensor
        A keras tensor
    """
 

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([x, se])
    return x


def level_block(x, dim, level, inc):
    
    if level > 0:
        m = conv_layers(x, dim)
        x = MaxPool2D(pool_size=(2, 2))(m)
        x = level_block(x,int(inc*dim), level-1, inc)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(dim, 2, padding='same')(x)
        m = concatenate([m,x])
        x = conv_layers(m, dim)
    else:
        x = conv_layers(x, dim)
    return x


def level_block_modified(x, dim, level, inc):
    
    if level > 0:
        m = res_block(x, dim, encoder_path=True)##########
        x = Conv2D(int(inc*dim), 2, strides=2, padding='same')(m)
        x = level_block_modified(x, int(inc*dim), level-1, inc)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(dim, 2, padding='same')(x)

        m = concatenate([m,x])
        m = se_block(m, 8)
        x = res_block(m, dim, encoder_path=False)
    else:
        x = res_block(x, dim, encoder_path=True) #############
    return x


def conv_layers(x, dim):

    x = Conv2D(dim, 3, padding='same')(x)
    x = Activation("relu")(x)

    x = Conv2D(dim, 3, padding='same')(x)
    x = Activation("relu")(x)

    return x

def res_block(x, dim, encoder_path=True):

    m = BatchNormalization(axis = -1)(x)
    m = PReLU(shared_axes = [1, 2])(m)
    m = Conv2D(dim, 3, padding='same')(m)

    m = BatchNormalization(axis = -1)(m)
    m = PReLU(shared_axes = [1, 2])(m)
    m = Conv2D(dim, 3, padding='same')(m)

    if encoder_path:
        x = add([x, m])
    else:
        x = Conv2D(dim, 1, padding='same', use_bias=False)(x)
        x = add([x,m])
    return  x

