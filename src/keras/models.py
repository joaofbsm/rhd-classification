"""Deep learning models"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv3D, Convolution3D, MaxPooling3D, ZeroPadding3D, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential


def c3d_test(summary=True, backend='tf'):
    """
    C3D model used for development test.
    """

    model = Sequential()
    if backend == 'tf':
        input_shape = (16, 112, 112, 3)  # l, h, w, c
    else:
        input_shape = (3, 16, 112, 112)  # c, l, h, w

    model.add(Convolution3D(64, (3, 3, 3),
                            padding='same', name='conv1',
                            input_shape=input_shape, trainable=False))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1', trainable=False))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3),
                            padding='same', name='conv2', trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2', trainable=False))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3),
                            padding='same', name='conv3a', trainable=False))
    model.add(Convolution3D(256, (3, 3, 3),
                            padding='same', name='conv3b', trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3', trainable=False))
    # 4th layer group
    model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv4a', trainable=False))
    model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv4b', trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4', trainable=False))
    # 5th layer group
    model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv5a', trainable=False))
    model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv5b', trainable=False))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5', trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5', trainable=False))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7', trainable=False))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())

    return model


def c3d(summary=False, backend='tf', learn_visual_features=False):
    """
    Return the Keras model for the C3D architecture.
    """

    model = Sequential()
    if backend == 'tf':
        input_shape = (16, 112, 112, 3)  # l, h, w, c
    else:
        input_shape = (3, 16, 112, 112)  # c, l, h, w

    model.add(Convolution3D(64, (3, 3, 3),
                          padding='same', name='conv1',
                          input_shape=input_shape, trainable=learn_visual_features))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1', trainable=learn_visual_features))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3),
                          padding='same', name='conv2', trainable=learn_visual_features))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2', trainable=learn_visual_features))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3),
                          padding='same', name='conv3a', trainable=learn_visual_features))
    model.add(Convolution3D(256, (3, 3, 3),
                          padding='same', name='conv3b', trainable=learn_visual_features))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3', trainable=learn_visual_features))
    # 4th layer group
    model.add(Convolution3D(512, (3, 3, 3),
                          padding='same', name='conv4a', trainable=learn_visual_features))
    model.add(Convolution3D(512, (3, 3, 3),
                          padding='same', name='conv4b', trainable=learn_visual_features))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4', trainable=learn_visual_features))
    # 5th layer group
    model.add(Convolution3D(512, (3, 3, 3),
                          padding='same', name='conv5a', trainable=learn_visual_features))
    model.add(Convolution3D(512, (3, 3, 3),
                          padding='same', name='conv5b', trainable=learn_visual_features))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5', trainable=learn_visual_features))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5', trainable=learn_visual_features))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())

    return model


def c3d_int_model(model, layer, batch_normalization=False, backend='tf', learn_visual_features=True):
    """
    Update the architecture of a C3D model after the pre-trained weights are loaded. With this function it is possible
    to get the model with any of its layers as the last one.
    """
    
    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w

    int_model = Sequential()

    int_model.add(Convolution3D(64, (3, 3, 3),
                            padding='same', name='conv1', trainable=learn_visual_features,
                            input_shape=input_shape,
                            weights=model.layers[0].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))

    if layer == 'conv1':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1', trainable=learn_visual_features))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))

    if layer == 'pool1':
        return int_model

    # 2nd layer group
    int_model.add(Convolution3D(128, (3, 3, 3),
                            padding='same', name='conv2', trainable=learn_visual_features,
                            weights=model.layers[2].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv2':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2', trainable=learn_visual_features))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'pool2':
        return int_model

    # 3rd layer group
    int_model.add(Convolution3D(256, (3, 3, 3),
                            padding='same', name='conv3a', trainable=learn_visual_features,
                            weights=model.layers[4].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv3a':
        return int_model
    int_model.add(Convolution3D(256, (3, 3, 3),
                            padding='same', name='conv3b', trainable=learn_visual_features,
                            weights=model.layers[5].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv3b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3', trainable=learn_visual_features))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'pool3':
        return int_model

    # 4th layer group
    int_model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv4a', trainable=learn_visual_features,
                            weights=model.layers[7].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv4a':
        return int_model
    int_model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv4b', trainable=learn_visual_features,
                            weights=model.layers[8].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv4b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4', trainable=learn_visual_features))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'pool4':
        return int_model

    # 5th layer group
    int_model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv5a', trainable=learn_visual_features,
                            weights=model.layers[10].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv5a':
        return int_model
    int_model.add(Convolution3D(512, (3, 3, 3),
                            padding='same', name='conv5b', trainable=learn_visual_features,
                            weights=model.layers[11].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'conv5b':
        return int_model
    int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad', trainable=learn_visual_features))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5', trainable=learn_visual_features))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'pool5':
        return int_model

    int_model.add(Flatten())
    # FC layers group
    int_model.add(Dense(4096, name='fc6', trainable=learn_visual_features,
                            weights=model.layers[15].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'fc6':
        return int_model
    if not batch_normalization:
      int_model.add(Dropout(.5))
    int_model.add(Dense(4096, name='fc7', trainable=learn_visual_features,
                            weights=model.layers[17].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('relu'))
    
    if layer == 'fc7':
        return int_model
    if not batch_normalization:
        int_model.add(Dropout(.5))
    int_model.add(Dense(487, name='fc8', trainable=learn_visual_features,
                            weights=model.layers[19].get_weights()))
    if batch_normalization:
        int_model.add(BatchNormalization())
    int_model.add(Activation('softmax'))
    
    if layer == 'fc8':
        return int_model

    return None


def functional_c3d(input_type, learn_visual_features=True):
    """
    C3D network implementation using the Keras Functional API. This implementation is used to create the two armed
    network architecture. The input_type refers to either Doppler or non-Doppler videos.
    """

    input_shape = (16, 112, 112, 3)
    input_layer = Input(shape=input_shape)

    x = Convolution3D(64, (3, 3, 3),
                      padding='same', name='{}_conv1'.format(input_type),
                      input_shape=input_shape, trainable=learn_visual_features)(input_layer)

    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                     padding='valid', name='{}_pool1'.format(input_type), trainable=learn_visual_features)(x)
    # 2nd layer group
    x = Convolution3D(128, (3, 3, 3),
                      padding='same', name='{}_conv2'.format(input_type), trainable=learn_visual_features)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='valid', name='{}_pool2'.format(input_type), trainable=learn_visual_features)(x)
    # 3rd layer group
    x = Convolution3D(256, (3, 3, 3),
                      padding='same', name='{}_conv3a'.format(input_type), trainable=learn_visual_features)(x)
    x = Convolution3D(256, (3, 3, 3),
                      padding='same', name='{}_conv3b'.format(input_type), trainable=learn_visual_features)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='valid', name='{}_pool3'.format(input_type), trainable=learn_visual_features)(x)
    # 4th layer group
    x = Convolution3D(512, (3, 3, 3),
                      padding='same', name='{}_conv4a'.format(input_type), trainable=learn_visual_features)(x)
    x = Convolution3D(512, (3, 3, 3),
                      padding='same', name='{}_conv4b'.format(input_type), trainable=learn_visual_features)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='valid', name='{}_pool4'.format(input_type), trainable=learn_visual_features)(x)
    # 5th layer group
    x = Convolution3D(512, (3, 3, 3),
                      padding='same', name='{}_conv5a'.format(input_type), trainable=learn_visual_features)(x)
    x = Convolution3D(512, (3, 3, 3),
                      padding='same', name='{}_conv5b'.format(input_type), trainable=learn_visual_features)(x)
    x = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='{}_zeropad5'.format(input_type),
                      trainable=learn_visual_features)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='valid', name='{}_pool5'.format(input_type), trainable=learn_visual_features)(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(.5)(x)
    output_layer = Dense(487, activation='softmax', name='fc8')(x)

    return Model(input_layer, output_layer)


def two_armed_c3d(left_model, right_model):
    """
    C3D architecture with two arms. One of the arms will focus in Doppler videos while the other will focus in
    non-Doppler ones.
    """

    left_model_input, left_model_output = left_model.layers[0].input, left_model.layers[-1].output
    right_model_input, right_model_output = right_model.layers[0].input, right_model.layers[-1].output
    concatenated = tf.keras.layers.concatenate([left_model_output, right_model_output])
    x = Dense(4096, activation='relu', name='fc6')(concatenated)
    x = Dropout(.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(.5)(x)
    output_layer = Dense(2, activation='softmax', name='fc8')(x)

    return Model([left_model_input, right_model_input], output_layer)