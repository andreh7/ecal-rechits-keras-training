#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


#----------------------------------------------------------------------


def makeModel(input_shape):
    model = Sequential()

    # input shape: (1 color) x (7 x 23) images
    # nn.SpatialConvolutionMM(1 -> 64, 5x5, 1,1, 2,2)
    model.add(Convolution2D(64,
                            5,
                            5,
                            border_mode = 'same',
                            input_shape = input_shape,
                            ))

    # input shape: 64 x 7 x 23
    # nn.ReLU
    model.add(Activation('relu'))

    # input shape: 64 x 7 x 23
    # nn.SpatialMaxPooling(2,2,2,2)
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2), border_mode = 'same'))
    
    # input shape: 64 x 4 x 12
    # nn.SpatialConvolutionMM(64 -> 64, 5x5, 1,1, 2,2)
    #
    # this somehow works in Torch but keras complains that the convolution
    # window size (5x5) is too large for the input (4x12), so we reduced
    # it to 3x3 here
    model.add(Convolution2D(64,
                            3,
                            3,
                            border_mode = 'same',
                            ))

    # input shape: 64 x 4 x 12
    # nn.ReLU
    model.add(Activation('relu'))

    # input shape: 64 x 4 x 12
    # nn.SpatialMaxPooling(2,2,2,2,0.5,0.5)
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2), border_mode = 'same'))

    # input shape: 64 x 2 x 6
    # nn.View
    model.add(Flatten())

    # input shape: 768
    # nn.Dropout(0.500000)
    model.add(Dropout(0.5))

    # input shape: 768
    # nn.Linear(320 -> 128)
    model.add(Dense(128))
    
    # nn.ReLU
    model.add(Activation('relu'))
        
    # nn.Linear(128 -> 1)
    model.add(Dense(1))
    
    # nn.Tanh
    # model.add(Activation('tanh'))    # output range -1..1
    model.add(Activation('sigmoid')) # output range 0..1

    return model
#----------------------------------------------------------------------
