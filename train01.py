#!/usr/bin/env python

import time
import numpy as np
import os

# Keras complains for the convolutional layers
# that border_mode 'same' is not supported with the
# Theano backend (see e.g. https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py#L634 ) 
#
# but from here: https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip-installation it
# looks like on MacOS, only CPU is supported, not the GPU...
#
# from here: https://github.com/fchollet/keras/wiki/Keras,-now-running-on-TensorFlow#performance
# it also looks like Theano is more performant in some cases (but much worse in others)
#
# edit the 'backend' parameter in ~/.keras/keras.json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from ROCModelCheckpoint import ROCModelCheckpoint

#----------------------------------------------------------------------

# only use e.g. every 10'th element
strideSize = 40
    

outputDir = "plots-" + time.strftime("%Y-%m-%d-%H%M")

#----------------------------------------------------------------------


def makeModel():
    model = Sequential()

    # input shape: (1 color) x (7 x 23) images
    # nn.SpatialConvolutionMM(1 -> 64, 5x5, 1,1, 2,2)
    model.add(Convolution2D(64,
                            5,
                            5,
                            border_mode = 'same',
                            input_shape = (1, 7, 23),
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
    model.add(Activation('tanh'))

    return model


#----------------------------------------------------------------------
# see http://keras.io/callbacks/ for the Callback interface

class LossHistory(Callback):

    #----------------------------------------
    def on_train_begin(self, logs={}):
        self.losses = []

    #----------------------------------------
        
    def on_batch_end(self, batch, logs={}):
        # logs has the following keys:
        #   'acc', 'loss', 'batch', 'size'

        self.losses.append(logs.get('loss'))


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

import pylab
pylab.close('all')

print "loading data"

trainData = np.load("gjet-20-40-train.npz")
testData = np.load("gjet-20-40-test.npz")

# TODO: normalize these to same weight for positive and negative samples
trainWeights = np.ones(trainData['labels'].shape)
testWeights  = np.ones(testData['labels'].shape)

print "building model"
model = makeModel()


# see e.g. https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py#L81

### model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
model.compile(loss='binary_crossentropy',
              # optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),

              optimizer = 'adam',

              class_mode='binary')


print 'starting training at', time.asctime()

trainLossHistory = LossHistory()

testAuc = ROCModelCheckpoint('./roc.h5', testData['input'][::strideSize], testData['labels'][::strideSize], testWeights[::strideSize], verbose=True)

# see https://github.com/fchollet/keras/blob/d2f7593a35c4d2df8d8b4f434f20097448595cb0/keras/callbacks.py#L220
# for definition of ModelCheckpoint
valLoss = ModelCheckpoint('./logloss.h5', monitor = 'val_loss', verbose = True, save_best_only = True)

callbacks = [
            EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
            valLoss,
            testAuc,
            trainLossHistory,
            ]

# history will not be set if one presses CTRL-C...
history = None
    
try:
	history = model.fit(
            trainData['input'][::strideSize], trainData['labels'][::strideSize],
            sample_weight = trainWeights[::strideSize],
            # sample_weight=np.power(weights, 0.7))
            
            batch_size = 3*32,
            # nb_epoch = 20,

            # DEBUG
            nb_epoch = 2,


            show_accuracy = True, 

            shuffle = True, # shuffle at each epoch (but this is the default)
            
            validation_data = (testData['input'][::strideSize], testData['labels'][::strideSize]),
            callbacks = callbacks,
            )

except KeyboardInterrupt:
	print 'interrupted'

#--------------------
    
def makePlots():

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    import pylab

    # training loss
    pylab.figure()
    pylab.plot(trainLossHistory.losses)
    pylab.xlabel('iteration')
    pylab.ylabel('training loss')
    pylab.title('train loss')
    pylab.savefig(outputDir + "/" + "train-loss.png")

    # test AUCs
    pylab.figure()
    pylab.plot(testAuc.aucs)
    pylab.xlabel('epoch')
    pylab.ylabel('test set AUC')
    pylab.title('test AUC')

    pylab.savefig(outputDir + "/" + "test-auc.png")

    print "saved plots to",outputDir
    
#--------------------
    
makePlots()
