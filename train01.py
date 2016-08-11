#!/usr/bin/env python

import time
import numpy as np
import os, sys

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

sys.path.append(os.path.expanduser("~/torchio")); import torchio


#----------------------------------------------------------------------

outputDir = "plots-" + time.strftime("%Y-%m-%d-%H%M")


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

def datasetLoadFunction(dataDesc, size, cuda):
    # returns trainData, testData

    assert dataDesc['inputDataIsSparse'], "non-sparse input data is currently not supported"

    retval = []
    
    for filesKey, sizeKey in (
        ('train_files', 'trsize'),
        ('test_files', 'tesize')):

        # get the number of events to be read from each file
        thisSize = dataDesc.get(sizeKey, None)
        
        for fname in dataDesc[filesKey]:
            print "opening",fname
            thisData = torchio.read(fname)

            # typical structure:
            # {'y': <torchio.torch.FloatTensor instance at 0x7f05052000e0>, 
            #  'X': {
            #        'y': <torchio.torch.IntTensor instance at 0x7f05054abcf8>, 
            #        'x': <torchio.torch.IntTensor instance at 0x7f05054abb90>, 
            #        'energy': <torchio.torch.FloatTensor instance at 0x7f05054ab998>, 
            #        'firstIndex': <torchio.torch.IntTensor instance at 0x7f05054abe60>, 
            #        'numRecHits': <torchio.torch.IntTensor instance at 0x7f05054ab878>}, 
            # 'mvaid': <torchio.torch.FloatTensor instance at 0x7f0505200200>, 
            # 'weight': <torchio.torch.FloatTensor instance at 0x7f05054abfc8>}
        

    return retval


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 2, "usage: " + os.path.basename(sys.argv[0]) + " modelFile.py dataFile.py"

execfile(ARGV[0])
execfile(ARGV[1])
#----------

havePylab = False

try:
    import pylab
    havePylab = True
except ImportError:
    pass

if havePylab:
    pylab.close('all')

print "loading data"

cuda = True
trainData, trsize = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], cuda)
testData,  tesize = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], cuda)

# convert labels from -1..+1 to 0..1 for cross-entropy loss
# must clone to assign

def cloneFunc(data):
    return dict( [( key, np.copy(value) ) for key, value in data.items() ])

    ### retval = {}
    ### for key, value in data.items():
    ###     retval[key] = np.

trainData = cloneFunc(trainData); testData = cloneFunc(testData)


# TODO: normalize these to same weight for positive and negative samples
trainWeights = np.ones(trainData['labels'].shape)
testWeights  = np.ones(testData['labels'].shape)


print "building model"
model = makeModel()

model.add(Activation('sigmoid'))

# see e.g. https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py#L81

### model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
model.compile(loss='binary_crossentropy',
              # optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),

              optimizer = 'adam',

              class_mode='binary')


print 'starting training at', time.asctime()

trainLossHistory = LossHistory()

testAuc = ROCModelCheckpoint('./roc.h5', testData['input'], testData['labels'], testWeights, verbose=True)

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

if havePylab:    
    makePlots()
    pylab.show()
    
