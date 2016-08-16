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

sys.path.insert(0, os.path.expanduser("~/keras"))
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from ROCModelCheckpoint import ROCModelCheckpoint

sys.path.append(os.path.expanduser("~/torchio")); import torchio


#----------------------------------------------------------------------

outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")


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

class EpochStartBanner(Callback):
    # prints a banner when a new training epoch is started

    def __init__(self, logfile = None):
        self.logfile = logfile

    def on_epoch_begin(self, epoch, logs={}):
        
        fouts = [ sys.stdout ]
        if self.logfile != None:
            fouts.append(self.logfile)

        nowStr = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for fout in fouts:

            print >> fout, "----------------------------------------"
            print >> fout, "starting epoch %d at" % (epoch + 1), nowStr
            print >> fout, "----------------------------------------"

            fout.flush()

#----------------------------------------------------------------------

class TrainingTimeMeasurement(Callback):
    # use this as first callback (assuming no other callbacks do a significant
    # amount of computation before the start of the batch)

    def __init__(self, numTrainingSamples, logfile = None):
        super(Callback, self).__init__()

        self.numTrainingSamples = numTrainingSamples

        self.fouts = [ sys.stdout ]
        if logfile != None:
            self.fouts.append(logfile)


    def on_epoch_begin(self, epoch, logs={}):
        self.epochStartTime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        deltaT = time.time() - self.epochStartTime

        for fout in self.fouts:
            # TODO: need to know number of training samples
            print >> fout, "time to learn 1 sample: %.3f ms" % ( deltaT / self.numTrainingSamples * 1000.0)
            print >> fout, "time to train entire batch: %.2f min" % (deltaT / 60.0)
            fout.flush()

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
trainWeights = trainData['weights']
testWeights  = testData['weights']

#----------
print "building model"
model = makeModel()

model.add(Activation('sigmoid'))

#----------
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

logfile = open(os.path.join(outputDir, "train.log"), "w")

#----------
# write out BDT/MVA id labels (for performance comparison)
#----------
for name, weights, label, output in (
    ('train', trainWeights, trainData['labels'], trainData['mvaid']),
    ('test',  testWeights,  testData['labels'],  testData['mvaid']),
    ):
    np.savez(os.path.join(outputDir, "roc-data-%s-mva.npz" % name),
             weight = weights,
             output = output,
             label = label)

#----------

print "----------"
print "model:"
print model.summary()
print "----------"
print "the model has",model.count_params(),"parameters"

print >> logfile,"----------"
print >> logfile,"model:"
model.summary(file = logfile)
print >> logfile,"----------"
print >> logfile, "the model has",model.count_params(),"parameters"
logfile.flush()


# see e.g. https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py#L81

### model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
model.compile(loss='binary_crossentropy',
              # optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),

              optimizer = 'adam',

              )


print 'starting training at', time.asctime()

trainLossHistory = LossHistory()

trainAuc = ROCModelCheckpoint(outputDir, 'train', trainData['input'], trainData['labels'], trainWeights, verbose=True, logFile = logfile)
testAuc  = ROCModelCheckpoint(outputDir, 'test',  testData['input'],  testData['labels'],  testWeights,  verbose=True, logFile = logfile)

callbacks = [
            TrainingTimeMeasurement(len(trainData['labels']), logfile),
            EpochStartBanner(logfile),
            trainAuc,
            testAuc,
            trainLossHistory,
            ]

# history will not be set if one presses CTRL-C...
history = None
    
try:
    history = model.fit(
        trainData['input'], trainData['labels'],
        sample_weight = trainWeights,
        # sample_weight=np.power(weights, 0.7))
        
        batch_size = 32,
        nb_epoch = 1000,
        
        # show_accuracy = True, 
        
        shuffle = True, # shuffle at each epoch (but this is the default)
        
        validation_data = (testData['input'], testData['labels']),
        callbacks = callbacks,
        )

except KeyboardInterrupt:
    print
    print 'interrupted'

#--------------------
    
def makePlots():
    
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

