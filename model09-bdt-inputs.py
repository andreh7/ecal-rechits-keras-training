#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
# in future versions
# from keras.layers.advanced_activations import PRelU

import numpy as np

#----------------------------------------------------------------------

isBarrel = True

if globals().has_key('selectedVariables'):
    ninputs = len(selectedVariables)
else:
    if isBarrel:
        ninputs = 12
    else:
        ninputs = 13

# set to None to disable the dropout layer
dropOutProb = 0.5

# default parameters
numHiddenLayers = 3
nodesPerHiddenLayer = ninputs * 2

# put a dropout layer after each layer, not only at the end
dropOutPerLayer = False

nonlinearity = Activation('relu')

#----------------------------------------
modelParams = dict(
    # maxGradientNorm = 3.3, # typically 0.99 percentile of the gradient norm before diverging
    )

#----------------------------------------

def makeModelHelper(numHiddenLayers, nodesPerHiddenLayer):

    # 2-class problem
    noutputs = 1

    # 13 input variables
    #   phoIdInput :
    #     {
    #       s4 : FloatTensor - size: 1299819
    #       scRawE : FloatTensor - size: 1299819
    #       scEta : FloatTensor - size: 1299819
    #       covIEtaIEta : FloatTensor - size: 1299819
    #       rho : FloatTensor - size: 1299819
    #       pfPhoIso03 : FloatTensor - size: 1299819
    #       phiWidth : FloatTensor - size: 1299819
    #       covIEtaIPhi : FloatTensor - size: 1299819
    #       etaWidth : FloatTensor - size: 1299819
    #       esEffSigmaRR : FloatTensor - size: 1299819
    #       r9 : FloatTensor - size: 1299819
    #       pfChgIso03 : FloatTensor - size: 1299819
    #       pfChgIso03worst : FloatTensor - size: 1299819
    #     }
    
    # size of minibatch
    batchSize = 32

    # how many minibatches to unpack at a time
    # and to store in the GPU (to have fewer
    # data transfers to the GPU)
    # batchesPerSuperBatch = math.floor(6636386 / batchSize)
    
    model = Sequential()

    for i in range(numHiddenLayers):

        isLastLayer = i == numHiddenLayers - 1

        #----------
        # special treatment needed for first layer
        #----------
        if i == 0:
            input_dim = ninputs
        else:
            # autodetect
            input_dim = None

        #----------

        if not isLastLayer:

            import copy
            thisNonlinearity = copy.deepcopy(nonlinearity)
            num_units = nodesPerHiddenLayer

        else:
            # sigmoid at output
            thisNonlinearity = Activation('sigmoid')

            num_units = 1

        # set the name of the activation layer by hand 
        # (necessary after cloning)
        thisNonlinearity.name = "activation_%d" % (i + 1)

        if dropOutProb != None:
            if isLastLayer or dropOutPerLayer and i > 0:
                # add a dropout layer at the end
                # or in between (but not at the beginning)
                model.add(Dropout(dropOutProb))

        # default initialization is glorot_uniform
        model.add(Dense(num_units, input_dim = input_dim))

        model.add(thisNonlinearity)

    # end of loop over hidden layers

    return model

#----------------------------------------------------------------------

def makeModel():
    return makeModelHelper(
        numHiddenLayers = numHiddenLayers,
        nodesPerHiddenLayer = nodesPerHiddenLayer
        )

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):

    # assert not inputDataIsSparse, "input data is not expected to be sparse"
  
    return [ dataset['input'][rowIndices] ]

#----------------------------------------------------------------------
# function makeInputView(inputValues, first, last)
# 
#   assert(first >= 1)
#   assert(last <= inputValues:size()[1])
# 
#   return inputValues:sub(first,last)
# 
# end

# ----------------------------------------------------------------------
