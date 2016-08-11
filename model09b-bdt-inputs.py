#!/usr/bin/env th

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

import numpy as np

#----------------------------------------------------------------------

ninputs = 13


def makeModel():

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
    

    nodesPerHiddenLayer = ninputs * 2

    numHiddenLayers = 3

    # size of minibatch
    batchSize = 32

    # how many minibatches to unpack at a time
    # and to store in the GPU (to have fewer
    # data transfers to the GPU)
    # batchesPerSuperBatch = math.floor(6636386 / batchSize)
    
    model = Sequential()

    for i in range(numHiddenLayers):

        if i == 0:

            model.add(Dense(nodesPerHiddenLayer, input_dim = ninputs))

        elif i == numHiddenLayers - 1:
  
            # add a dropout layer at the end
            #  model:add(nn.Dropout(0.3))
            model.add(Dense(noutputs))
  
        else:
  
            model.add(Dense(nodesPerHiddenLayer))


        if i < numHiddenLayers - 1:
            model.add(Activation('relu'))
  
    # end of loop over hidden layers

    return model

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):

    assert not inputDataIsSparse, "input data is not expected to be sparse"
  
    batchSize = len(rowIndices)
  
    retval = np.zeros(batchSize, ninputs)
  
    #----------
  
    for i in range(batchSize):
  
        rowIndex = rowIndices[i]
        retval[i] = dataset.data[rowIndex]
  
    # end of loop over minibatch indices
  
    return retval

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
