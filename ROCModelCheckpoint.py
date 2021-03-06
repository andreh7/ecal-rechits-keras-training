#!/usr/bin/env python

# class taken from https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py

from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys, time

import numpy as np
import os

class ROCModelCheckpoint(Callback):

    #----------------------------------------
    def __init__(self, outputDir, sampleLabel, X, y, weights, verbose=True, logFile = None):
        # filepath can be None (e.g. if no h5py is available)

        super(Callback, self).__init__()

        self.logFile = logFile

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.sampleLabel = sampleLabel
        self.best = 0.0

        self.aucs = []

        self.fouts = [ sys.stdout ]
        if logFile != None:
            self.fouts.append(logFile)
        
        self.outputDir = outputDir

    #----------------------------------------
        
    def on_epoch_end(self, epoch, logs={}):

        start = time.time()
        predictions = self.model.predict(self.X, verbose = True).ravel()
        deltaT = time.time() - start

        for fout in self.fouts:
            print >> fout, "time to evaluate entire",self.sampleLabel,"batch: %.2f min" % (deltaT / 60.0)

        #----------
        # calculate AUC
        #----------

        auc = roc_auc_score(self.y.ravel(), # labels
                            predictions,
                            sample_weight = self.weights,
                            average = None,
                            )

        self.aucs.append(auc)

        for fout in self.fouts:
            print >> fout
            print >> fout, "%s AUC: %f" % (self.sampleLabel, auc)
            fout.flush()

        #----------
        # write predictions out
        #----------
        outputFname = os.path.join(self.outputDir, "roc-data-" + self.sampleLabel + "-%04d.npz" % (epoch + 1))
        np.savez(outputFname, 
                 weight = self.weights,
                 output = predictions,
                 label  = self.y
                 )

        ### fpr, tpr, _ = roc_curve(self.y, self.model.predict(self.X, verbose=True).ravel(), sample_weight=self.weights)
        ### select = (tpr > 0.1) & (tpr < 0.9)
        ### current = auc(tpr[select], 1 / fpr[select])
        ### 
        ### if current > self.best:
        ###     if self.verbose > 0:
        ###         print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
        ###               % (epoch, 'AUC', self.best, current, self.filepath))
        ###     self.best = current
        ###     self.model.save_weights(self.filepath, overwrite=True)
        ### else:
        ###     if self.verbose > 0:
        ###         print("Epoch %05d: %s did not improve" % (epoch, 'AUC'))
