#!/usr/bin/env python

# class taken from https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py

from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys

class ROCModelCheckpoint(Callback):

    #----------------------------------------
    def __init__(self, sampleLabel, X, y, weights, verbose=True, logFile = None):
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
        
    #----------------------------------------
        
    def on_epoch_end(self, epoch, logs={}):


        predictions = self.model.predict(self.X, verbose = True).ravel()
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
