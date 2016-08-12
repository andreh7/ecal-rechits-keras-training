#!/usr/bin/env python

# class taken from https://github.com/ml-slac/deep-jets/blob/master/training/conv-train.py

from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ROCModelCheckpoint(Callback):

    #----------------------------------------
    def __init__(self, sampleLabel, X, y, weights, verbose=True):
        # filepath can be None (e.g. if no h5py is available)

        super(Callback, self).__init__()

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.sampleLabel = sampleLabel
        self.best = 0.0

        self.aucs = []
        
    #----------------------------------------
        
    def on_epoch_end(self, epoch, logs={}):


        predictions = self.model.predict(self.X, verbose = True).ravel()
        auc = roc_auc_score(self.y.ravel(), # labels
                            predictions,
                            sample_weight = self.weights,
                            average = None,
                            )

        print
        print "%s AUC: %.3f" % (self.sampleLabel, auc)
        self.aucs.append(auc)
        

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
