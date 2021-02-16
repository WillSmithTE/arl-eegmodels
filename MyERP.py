# -*- coding: utf-8 -*-

"""
 Sample script using EEGNet to classify Event-Related Potential (ERP) EEG data
 from a four-class classification task, using the sample dataset provided in
 the MNE [1, 2] package:
     https://martinos.org/mne/stable/manual/sample_dataset.html#ch-sample-data
   
 The four classes used from this dataset are:
     LA: Left-ear auditory stimulation
     RA: Right-ear auditory stimulation
     LV: Left visual field stimulation
     RV: Right visual field stimulation

 The code to process, filter and epoch the data are originally from Alexandre
 Barachant's PyRiemann [3] package, released under the BSD 3-clause. A copy of 
 the BSD 3-clause license has been provided together with this software to 
 comply with software licensing requirements. 
 
 When you first run this script, MNE will download the dataset and prompt you
 to confirm the download location (defaults to ~/mne_data). Follow the prompts
 to continue. The dataset size is approx. 1.5GB download. 
 
 For comparative purposes you can also compare EEGNet performance to using 
 Riemannian geometric approaches with xDAWN spatial filtering [4-8] using 
 PyRiemann (code provided below).

 [1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
     L. Parkkonen, M. , MNE software for processing MEG and EEG data, 
     NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

 [2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, 
     R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. , MEG and EEG data 
     analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013.

 [3] https://github.com/alexandrebarachant/pyRiemann. 

 [4] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry"
     arXiv:1409.0107. link

 [5] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer 
     Interface Based on Riemannian Geometry", arXiv: 1310.8115.

 [6] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian 
     distance for BCI applications," in 2011 5th International IEEE/EMBS 
     Conference on Neural Engineering (NER), 2011, 348-351.

 [7] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Multiclass 
     Brain-Computer Interface Classification by Riemannian Geometry,” in IEEE 
     Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

 [8] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, “Classification of 
     covariance matrices using a Riemannian-based kernel for BCI applications“, 
     in NeuroComputing, vol. 112, p. 172-178, 2013.


 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those 
 portions are released world-wide under the terms of the Creative Commons Zero 
 1.0 (CC0) license.  
 
 Other portions of this project are subject to domestic copyright protection 
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
 license.  The complete text of the license governing this material is in 
 the file labeled LICENSE.TXT that is a part of this project's official 
 distribution. 
"""

import numpy as np

import mne
from mne import io
from mne.datasets import sample

from EEGModels import EEGNet, DeepConvNet, EEGNet_old

from tensorflow import py_func, double

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from util import read, save

from getDataAndLabels1Subj1Filtered import getDataAndLabels, channelsSamplesTrialKernels, getConfusionMatrixNames, getNumClasses

def getClassWeights(arg):
    return dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(arg), arg)))

def shuffle(X, y):
    indexes = np.random.permutation(len(X))
    return (X[indexes], y[indexes])

def swapOnesAndZeroes(labels):
    zeroes = (labels == 0)
    ones = (labels == 1)
    labels[zeroes] = 1
    labels[ones] = 0
    return labels

class ERPExperiment():
    def __init__(self):
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
        std_scaler = StandardScaler()

        [data, labels] = getDataAndLabels()
        labels = swapOnesAndZeroes(labels)
        X = data *1000 # format is in (channels, samples, trials)
        y = labels
        
        X = std_scaler.fit_transform(X)

        self.chans, self.samples, self.trials, self.kernels = channelsSamplesTrialKernels(data)
        # chans, samples, trials, kernels = channelsSamplesTrialKernels(data)

    #pylint: disable=too-many-function-args
        X = X.reshape(self.trials, self.kernels, self.chans, self.samples)
        
        self.X_train,X_other,self.y_train,y_other=train_test_split(X,y,test_size=0.5,stratify=y)
        self.X_validate,self.X_test,self.y_validate,self.y_test=train_test_split(X_other,y_other,test_size=0.5,stratify=y_other)
        
        # X, y = shuffle(X, y)

        # half = (self.trials//4)*2
        # threeQuarters = (self.trials//4) * 3

        # # take 50/25/25 percent of the data to train/validate/test
        # self.X_train      = X[0:half,]
        # self.y_train      = y[0:half]
        # self.X_validate   = X[half:threeQuarters,]
        # self.y_validate   = y[half:threeQuarters]
        # self.X_test       = X[threeQuarters:,]
        # self.y_test       = y[threeQuarters:]

    # convert labels to one-hot encodings.
        self.Y_train      = np_utils.to_categorical(self.y_train)
        self.Y_validate   = np_utils.to_categorical(self.y_validate)
        self.Y_test       = np_utils.to_categorical(self.y_test)
    
        print('X_train shape:', self.X_train.shape)
        print('X_validate shape:', self.X_validate.shape)
        print('X_testshape:', self.X_test.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_validate.shape[0], 'validate samples')
        print(self.X_test.shape[0], 'test samples')

        print("chans:", self.chans, "samples:", self.samples)
        
    def multiTrainAndPredict(
        self,
        numberExperiments = 2,
        epochs = 300,
        batchSize = 1000,
        class_weights = None,
        F1 = 8,
        D = 2,
        kernLength = None,
        dropoutRate = 0.5,
        learningRate = 0.001,
    ):
        i = 0
        while i < numberExperiments:
            self.trainAndPredict(
                epochs,
                batchSize,
                class_weights,
                F1,
                D,
                kernLength,
                dropoutRate,
                learningRate,
            )
            i += 1

    def trainAndPredict(
        self,
        epochs = 300,
        batchSize = 1000,
        class_weights = None,
        F1 = 8,
        D = 2,
        kernLength = None,
        dropoutRate = 0.5,
        learningRate = 0.001,
    ):
        if class_weights is None:
            class_weights = getClassWeights(self.y_train)
        if kernLength is None:
            kernLength = int(self.samples/2)
        # class_weights = {1:1, 0:1}
        # class_weights = {0:22, 1:1}

        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
        # model configurations may do better, but this is a good starting point)

        F2 = F1 * D
        
        print('F1 (temporal filters)', F1)
        print('D (spatial filters', D)
        print('F2 (pointwise filters', F2)
        print('kernLength', kernLength)
        print('learningRate', learningRate)
        print('class_weights', class_weights)
        print('epochs', epochs)
        print('batchSize', batchSize)

        model = EEGNet(nb_classes = getNumClasses(), Chans = self.chans, Samples = self.samples, 
                    dropoutRate = dropoutRate, kernLength = kernLength, F1 = F1, D = D, F2 = F2, 
                    dropoutType = 'Dropout')
        
        # model = DeepConvNet(nb_classes=getNumClasses(), Chans=self.chans, Samples=self.samples, dropoutRate=dropoutRate)
        
        # model = EEGNet_old(nb_classes = getNumClasses(), Chans = self.chans, Samples = self.samples, 
        #     dropoutRate = dropoutRate)


        optimizer = Adam(lr=learningRate)

        metrics = ['accuracy']

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = metrics) 

        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

        class OnEpochEndCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                x_test = self.validation_data[0]
                y_test = self.validation_data[1]
                # x_test, y_test = self.validation_data
                predictions = self.model.predict(x_test)
                y_test = np.argmax(y_test, axis=-1)
                predictions = np.argmax(predictions, axis=-1)
                c = confusion_matrix(y_test, predictions)

                roc_auc = roc_auc_score(y_test, predictions)

                print('Confusion matrix:\n', c)
                print('sensitivity', c[0, 0] / (c[0, 1] + c[0, 0]))
                print('specificity', c[1, 1] / (c[1, 1] + c[1, 0]))
                print('roc_auc_score', roc_auc)
                
        model.fit(self.X_train, self.Y_train, batch_size = batchSize, epochs = epochs, 
                                verbose = 2, validation_data=(self.X_validate, self.Y_validate),
                                callbacks=[checkpointer, OnEpochEndCallback()], class_weight = class_weights)

        probs       = model.predict(self.X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == self.Y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))

        if getNumClasses() == 2:
            roc_auc = roc_auc_score(self.y_test, preds)
            
            print('roc_auc_score', roc_auc)

            probsConverted = probs[:,1]
            fpr, tpr, thresholds = roc_curve(self.y_test, probsConverted)
            
            gmeans = np.sqrt(tpr * (1-fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
            
            roc_auc = auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig('roc')

        print('confusion_matrix')
        print(confusion_matrix(self.y_test, preds))
        log(epochs, batchSize, self.samples, kernLength, dropoutRate, learningRate, roc_auc, acc, F1, D)

from datetime import datetime
from csvUtil import writeRow
def log(epochs, batchSize, sampleRate, kernLength, dropout, learning, roc_auc, accuracy, F1, D):
    date = datetime.today().strftime('%d/%m/%y')
    dataset = 'all'
    writeRow([date, epochs, dataset, batchSize, sampleRate, kernLength, dropout, learning, roc_auc, accuracy, F1, D])

# names        = getConfusionMatrixNames()

# plot loss
# plt.figure(0)
# plt.plot(fittedModel.history['loss'])
# plt.plot(fittedModel.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('plot-loss')
