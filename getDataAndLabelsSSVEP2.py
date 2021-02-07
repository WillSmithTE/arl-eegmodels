import scipy.io
import numpy as np
from util import save, read

DATA_PATH = 'pickles/ssvep2.pickle'
LABELS_PATH = 'pickles/ssvep2Labels.pickle'

def getFileNames():
    filenames = []
    for i in range(1, 103):
        filename = 'OnlineSSVEP/S' + "{:03d}".format(i) + '.mat'
        filenames.append(filename)
        
    return filenames

def getDataAndLabels():
    [data, labels] = [read(DATA_PATH), read(LABELS_PATH)]
    filenames = getFileNames()[0]
    if data is None or labels is None:
        data = None
        labels = []
        for filename in filenames:
            print('getting data from file ' + filename)
            [subjectData, subjectLabels] = getDataAndLabelsForSubject(filename)
            if data is None:
                data = subjectData
            else:
                data = np.concatenate([data, subjectData], axis=-1)
            labels = labels + subjectLabels
        labels = np.array(labels)
        save(data, DATA_PATH)
        save(labels, LABELS_PATH)
    return [data, labels]

def getDataAndLabelsForSubject(filename = getFileNames()[0]):
    rawData = scipy.io.loadmat(filename)['data']
    data = None
    labels = []
    for i in range(0, 12):
        targetData = rawData[:,:,:,:,i]
        shape = targetData.shape
        reshaped = targetData.reshape(shape[0], shape[1], shape[2] * shape[3])
        if data is None:
            data = reshaped
        else:
            data = np.concatenate([data, reshaped], axis=-1)
        for _ in range(0, reshaped.shape[-1]):
            labels.append(i)
    return [data, labels]

def channelsSamplesTrialKernels(data):
    return data.shape[0], data.shape[1], data.shape[2], 1

def transformLabels(labels):
    return labels

def getConfusionMatrixNames():
    return range(1,13)

def getNumClasses():
    return 12