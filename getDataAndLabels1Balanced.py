import scipy.io
import numpy as np
from util import save, read

files = {
    '01': ['1', '2', '3', '4'],
    '02': ['1', '2', '3'],
    '03': ['1', '2', '3', '4'],
    '04': ['1', '2', '3', '4']
}

directories = ['S01', 'S02', 'S03', 'S04']
indexes = ['1', '2', '3', '4']
DATA_PATH = 'pickles/data.pickle'
LABELS_PATH = 'pickles/labels.pickle'

def getLabels(dir, index):
    mat = getMatFile(dir, index, 'labels')
    return mat['labels'][0]

def getData(dir, index):
    mat = getMatFile(dir, index, 'data')
    return mat['data']

def getMatFile(dir, index, dataOrLabels):
    return scipy.io.loadmat('Dataset1/S' + dir + '/S' + dir + '_session' + index + '_' + dataOrLabels + '.mat')

def doStuff(dir, index, data, labels):
    print('reading file ', dir, index)
    labelsToAdd = getLabels(dir, index)
    lenLabels = len(labelsToAdd)
    labelsToAdd = np.append(labelsToAdd[0:80], labelsToAdd[lenLabels-80:lenLabels])
    
    dataToAdd = getData(dir, index)
    dataToAdd = np.append(dataToAdd[:,:,0:80], dataToAdd[:,:,lenLabels-80:lenLabels], axis=2)
    if labels is None:
        labels = labelsToAdd
    else:
        labels = np.concatenate([labels, labelsToAdd])
    if data is None:
        data = dataToAdd
    else:
        data = np.concatenate([data, dataToAdd], axis=2)
    return [data, labels]

def getDataAndLabels():
    data = read(DATA_PATH)
    labels = read(LABELS_PATH)
    if data is None or labels is None:
        for dir in files:
            for index in files[dir]:
                [data, labels] = doStuff(dir, index, data, labels)
        save(data, DATA_PATH)
        save(labels, LABELS_PATH)
    labels = transformLabels(labels)
    return [data, labels]

def channelsSamplesTrialKernels(data):
    return data.shape[0], data.shape[1], data.shape[2], 1

def transformLabels(labels):
    return labels - 1

def getConfusionMatrixNames():
    return ['1', '2']

def getNumClasses():
    return 2