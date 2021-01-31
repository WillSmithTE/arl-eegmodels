import scipy.io
import numpy as np
from util import save, read
from channelPrune import takeOnlyCertainChannels
from downsample import downSample

files = {
    '01': ['1', '2', '3', '4'],
    '02': ['1', '2', '3'],
    '03': ['1', '2', '3', '4'],
    '04': ['1', '2', '3', '4']
}

directories = ['S01', 'S02', 'S03', 'S04']
indexes = ['1', '2', '3', '4']
DATA_PATH = 'pickles/data1_filtered.pickle'
LABELS_PATH = 'pickles/labels1_filtered.pickle'

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
    if labels is None:
        labels = getLabels(dir, index)
    else:
        labels = np.concatenate([labels, getLabels(dir, index)])
    return [data, labels]

def getDataAndLabels():
    data = read(DATA_PATH)
    labels = read(LABELS_PATH)
    if data is None or labels is None:
        for dir in files:
            for index in files[dir]:
                [data, labels] = doStuff(dir, index, data, labels)
        data = getFilteredData()
        save(data, DATA_PATH)
        save(labels, LABELS_PATH)
    data = transformData(data)
    labels = transformLabels(labels)
    return [data, labels]

import scipy.io

def getFilteredData():
    return scipy.io.loadmat('filtered_erp1')['Filtered_signal']

def channelsSamplesTrialKernels(data):
    return data.shape[0], data.shape[1], data.shape[2], 1

def transformLabels(labels):
    return labels - 1

def transformData(data):
    data = takeOnlyCertainChannels(data)
    data = downSample(data)
    return data

def getConfusionMatrixNames():
    return ['1', '2']

def getNumClasses():
    return 2