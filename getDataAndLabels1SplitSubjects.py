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
    if labels is None:
        labels = getLabels(dir, index)
    else:
        labels = np.concatenate([labels, getLabels(dir, index)])
    if data is None:
        data = getData(dir, index)
    else:
        data = np.concatenate([data, getData(dir, index)], axis=2)
    return [data, labels]

def getDataAndLabels():
    X_train = np.concatenate([getData('01', '1'), getData('01', '2'), getData('01', '3'), getData('01', '4'), getData('02', '1'), getData('02', '2'), getData('02', '3')], axis=2)
    X_validate = np.concatenate([getData('03', '1'), getData('03', '2'), getData('03', '3'), getData('03', '4')], axis=2)
    X_test = np.concatenate([getData('04', '1'), getData('04', '2'), getData('04', '3'), getData('04', '4')], axis=2)
    y_train = np.concatenate([getLabels('01', '1'), getLabels('01', '2'), getLabels('01', '3'), getLabels('01', '4'), getLabels('02', '1'), getLabels('02', '2'), getLabels('02', '3')])
    y_validate = np.concatenate([getLabels('03', '1'), getLabels('03', '2'), getLabels('03', '3'), getLabels('03', '4')])
    y_test = np.concatenate([getLabels('04', '1'), getLabels('04', '2'), getLabels('04', '3'), getLabels('04', '4')])

    [X_train, X_validate, X_test] = list(map(lambda x: transformData(x), [X_train, X_validate, X_test]))
    [y_train, y_validate, y_test] = list(map(lambda x: transformLabels(x), [y_train, y_validate, y_test]))

    return [X_train, X_validate, X_test, y_train, y_validate, y_test]

# Old one with all subjects put together
# def getDataAndLabels():
#     data = read(DATA_PATH)
#     labels = read(LABELS_PATH)
#     if data is None or labels is None:
#         for dir in files:
#             for index in files[dir]:
#                 [data, labels] = doStuff(dir, index, data, labels)
#         save(data, DATA_PATH)
#         save(labels, LABELS_PATH)
#     labels = transformLabels(labels)
#     return [data, labels]

def channelsSamplesTrialKernels(data):
    return data.shape[0], data.shape[1], data.shape[2], 1

def transformLabels(labels):
    return labels - 1

def transformData(data):
    filteredData = takeOnlyCertainChannels(data)
    downSampledData = downSample(filteredData)
    multipliedData = downSampledData * 1000
    return multipliedData

def getConfusionMatrixNames():
    return ['1', '2']

def getNumClasses():
    return 2