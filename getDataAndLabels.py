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
    printThing(getLabels(dir, index))
    printThing(getData(dir, index))
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
    data = read(DATA_PATH)
    labels = read(LABELS_PATH)
    if data is None or labels is None:
        for dir in files:
            for index in files[dir]:
                [data, labels] = doStuff(dir, index, data, labels)
        save(data, DATA_PATH)
        save(labels, LABELS_PATH)
    return [data, labels]