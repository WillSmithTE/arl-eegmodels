import scipy.io
import numpy as np
from util import save, read

DATA_PICKLE = 'pickles/dataSSVEP1.pickle'
LABELS_PICKLE = 'pickles/labelsSSVEP1.pickle'

def getDataFileName(subject, session):
    return getFileName(subject, session, 'data')

def getLabelsFileName(subject, session):
    return getFileName(subject, session, 'labels')

def getFileName(subject, session, labelsOrData):
    return 'SSVEPDataset1/ssvep_' + labelsOrData + '_sub' + subject + '_session_' + session + '.npy'

sessions2And3 = ['2', '3']

files = {
    '1': sessions2And3,
    '2': sessions2And3,
    '3': sessions2And3,
    '4': sessions2And3,
    '5': sessions2And3,
    '6': sessions2And3
}

def getDataAndLabels():
    data = read(DATA_PICKLE)
    labels = read(LABELS_PICKLE)
    if data is None or labels is None:
        for subject in files:
            for session in files[subject]:
                if data is None:
                    data = getData(subject, session)
                else:
                    data = np.concatenate([data, getData(subject, session)])
                if labels is None:
                    labels = getLabels(subject, session)
                else:
                    labels = np.concatenate([labels, getLabels(subject, session)])
        save(data, DATA_PICKLE)
        save(labels, LABELS_PICKLE)
    labels = transformLabels(labels)
    return [data, labels]

def getData(subject, session):
    return np.load(getDataFileName(subject, session))

def getLabels(subject, session):
    return np.load(getLabelsFileName(subject, session))

def channelsSamplesTrialKernels(data):
    return data.shape[1], data.shape[2], data.shape[0], 1

def transformLabels(labels):
    return labels - 1