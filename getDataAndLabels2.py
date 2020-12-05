import scipy.io
import numpy as np
from util import save, read

DATA_PICKLE = 'pickles/data2.pickle'
LABELS_PICKLE = 'pickles/labels2.pickle'

DATA_FILE = 'Dataset2/rsvp1_data_session_1.npy'
LABELS_FILE = 'Dataset2/rsvp1_labels_session_1.npy'

def getDataAndLabels():
    data = read(DATA_PICKLE)
    labels = read(LABELS_PICKLE)
    if data is None or labels is None:
        data = np.load(DATA_FILE)
        labels = np.load(LABELS_FILE)
        save(data, DATA_PICKLE)
        save(labels, LABELS_PICKLE)
    return [data, labels]

def channelsSamplesTrialKernels(data):
    return data.shape[2], data.shape[3], data.shape[0], 1