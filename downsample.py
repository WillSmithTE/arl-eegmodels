## DOWNSAMPLING THE SIGNAL
import numpy as np
import scipy.signal


def downSample(data):
    reshaped = data.reshape(data.shape[2], data.shape[1], data.shape[0])
    Signal_A_240 = reshaped

    # secs = Signal_A_240.shape[1]/256# Number of seconds in signal
    # samps = int(secs*128)     # Number of samples to downsample
    samps = 128

    Signal_A=np.zeros([Signal_A_240.shape[0],samps,Signal_A_240.shape[2]])

    for i in range(0,Signal_A_240.shape[0]):
        Signal_A[i,:,:] = scipy.signal.resample(Signal_A_240[i,:,:], int(samps))

    Signal_A = Signal_A.reshape(data.shape[0], samps, data.shape[2])
    return Signal_A
