# https://github.com/poganyg/IIR-filter

import matplotlib.pyplot as plt
import numpy as np
from IIR2Filter import IIR2Filter
from getDataAndLabels1Subj1 import getDataAndLabels, channelsSamplesTrialKernels, getConfusionMatrixNames, getNumClasses

[data, labels] = getDataAndLabels()

fs = 200
FilterMains = IIR2Filter(3,[0.5,40],'bandpass', fs=231)

# impulse = np.zeros(1000)
# impulse[0] = 1
# impulseResponse = np.zeros(len(impulse))
impulseResponse = data[0]

for i in range(len(impulseResponse)):
    for j in range(len(impulseResponse[i])):
        impulseResponse[i][j] = FilterMains.filter(impulseResponse[i][j])

# To obtain the frequency response from the impulse response the Fourier
# transform of the impulse response has to be taken. As it produces
# a mirrored frequency spectrum, it is enough to plot the first half of it.
freqResponse = np.fft.fft(impulseResponse)
freqResponse = abs(freqResponse[0:int(len(freqResponse)/2)])
xfF = np.linspace(0,fs/2,len(freqResponse))

plt.figure("Frequency Response")
plt.plot(xfF,np.real(freqResponse))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Bandstop")
plt.show()