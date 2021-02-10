import numpy as np

testInput = np.zeros((6,231,5121))

def takeSubset(data = testInput):
    #convert -.1s->0.8s to 0.0s->0.5s
    samples = data.shape[1]
    point1Seconds = samples//9
    start = point1Seconds #start .1 seconds into epochs
    end = point1Seconds*6 #end .6 seconds into epochs (ie at 0.5s)
    subset = data[:, start:end, :]
    return subset