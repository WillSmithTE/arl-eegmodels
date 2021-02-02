from getDataAndLabels1 import *

def getSizes():
    [data, labels] = getDataAndLabels()

    starts = []
    ends = []
    sizes = []
    i = 0

    numLabels = len(labels)
    while i < numLabels:
        if labels[i] == 0:
            print('found 1, starting at ', i)
            starts.append(i)
            while labels[i] == 0:
                i += 1
            print('ending at ', i)
            ends.append(i)
        i += 1

    print('found ', len(starts))
    i = 0

    while i < len(starts):
        size = ends[i] - starts[i]
        sizes.append(size)
        i += 1

    print(sizes)
    
import pandas as pd
def getCorrelations():
    df = pd.read_csv('~/Downloads/results_all.csv')
    return df.corr()
