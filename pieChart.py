import matplotlib.pyplot as plt
import numpy as np

def pieChart(labels):
    counts = getCounts(labels)
    labels = counts.keys()
    values = counts.values()
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.show()

def getCounts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
