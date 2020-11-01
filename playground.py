from getDataAndLabels import getDataAndLabels
import matplotlib.pyplot as plt
import numpy as np
from pieChart import pieChart

[data, labels] = getDataAndLabels()
pieChart(labels)
