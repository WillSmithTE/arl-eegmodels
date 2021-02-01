from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.trainAndPredict(epochs = 10)
