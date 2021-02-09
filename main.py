from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=30, dropoutRate=0.5, F1=8, D=2, learningRate=0.001)
experiment.multiTrainAndPredict(batchSize=16, dropoutRate=0.5, F1=8, D=2, learningRate=0.001)
experiment.multiTrainAndPredict(batchSize=30, dropoutRate=0.5, F1=8, D=2, learningRate=0.001, epochs=500)
experiment.multiTrainAndPredict(batchSize=30, dropoutRate=0.3, F1=8, D=2, learningRate=0.001)
# ssvepExperiment = SSVEPExperiment()
# ssvepExperiment.multiTrainAndPredict(batchSize=1000, epochs=500)
