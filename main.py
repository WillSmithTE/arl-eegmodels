from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=80, dropoutRate=0.5, F1=8, D=2, learningRate=0.001, epochs=100, numberExperiments=1)
experiment.multiTrainAndPredict(batchSize=160, dropoutRate=0.5, F1=8, D=2, learningRate=0.001, epochs=100, numberExperiments=1)

# ssvepExperiment = SSVEPExperiment()
# ssvepExperiment.multiTrainAndPredict(batchSize=1000, epochs=500)
