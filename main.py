from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=100, dropoutRate=0.5, F1=8, D=2, learningRate=0.001, epochs=500, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=100, dropoutRate=0.5, F1=16, D=1, learningRate=0.001, epochs=500, numberExperiments=4)

# ssvepExperiment = SSVEPExperiment()
# ssvepExperiment.multiTrainAndPredict(batchSize=1000, epochs=500)
