from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

# experiment = ERPExperiment()
# experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.5, F1=16, D=1, learningRate=0.001, numberExperiments=8)
ssvepExperiment = SSVEPExperiment()
ssvepExperiment.multiTrainAndPredict(batchSize=1000, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=3000, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=10000, epochs=500)
