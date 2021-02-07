from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

# experiment = ERPExperiment()
# experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.5, F1=16, D=1, learningRate=0.001, numberExperiments=8)
ssvepExperiment = SSVEPExperiment()
ssvepExperiment.multiTrainAndPredict(batchSize=80, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=100, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=150, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=200, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=300, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=400, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=600, epochs=500)
ssvepExperiment.multiTrainAndPredict(batchSize=800, epochs=500)
