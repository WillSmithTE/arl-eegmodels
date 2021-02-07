from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

# experiment = ERPExperiment()
# experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.5, F1=16, D=1, learningRate=0.001, numberExperiments=8)
ssvepExperiment = SSVEPExperiment()
ssvepExperiment.multiTrainAndPredict()
