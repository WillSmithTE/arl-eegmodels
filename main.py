from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=4000, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=4000, D=4)
experiment.multiTrainAndPredict(batchSize=4000, D=4, F1=32)
experiment.multiTrainAndPredict(batchSize=4000, D=8)
experiment.multiTrainAndPredict(batchSize=4000, D=8, F1=16)
