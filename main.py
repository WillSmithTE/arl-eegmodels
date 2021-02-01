from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict()
experiment.multiTrainAndPredict(batchSize=500)
experiment.multiTrainAndPredict(batchSize=2000)
experiment.multiTrainAndPredict(batchSize=3000)
experiment.multiTrainAndPredict(batchSize=4000)
