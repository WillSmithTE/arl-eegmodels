from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(F1=16)
experiment.multiTrainAndPredict(batchSize=500, F1=16)
experiment.multiTrainAndPredict(batchSize=2000, F1=16)
experiment.multiTrainAndPredict(batchSize=3000, F1=16)
experiment.multiTrainAndPredict(batchSize=4000, F1=16)
experiment.multiTrainAndPredict(batchSize=10000)
experiment.multiTrainAndPredict(batchSize=10000, F1=16)

experiment.multiTrainAndPredict(F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=2000, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=3000, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=4000, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=10000, F1=16, D=1)
