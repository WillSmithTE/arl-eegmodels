from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=3000)
experiment.multiTrainAndPredict(batchSize=3000, dropoutRate=.65)
experiment.multiTrainAndPredict(batchSize=3000, dropoutRate=.8)
experiment.multiTrainAndPredict(batchSize=3000, learningRate=.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=3000, F1=16)
experiment.multiTrainAndPredict(batchSize=3000, F1=32)
experiment.multiTrainAndPredict(batchSize=3000, F1=64)
experiment.multiTrainAndPredict(batchSize=3000, F1=128)
experiment.multiTrainAndPredict(batchSize=3000, D=1, numberExperiments=4)
