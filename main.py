from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=64, dropoutRate=0.6, F1=16, D=4, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=128, dropoutRate=0.6, F1=16, D=4, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=256, dropoutRate=0.6, F1=16, D=4, learningRate=0.0005, numberExperiments=4)
