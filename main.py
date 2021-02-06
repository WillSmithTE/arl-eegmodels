from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=64, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=128, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=256, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)

experiment.multiTrainAndPredict(batchSize=100, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=200, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=300, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=400, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4)
