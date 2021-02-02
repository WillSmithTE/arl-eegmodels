from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.6, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.6, F1=16, D=2)
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.6, F1=16, D=1, learningRate=0.005)
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.6, F1=16, D=2, learningRate=0.005)
experiment.multiTrainAndPredict(batchSize=300, epochs=300, dropoutRate=0.6, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=300, epochs=300, dropoutRate=0.6, F1=16, D=2)
experiment.multiTrainAndPredict(batchSize=300, epochs=300, dropoutRate=0.6, F1=16, D=1, learningRate=0.005)
experiment.multiTrainAndPredict(batchSize=300, epochs=300, dropoutRate=0.6, F1=16, D=2, learningRate=0.005)
