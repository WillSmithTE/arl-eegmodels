from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=4000, D=8, learningRate=0.005)
experiment.multiTrainAndPredict(batchSize=4000, D=8, F1=16, learningRate=0.005)
experiment.multiTrainAndPredict(batchSize=4000, D=8, learningRate=0.005, epochs=600)
experiment.multiTrainAndPredict(batchSize=4000, D=8, F1=16, learningRate=0.005, epochs=600)
experiment.multiTrainAndPredict(batchSize=4000, D=8, epochs=600)
experiment.multiTrainAndPredict(batchSize=4000, D=8, F1=16, epochs=600)
experiment.multiTrainAndPredict(batchSize=4000, D=8, epochs=1000)
experiment.multiTrainAndPredict(batchSize=4000, D=8, F1=16, epochs=1000)
