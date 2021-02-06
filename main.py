from MyERP import ERPExperiment
from MySSVEP import SSVEPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=64, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=128, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)
experiment.multiTrainAndPredict(batchSize=256, dropoutRate=0.6, F1=16, D=1, learningRate=0.0005, numberExperiments=4)

experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4, epochs=100)
experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4, epochs=200)
experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4, epochs=300)
experiment.multiTrainAndPredict(batchSize=4000, dropoutRate=0.6, F1=16, D=1, learningRate=0.001, numberExperiments=4, epochs=400)

# ssvepExperiment = SSVEPExperiment()
# ssvepExperiment.multiTrainAndPredict()
