from MyERP import ERPExperiment
from csvUtil import writeRow

experiment = ERPExperiment()
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.65, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=100, dropoutRate=0.65, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.7, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=100, dropoutRate=0.7, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=300, dropoutRate=0.6, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=500, epochs=100, dropoutRate=0.6, F1=16, D=1)

experiment.multiTrainAndPredict(batchSize=5000, epochs=300, dropoutRate=0.65, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=5000, epochs=100, dropoutRate=0.65, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=5000, epochs=300, dropoutRate=0.7, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=5000, epochs=100, dropoutRate=0.7, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=5000, epochs=300, dropoutRate=0.6, F1=16, D=1)
experiment.multiTrainAndPredict(batchSize=5000, epochs=100, dropoutRate=0.6, F1=16, D=1)
