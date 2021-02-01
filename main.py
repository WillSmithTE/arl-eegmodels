from MyERP import trainAndPredict
from csvUtil import writeRow

(roc_auc, accuracy) = trainAndPredict(epochs = 10)
writeRow([roc_auc, accuracy])
