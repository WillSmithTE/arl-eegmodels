from MyERP import trainAndPredict
from csvUtil import writeRow

(roc_auc, accuracy) = trainAndPredict()
writeRow([roc_auc, accuracy])
