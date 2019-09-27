
library(stats)
library(pROC)
data = read.csv("ouput\\roc.csv")
multiclass.roc(V2~V1, data = data,print.auc=TRUE,plot=TRUE,print.thres=TRUE)
#plot(g.print.auc=TRUE,plot=TRUE,print.thres=TRUE,smooth=T)
