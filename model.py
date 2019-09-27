import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
from collections import Counter
data_category = pd.read_csv("input\\sum.csv")
data_category = data_category.dropna()
data_category["other"] = 0
data_category = data_category.fillna(0)
attribution = np.array(data_category[["x","other"]])
category = KMeans(n_clusters=5)
k_m = category.fit(attribution)
data_category['label']=k_m.labels_

#DT
print(Counter(data_category["label"]))
data_DT = data_category.drop(["x","label","other"],axis = 1)
DT_result = data_category["label"]
x_train, x_test, y_train, y_test = train_test_split(data_DT,DT_result,test_size = 0.25,random_state = 7)
print(Counter(y_train))
DT = tree.DecisionTreeClassifier(max_depth=7)
DT.fit(x_train, y_train)
feature_importance = pd.DataFrame(x_train.columns,DT.feature_importances_)
feature_importance.to_csv("output\\importance.csv")
print(x_train.columns, DT.feature_importances_)

from sklearn import tree
import pydotplus
x = pd.DataFrame(x_train)
y = pd.DataFrame(y_train)
dot_data = tree.export_graphviz(DT, out_file=None,feature_names=x.columns,class_names=["lowest","lower","middle","higher","highest"])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg("tree.jpg")
graph.write_jpg("output\\tree.jpg")
result = DT.predict(x_test)
Roc = pd.DataFrame(result, y_test)
Roc.to_csv("output\\Roc.csv")
print(accuracy_score(y_test, DT.predict(x_test)))
