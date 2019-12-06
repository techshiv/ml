import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
np.random.seed(2)
ids=load_iris()
x=pd.DataFrame(ids.data)
y=pd.DataFrame(ids.target)
colormap=np.array(['red','blue','green'])
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3).fit(x)
plt.title("KMeans")
plt.scatter(x[2],x[3],c=colormap[km.labels_])
plt.show()
import sklearn.metrics as sm
print('K Means Accuracy:',sm.accuracy_score(y,km.labels_))
print('Confusion Matrix:\n',sm.confusion_matrix(y,km.labels_))
from sklearn.mixture import GaussianMixture
gm=GaussianMixture(n_components=3).fit(x)
ycluster=gm.predict(x)
plt.title("EM")
plt.scatter(x[2],x[3],c=colormap[ycluster])
plt.show()
print('EM Accuracy:',sm.accuracy_score(y,ycluster))
print('Confusion Matrix:\n',sm.confusion_matrix(y,ycluster))