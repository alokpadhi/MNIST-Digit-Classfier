
# coding: utf-8

# In[45]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import matplotlib
get_ipython().magic('matplotlib inline')


# In[46]:

# fetch MNIST datasets
#from six.moves import urllib
#from sklearn.datasets import fetch_mldata
#try:
#    mnist = fetch_mldata('MNIST original')
#except urllib.error.HTTPError as ex:
#    print("Could not download MNIST data from mldata.org, trying alternative...")

    # Alternative method to load MNIST, if mldata.org is down
#    from scipy.io import loadmat
#   mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
#    mnist_path = "./mnist-original.mat"
#    response = urllib.request.urlopen(mnist_alternative_url)
#    with open(mnist_path, "wb") as f:
#        content = response.read()
#        f.write(content)
#    mnist_raw = loadmat(mnist_path)
#    mnist = {
#        "data": mnist_raw["data"].T,
#        "target": mnist_raw["label"][0],
#        "COL_NAMES": ["label", "data"],
#        "DESCR": "mldata.org dataset: mnist-original",
#    }
#    print("Success!")


# In[47]:

import scipy.io as sio
mnist_raw = sio.loadmat('/home/alok-kumar/ML_Work/HandsonML/MNIST/nb_scripts/mnist-original.mat')
mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
X, y = mnist["data"], mnist["target"]
X.shape # data dimension


# In[48]:

y.shape # target value dimension


# In[49]:

some_digit = X[34000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
          interpolation="nearest")
plt.axis("off")
plt.show()


# In[50]:

y[34000]


# In[51]:

# split train and test data set
X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]


# In[52]:

# shuffle the indexes of X-train
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[53]:

# train  a binary classifier with taking only one digit, let it be 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


# In[54]:

# train binary classfier using stochastic gradient descent
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[55]:

sgd_clf.predict([some_digit])


# ## Perfomance measure using crossvalidation with accuracy score

# In[56]:

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[57]:

from sklearn.base import BaseEstimator ## classify every image into not 5 
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# In[58]:

never5clf = Never5Classifier()
cross_val_score(never5clf, X_train, y_train_5, cv=3, scoring="accuracy")


# ## confusion matrix performance measure

# In[59]:

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[60]:

# let'd try confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)


# In[61]:

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)


# In[62]:

# precision and recall score 

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

precision = precision_score(y_train_5, y_train_pred)
print("Precision score:",  "{0:.2f}".format(precision))


# In[63]:

recall = recall_score(y_train_5, y_train_pred)
print("Recall score: ", "{0:.2f}".format(recall))


# In[64]:

# plot between precison and recall
precision, recall, _ = precision_recall_curve(y_train_5, y_train_pred)

plt.step(precision, recall, color='b', alpha=0.2)
plt.fill_between(precision, recall, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.xlim([0.0, 1.2])
plt.ylim([0.0, 1.2])
plt.title('Binary classification precision-recall curve')


# In[65]:

# calculating the f1-score to measure a classifier
from sklearn.metrics import f1_score
F1_score = f1_score(y_train_5, y_train_pred)
print("F1 score is: ", "{0:.2f}".format(F1_score))


# ## Precision-Recall tradeoff

# In[66]:

y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[67]:

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[68]:

# let's change the threshold
threshold = 90000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[69]:

# compute y_scores for all instances 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                            method="decision_function")


# In[70]:

y_scores.shape


# In[71]:

# hack to work around issue #9589 in Scikit-Learn 0.19.0
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]


# In[72]:

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[73]:

y_scores.shape


# In[79]:

def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


# In[80]:

plt.figure(num=1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plot_precision_recall_threshold(precisions, recalls, thresholds)
plt.show()


# In[120]:

(y_train_pred == (y_scores > 0)).all()


# In[130]:

y_train_pred_90 = (y_scores > 10000)
# to have precision score 90


# In[131]:

print("%.2f" % round(precision_score(y_train_5, y_train_pred_90), 2))


# In[132]:

print("%0.2f" % round(recall_score(y_train_5, y_train_pred_90), 2))


# ## ROC Curve

# In[133]:

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# In[140]:

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")


# In[144]:

plt.figure(num=1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plot_roc_curve(fpr, tpr)
plt.show


# In[145]:

# calulate the area under curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# In[148]:

# let's try random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")


# In[149]:

# convet probability to score for random forest clf
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[150]:

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
plt.show()


# In[151]:

roc_auc_score(y_train_5, y_scores_forest) # auc score of random forest


# In[153]:

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)


# In[155]:

precision_score_forest = precision_score(y_train_5, y_train_pred_forest)
precision_score_forest


# In[157]:

recall_score_forest = recall_score(y_train_5, y_train_pred_forest)
recall_score_forest


# In[ ]:



