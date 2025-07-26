import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dataset = pd.read_csv("D:/tinku/outstanding/pd_speech_features.csv")

x = dataset.drop(['id','class'],axis=1)
y = dataset['class'].values

y.shape

unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements))) 


x.describe()

X_train, X_test, y_train, y_test = train_test_split(
     x, y, test_size=0.2, random_state=0)

clfRF=RandomForestClassifier(n_estimators=20,
                       criterion='gini',
                       max_depth=None,
                       min_weight_fraction_leaf=0.0,
                       max_features=10, max_leaf_nodes=4,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       bootstrap=True, oob_score=False,
                       n_jobs=5, random_state=None,
                       verbose=0, warm_start=False, class_weight=None,
                       )

clfRF=clfRF.fit(X_train,y_train)

import eli5 
from eli5.sklearn import PermutationImportance



perm = PermutationImportance(clfRF, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

feature_scores = pd.Series(clfRF.feature_importances_, index=X_train.columns).sort_values(ascending=False)

for i,v in enumerate(feature_scores):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.bar([x for x in range(len(feature_scores))], feature_scores)
plt.show()

features_score=clfRF.feature_importances_

features = x.columns

indices = np.argsort(features_score)[-30:]


plt.title('Feature Importances')
plt.barh(range(len(indices)), features_score[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
"""
f, ax = plt.subplots(figsize=(30, 24))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=x)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()
"""
x.isnull().sum().sort_values(ascending=False)

x.numPulses[(x.numPulses==0)].count()

import seaborn as sns

z=x['numPulses']
y=x['numPeriodsPulses']


sns.scatterplot(z,y)

# Detect Outliers with isolation forest algorithm
from sklearn.ensemble import IsolationForest

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(x)

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

scaler_pipeline = Pipeline([('rob_scale', RobustScaler())])

x_scaled = scaler_pipeline.fit_transform(x)

x_scaled = pd.DataFrame(x_scaled,columns=x.columns,index=x.index)

x_scaled.describe()

import seaborn as sns

z=x_scaled['numPulses']
y=x_scaled['numPeriodsPulses']


sns.scatterplot(z,y)

sns.countplot(dataset['class'].values)
plt.xlabel('class Values')
plt.ylabel('class Counts')
plt.show()

print('No parkinson disease', round(dataset['class'].value_counts()[0]/len(dataset) * 100,2), '% of the dataset')
print('parkinson disease', round(dataset['class'].value_counts()[1]/len(dataset) * 100,2), '% of the dataset')


fig, ax = plt.subplots(1, 2, figsize=(18,4))

ppe_val = dataset['PPE'].values
dfa_val = dataset['DFA'].values

sns.distplot(ppe_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of ppe', fontsize=14)
ax[0].set_xlim([min(ppe_val), max(ppe_val)])

sns.distplot(dfa_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of dfa', fontsize=14)
ax[1].set_xlim([min(dfa_val), max(dfa_val)])

transformer = RobustScaler().fit(x)
x_scaled = transformer.transform(x)

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(x, y, test_size=0.2, random_state=42)

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)


print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))



correlation_values=dataset.corr()['class']
print(correlation_values.abs().sort_values(ascending=False))
