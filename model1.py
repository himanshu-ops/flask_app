import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import pickle

# reading dataset
dataset = pd.read_csv("E:/Study/Data Science/DAB_103_practice/Project_DAB_103/New_project/Final_Clean1.csv")

# converting string to numeric for ML model
objlist = dataset.select_dtypes(include="object").columns

le = LabelEncoder()

for feat in objlist:
    dataset[feat] = le.fit_transform(dataset[feat].astype(str))

# print(dataset.info())

dataset.to_csv('No_String.csv')

# dividing dataset into attributes and labels
A = dataset[
    ['AMT_ANNUITY', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 'NAME_HOUSING_TYPE',
     'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'AMT_CREDIT']]
# A = dataset.drop('TARGET', axis=1)
l = dataset['TARGET']

# splitting data
A_train, A_test, l_train, l_test = train_test_split(A, l, test_size=0.30)

# training and making predictions
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(A_train, l_train)

# prediction
l_pred = classifier.predict(A_test)

# confusion matrix
print(confusion_matrix(l_test, l_pred))
print(classification_report(l_test, l_pred))

# saving the trained model via pickle
pickle.dump(classifier, open('model1.pkl', 'wb'))

# predicting values with test set
prediction = classifier.predict_proba(A_test)

# getting auc score
auc = roc_auc_score(l_test, prediction[:, 1])

# printing auc score
print("The AUC score is: {}".format(auc))

# plotting ROC curve
fpr, tpr, _ = roc_curve(l_test, prediction[:, 1])

plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

