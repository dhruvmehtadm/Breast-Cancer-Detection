# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:06:23 2020

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

df.head(10)

df.describe()

df.shape

df.isnull()
df = df.dropna(axis = 1)

# count number of malignant and benign cells
df['diagnosis'].value_counts()

# visualise the count
sns.countplot(df['diagnosis'], label = 'count')

# transform string data into numbers (encode categorical data values)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.iloc[:,1] = label.fit_transform(df.iloc[:,1].values)

df.head(10)

# create pairplot
sns.pairplot(df.iloc[:, 1:5], hue = 'diagnosis')

# correlations b/w columns
df.iloc[:,1:12].corr()

#visualise correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:, 1:12].corr(), annot = True, fmt = '.0%')

# splitting dataset into independent and dependent set
X = df.iloc[:, 2:31].values
y = df.iloc[:, 1].values

# split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#create a function for 3 models
def models(X_train, y_train):
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression(random_state = 0)
    logistic.fit(X_train, y_train)
    
    #Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    decision = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    decision.fit(X_train, y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    random.fit(X_train, y_train)
    
    #Naive Bayes Algorithm
    from sklearn.naive_bayes import GaussianNB
    naive = GaussianNB()
    naive.fit(X_train, y_train)
    
    #Support Vector Classifier
    from sklearn.svm import SVC
    svc = SVC(kernel = 'rbf', random_state = 0)
    svc.fit(X_train, y_train)
    
    #print models accuracy on training set
    print('For Logistic Regression:', logistic.score(X_train, y_train))
    print('For Decision Tree Classifier:', decision.score(X_train, y_train))
    print('For Random Forest Classifier:', random.score(X_train, y_train))
    print('For Naive Bayes Algorithm:', naive.score(X_train, y_train))
    print('For Support Vector Classifier:', svc.score(X_train, y_train))
    return logistic, decision, random, naive,svc

#getting all models
model = models(X_train, y_train)

#test model accuracy (logistic regression) on test data on confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model[2].predict(X_test)) #model[0] means logistic regression

cm

#test model accuracy of all models on test data
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print('Model', i)
    print(classification_report(y_test, model[i].predict(X_test)))
    print(accuracy_score(y_test, model[i].predict(X_test)))
    print()
    
#print prediction of random forest classifier model as its accuracy on test set is best
pred = model[2].predict(X_test)
print(pred)
print()
print(y_test)
print()

#print prediction of support vector classifier model as its accuracy on test set is best
pred = model[4].predict(X_test)
print(pred)
print()
print(y_test)
print()
print("Random Forest Classifier and Support Vector Classifier gives the same accuracy")