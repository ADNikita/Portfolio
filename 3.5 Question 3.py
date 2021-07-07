#!/usr/bin/env python
# coding: utf-8

import pandas  as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv('training_mush.csv')
df.head()

df.columns

X = df.drop('class', axis = 1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

parameters = {'n_estimators': [10,50,10], 'max_depth': [1,9,2], 'min_samples_leaf': [1,7],              'min_samples_split': [2,9,2]}
clf_rf = RandomForestClassifier(random_state=0)

# n_estimators: от 10 до 50 с шагом 10
# max_depth: от 1 до 12 с шагом 2
# min_samples_leaf: от 1 до 7
# min_samples_split: от 2 до 9 с шагом 2


clf = GridSearchCV(clf_rf, parameters, cv=3, verbose=1)
clf.fit(X, y)


best_params = clf.best_params_
best_params


best_clf = RandomForestClassifier(n_estimators= 10, max_depth= 9, min_samples_leaf= 1, min_samples_split= 2, random_state=0)
best_clf.fit(X,y)


imp = pd.DataFrame(best_clf.feature_importances_, index=X.columns, columns=['importance'])



imp.sort_values('importance').plot(kind='barh', figsize=(12, 8)) 

df_mush = pd.read_csv('https://stepik.org/media/attachments/course/4852/testing_mush.csv')
df_mush

clf_predict = pd.DataFrame(best_clf.predict(df_mush))
clf_predict.sum()

from sklearn.metrics import confusion_matrix

df_mush_cor = pd.read_csv('testing_y_mush.csv')
df_mush_cor.head():

sns.heatmap(confusion_matrix(df_mush_cor, clf_predict), annot=True, cmap="Blues", annot_kws={"size": 16})





