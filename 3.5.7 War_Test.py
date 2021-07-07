#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/invasion.csv')
df.head()

oi_df = pd.read_csv("https://stepik.org/media/attachments/course/4852/operative_information.csv")
oi_df.head()


X = df.drop('class', axis= 1)
y = df['class'].replace({'fighter':  '1', 'transport': '2', 'cruiser': '3'})


clf = RandomForestClassifier()

params = {'n_estimators': range(10,50,10), 'max_depth': range(1,20,2), 'min_samples_split': range(1,5,1),          'min_samples_leaf': range(1,5,1)}

GS_clf = GridSearchCV(clf, params, cv=3, verbose=1)
GS_clf.fit(X, y)



best_params = GS_clf.best_params_


best_clf = RandomForestClassifier(max_depth= 3, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 10)
best_clf.fit(X,y)



pred_df = pd.DataFrame(best_clf.predict(oi_df))
pred_df.replace({'1': 'fighter', '2': 'transport', '3': 'cruiser'}).value_counts()


import seaborn as sns

imp = pd.DataFrame(best_clf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8)) 


data = pd.read_csv('https://stepik.org/media/attachments/course/4852/space_can_be_a_dangerous_place.csv')
data.head()


data.columns
data.groupby('buggers_were_noticed').agg({'dangerous':'mean'}).round(2)

data.groupby('black_hole_is_near').agg({'dangerous':'mean'}).round(2)

data.groupby('nearby_system_has_planemo').agg({'dangerous':'mean'}).round(2)


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

data_corr = data.drop(['r', 'phi', 'peradventure_index', 'dustiness'], axis= 1) #.corr(method= histogram_intersection)

sns.heatmap(data_corr.corr(), annot = True)

