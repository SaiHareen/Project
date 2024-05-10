# -*- coding: utf-8 -*-
#----------------importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import seaborn as sns

#importing the dataset
dataset = pd.read_csv("datasets/phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column

x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values

yx=y.ravel()
#spliting the dataset into training set and test set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,yx,test_size = 0.25, random_state =0 )

#----------------applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 700],
    'max_features': ['sqrt', 'log2'],
    'criterion' :['gini', 'entropy']}]

grid_search = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1)
grid_search.fit(x_train, y_train)
#printing best parameters 
print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 
#-------------------------------------------------------------------------

#fitting RandomForest regression with best params 
classifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_features = 'log2',  random_state = 0)
classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
acc=accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
#print(cm.confusion_matrix)
#dis=plot_confusion_matrix(classifier,X_test,y_test,class_labels=['phishing','legitimate'],cmap=plt.cm.Blues,normalize=normalize)
#print(dis cm
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt="d")
plt.show()
def output():
    print(cm)
    print("accuracy score of our model:",acc*100)


#for error of dataconversion warning columnvector y was pased when 1d array was expected
    
#pickle file joblib
dump(classifier, 'rf_model/rf_final1.pkl')


#-------------Features Importance random forest
names = dataset.iloc[:,:-1].columns
importances =classifier.feature_importances_
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])

vals=dataset['Result'].value_counts().keys().tolist() 
counts=dataset['Result'].value_counts().tolist() 

data_urls=dataset.iloc[:,-1:]
dataset['Result'].value_counts()
import matplotlib.pyplot as plt 
plt.bar(vals,counts,align='center', alpha=0.5)
plt.title("resultant urls ")
plt.show()

def plott():
    #-------------plotting variable importance
    plt.title("Variable Importances")
    plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
    plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
    plt.xlabel('Relative Importance')
    plt.show()


plott()
##output()
