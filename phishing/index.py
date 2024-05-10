# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.path.append('../rf/RandomForest')

import matplotlib.pyplot as plt
from Randomforest import *
#importing libraries
import sklearn
from joblib import load
#from sklearn.externals import joblib
import inputScript


#output()
#plott()
#load the pickle file
classifier = load('rf_model/rf_final1.pkl',"r")
'''
#input url
print("enter url")
url = input()


#checking and predicting
checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)
print(prediction[0])#1-Legitimate 0-Suspicious -1-Psiphy
if prediction[0]==-1:
    print("It is Phishy URL")
#elif prediction[0]==0:
#    print("It is Suspicious URL")
else:# prediction[0]==1:
    print("It is Legitimate URL")
    
'''