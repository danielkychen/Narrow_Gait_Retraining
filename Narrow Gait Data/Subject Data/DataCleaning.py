# -*- coding: utf-8 -*-
"""
Script for prepping data for scikit_learn

@author: Daniel
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
%matplotlib inline

plt.style.use('ggplot')

"""
FUNCTIONS-----------------------------------------------------------------------------------------------------------------
"""
def VariableSelect(VariableName, DataFrame):
    return (DataFrame.loc[:,DataFrame.columns.str.find(VariableName)!=-1])

def CombineTrials(rawDataframe, condition):
    tempDataframe = pd.DataFrame()
    newDataframe = pd.DataFrame()
    tempDataframe = rawDataframe[rawDataframe.columns[rawDataframe.columns.str.find(condition)!=-1]]
    
    Trial1 = pd.DataFrame()
    Trial2 = pd.DataFrame()
    Trial3 = pd.DataFrame()
    Trial4 = pd.DataFrame()
    Trial5 = pd.DataFrame()
    for i in range(0, len(tempDataframe.columns)):
        if (int(tempDataframe.columns[i].split(".", 1)[0].split("0")[3]) == 1):
            Trial1 = pd.concat([Trial1, tempDataframe.iloc[:,i]], axis=1)
            
        if (int(tempDataframe.columns[i].split(".", 1)[0].split("0")[3]) == 2):
            Trial2 = pd.concat([Trial2, tempDataframe.iloc[:,i]], axis=1)
            
        if (int(tempDataframe.columns[i].split(".", 1)[0].split("0")[3]) == 3):
            Trial3 = pd.concat([Trial3, tempDataframe.iloc[:,i]], axis=1)
            
        if (int(tempDataframe.columns[i].split(".", 1)[0].split("0")[3]) == 4):
            Trial4 = pd.concat([Trial4, tempDataframe.iloc[:,i]], axis=1)
            
        if (int(tempDataframe.columns[i].split(".", 1)[0].split("0")[3]) == 5):
            Trial5 = pd.concat([Trial5, tempDataframe.iloc[:,i]], axis=1)
            
    data ={"Subject": np.append(["Subject"], np.array(range(0, 16)))}
    SubjectColumn = pd.DataFrame(data)
    Trial1 = pd.concat([Trial1, SubjectColumn], axis=1)
    Trial2 = pd.concat([Trial2, SubjectColumn], axis=1)
    Trial3 = pd.concat([Trial3, SubjectColumn], axis=1)
    Trial4 = pd.concat([Trial4, SubjectColumn], axis=1)
    Trial5 = pd.concat([Trial5, SubjectColumn], axis=1)
    
    Trial1.columns = Trial1.loc[0,:]
    Trial1 = Trial1.drop(0, axis = 0)
    Trial2.columns = Trial2.loc[0,:]
    Trial2 = Trial2.drop(0, axis = 0)
    Trial3.columns = Trial3.loc[0,:]
    Trial3 = Trial3.drop(0, axis = 0)
    Trial4.columns = Trial4.loc[0,:]
    Trial4 = Trial4.drop(0, axis = 0)
    Trial5.columns = Trial5.loc[0,:]
    Trial5 = Trial5.drop(0, axis = 0)
    
    Trial1.index = range(0, 16)
    Trial2.index = range(0, 16)
    Trial3.index = range(0, 16)
    Trial4.index = range(0, 16)
    Trial5.index = range(0, 16)
    newDataframe = pd.concat([Trial1, Trial2, Trial3, Trial4, Trial5], axis=0)
    newDataframe.index = range(0, len(newDataframe))
    
    return newDataframe

def addCondition(Dataframe, condition):
    tempDataframe = pd.DataFrame()
    data = {"Condition": np.full(len(Dataframe), condition)}
    tempDataframe = pd.DataFrame(data)
    Dataframe = pd.concat([Dataframe, tempDataframe], axis=1)
    Dataframe = Dataframe.drop("DomKneeAngleAtPKAM1_MEAN", axis=1)
    
    return Dataframe

def combineLimbs(Dataframe):
    newDataframe = pd.DataFrame()
    Dom = Dataframe[Dataframe.columns[Dataframe.columns.str.find("Dom")!=-1]].iloc[:, range(0, 12)]
    NonDom = Dataframe[Dataframe.columns[Dataframe.columns.str.find("Dom")!=-1]].iloc[:, range(12, 23)]
    OtherVariables = Dataframe[Dataframe.columns[Dataframe.columns.str.find("Dom")==-1]]
    
    Dom = pd.concat([Dom, OtherVariables], axis = 1)
    data = {"Limb": np.full(len(Dom), "Dominant")}
    Dom = pd.concat([Dom, pd.DataFrame(data)], axis = 1)
    Dom = Dom.drop("DomTLAngleAverage_MEAN", axis=1)
    NonDom = pd.concat([NonDom, OtherVariables], axis = 1)
    data = {"Limb": np.full(len(NonDom), "NonDominant")}
    NonDom = pd.concat([NonDom, pd.DataFrame(data)], axis = 1)
    
    newColumns = list()
    for column in Dom.columns:
        newColumns.append(column.split("Dom")[len(column.split("Dom"))-1])   
    Dom.columns = newColumns
    
    newColumns = list()
    for column in NonDom.columns:
        newColumns.append(column.split("Dom")[len(column.split("Dom"))-1])   
    NonDom.columns = newColumns
    
    newDataframe = pd.concat([Dom, NonDom], axis = 0)
    
    newColumns = list()
    for column in newDataframe.columns:
        newColumns.append(column.split("_MEAN")[0])    
    newDataframe.columns = newColumns
    
    return newDataframe
"""
"""

"""
Load dataframe/s into python
"""
subjectDataDir = Path("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/Subject Data/")

RawDataDF = pd.read_csv(subjectDataDir / "FiveTrialsRawData.csv")

"""
Combine trials, and correctly format conditions and limbs
"""
Conditions = list(["Normal", "Crossover", "Narrow"])

newDF = pd.DataFrame()
for condition in Conditions:
    FullDF = pd.DataFrame()
    FullDF = CombineTrials(RawDataDF, condition)
    FullDF = addCondition(FullDF, condition)
    FullDF = combineLimbs(FullDF)
    newDF = pd.concat([newDF, FullDF], axis = 0)

"""
Wide conditions (due to slightly different formatting)
"""
FullDF = pd.DataFrame()
FullDF = CombineTrials(RawDataDF, "WideDom")
FullDF = addCondition(FullDF, "Wide")
data = {"Limb": np.full(len(FullDF), "Dominant")}
FullDF = pd.concat([FullDF, pd.DataFrame(data)], axis = 1)
newDF = pd.concat([newDF, FullDF], axis = 0)

FullDF = pd.DataFrame()
FullDF = CombineTrials(RawDataDF, "WideNonDom")

tempDataframe = pd.DataFrame()
data = {"Condition": np.full(len(FullDF), "Wide")}
tempDataframe = pd.DataFrame(data)
FullDF = pd.concat([FullDF, tempDataframe], axis=1)

data = {"Limb": np.full(len(FullDF), "NonDominant")}
FullDF = pd.concat([FullDF, pd.DataFrame(data)], axis = 1)









