# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:13:12 2019

Subject-Specific Lasso Regression for KAM, KFM, Impulse

@author: Daniel
"""

import pandas as pd
pd.set_option('display.max_columns', 10)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
%matplotlib qt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from statistics import mean
%config InlineBackend.figure_format='retina'
plt.style.use('ggplot')

plt.rcParams.update({'font.size': 2})
matplotlib.rcParams['lines.linewidth'] = 0.1

data = pd.read_csv("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/CleanDataABS.csv")

"""
Calculate Average of Normal gait for each subject---------------------------------------------------------------------------------------------------
"""
dataNormal = data.loc[data['Condition'] == 'Normal', :]
DomNormal = dataNormal.loc[dataNormal['Limb'] == 'Dominant', :]
NonDomNormal = dataNormal.loc[dataNormal['Limb'] == 'NonDominant', :]

AverageNormDF = pd.DataFrame(columns = dataNormal.columns)
for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = DomNormal.loc[DomNormal['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Normal', 'Dominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataNormal.columns)
    
    AverageNormDF = pd.concat([AverageNormDF, tempArray], axis = 0)

for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = NonDomNormal.loc[NonDomNormal['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Normal', 'NonDominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataNormal.columns)
    
    AverageNormDF = pd.concat([AverageNormDF, tempArray], axis = 0)

AverageNormDF.index = range(0, len(AverageNormDF))
AverageNormDF[AverageNormDF.columns[0:15]] = AverageNormDF.loc[:,AverageNormDF.columns[0:15]].apply(pd.to_numeric)

"""
Calculate Average of Crossover gait for each subject---------------------------------------------------------------------------------------------------
"""
dataCrossover = data.loc[data['Condition'] == 'Crossover', :]
DomCrossover = dataCrossover.loc[dataCrossover['Limb'] == 'Dominant', :]
NonDomCrossover = dataCrossover.loc[dataCrossover['Limb'] == 'NonDominant', :]

AverageCrossoverDF = pd.DataFrame(columns = dataCrossover.columns)
for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = DomCrossover.loc[DomCrossover['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Crossover', 'Dominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataCrossover.columns)
    
    AverageCrossoverDF = pd.concat([AverageCrossoverDF, tempArray], axis = 0)

for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = NonDomCrossover.loc[NonDomCrossover['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Crossover', 'NonDominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataCrossover.columns)
    
    AverageCrossoverDF = pd.concat([AverageCrossoverDF, tempArray], axis = 0)

AverageCrossoverDF.index = range(0, len(AverageCrossoverDF))
AverageCrossoverDF[AverageCrossoverDF.columns[0:15]] = AverageCrossoverDF.loc[:,AverageCrossoverDF.columns[0:15]].apply(pd.to_numeric)

"""
Calculate Average of Narrow gait for each subject---------------------------------------------------------------------------------------------------
"""
dataNarrow = data.loc[data['Condition'] == 'Narrow', :]
DomNarrow = dataNarrow.loc[dataNarrow['Limb'] == 'Dominant', :]
NonDomNarrow = dataNarrow.loc[dataNarrow['Limb'] == 'NonDominant', :]

AverageNarrowDF = pd.DataFrame(columns = dataNarrow.columns)
for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = DomNarrow.loc[DomNarrow['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Narrow', 'Dominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataNarrow.columns)
    
    AverageNarrowDF = pd.concat([AverageNarrowDF, tempArray], axis = 0)

for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = NonDomNarrow.loc[NonDomNarrow['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Narrow', 'NonDominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataNarrow.columns)
    
    AverageNarrowDF = pd.concat([AverageNarrowDF, tempArray], axis = 0)

AverageNarrowDF.index = range(0, len(AverageNarrowDF))
AverageNarrowDF[AverageNarrowDF.columns[0:15]] = AverageNarrowDF.loc[:,AverageNarrowDF.columns[0:15]].apply(pd.to_numeric)

"""
Calculate Average of Wide gait for each subject---------------------------------------------------------------------------------------------------
"""
dataWide = data.loc[data['Condition'] == 'Wide', :]
DomWide = dataWide.loc[dataWide['Limb'] == 'Dominant', :]
NonDomWide = dataWide.loc[dataWide['Limb'] == 'NonDominant', :]

AverageWideDF = pd.DataFrame(columns = dataWide.columns)
for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = DomWide.loc[DomWide['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Wide', 'Dominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataWide.columns)
    
    AverageWideDF = pd.concat([AverageWideDF, tempArray], axis = 0)

for subject in range(0, 16):
    tempDF = pd.DataFrame()
    tempDF = NonDomWide.loc[NonDomWide['Subject'] == subject, :'Stride_Width_Mean']
    tempArray = np.array(tempDF.mean(axis=0))
    tempArray = np.append(tempArray, [subject, 'Wide', 'NonDominant'])
    tempArray = pd.DataFrame(np.reshape(tempArray, (1, len(tempArray))), columns=dataWide.columns)
    
    AverageWideDF = pd.concat([AverageWideDF, tempArray], axis = 0)

AverageWideDF.index = range(0, len(AverageWideDF))
AverageWideDF[AverageWideDF.columns[0:15]] = AverageWideDF.loc[:,AverageWideDF.columns[0:15]].apply(pd.to_numeric)

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
Calculate change of each trial per subject with respect to baseline Normal average values
"""
#deltaDF = pd.DataFrame(columns = data.columns)
#Conditions = list(['Crossover', 'Narrow', 'Wide'])
#Limb = list(['Dominant', 'NonDominant'])
#
#for condition in Conditions:
#    for limb in Limb:
#        for subject in range(0, 16):
#            tempDF = pd.DataFrame()
#            partialDF = data.loc[(data['Condition']==condition)&(data['Subject']==subject)&(data['Limb']==limb), :]
#            
#            tempNorm = AverageNormDF.loc[(AverageNormDF['Subject']==subject)&(AverageNormDF['Limb']==limb), :]
#            
#            #Take difference between partialDF and tempNorm
#            tempDF = partialDF.loc[:, :'Stride_Width_Mean'] - tempNorm.values[0][0:14]
#            tempDF = pd.concat([tempDF, partialDF.loc[:, 'Subject':]], axis = 1)
#            
#            deltaDF = pd.concat([deltaDF, tempDF], axis = 0)
#            
#deltaDF[deltaDF.columns[0:15]] = deltaDF.loc[:, deltaDF.columns[0:15]].apply(pd.to_numeric)
#deltaDF.index = range(0, len(deltaDF))

"""
Perform one hot encoding for categorical features------------------------------------------------------------------------------------------
"""
#EncodedDF = pd.get_dummies(deltaDF, columns=['Subject', 'Condition', 'Limb'])
EncodedDF = pd.get_dummies(data, columns=['Subject', 'Condition', 'Limb'])

AverageNormDF = pd.get_dummies(AverageNormDF, columns=['Subject', 'Limb'])
AverageCrossoverDF = pd.get_dummies(AverageCrossoverDF, columns=['Subject', 'Limb'])
AverageNarrowDF = pd.get_dummies(AverageNarrowDF, columns=['Subject', 'Limb'])
AverageWideDF = pd.get_dummies(AverageWideDF, columns=['Subject', 'Limb'])
"""
------------------------------------------------------------------------------------------------------------------------------------------
Subject-Specific Lasso Regressions
"""
#Select subject
subject = list(EncodedDF.columns[14:30])

KAMarray = np.empty(0)
SubjectArray = np.zeros([1, 11])
for i in range(0, 16):
    SubjectDF = EncodedDF.loc[(EncodedDF[subject[i]]==1)&(EncodedDF['Limb_Dominant']==0),:]
    
    Features = SubjectDF.loc[:, ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Width_Mean']]
    
    #Polynomial features order 2 with only interaction terms
    poly = PolynomialFeatures(2, interaction_only=True)
    PolyFeatures = poly.fit_transform(Features)
    
    Target = SubjectDF.loc[:, 'PKFM']
    #Target = SubjectDF.loc[:, 'PKAM1']
    
    #Perform GridsearchCV to determine alpha for Ridge
    parameters = {'alpha': [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 4e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 100]}
    
    ridge = Ridge(max_iter=100000, normalize=True)
    ridgeCV = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=20)
    ridgeCV.fit(PolyFeatures, Target)
    
    ridgeReg = Ridge(alpha=ridgeCV.best_params_['alpha'], max_iter=100000, normalize=True)
    
    ridgeReg.fit(PolyFeatures, Target)
    
    print('Subject ' + str(i+1))
    print(ridgeReg.score(PolyFeatures, Target))
    print(ridgeReg.coef_)
#    
    SubjectArray = np.append(SubjectArray, [[ridgeReg.coef_[1], ridgeReg.coef_[2], ridgeReg.coef_[3], ridgeReg.coef_[4],
                                      ridgeReg.coef_[5], ridgeReg.coef_[6], ridgeReg.coef_[7], ridgeReg.coef_[8],
                                      ridgeReg.coef_[9], ridgeReg.coef_[10], ridgeReg.score(PolyFeatures, Target)]], axis = 0)
    
    NormalGait = AverageNormDF.loc[(AverageNormDF[subject[i]]==1)&(AverageNormDF['Limb_Dominant']==0),
                                   ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Width_Mean']]
    
    CrossoverGait = AverageCrossoverDF.loc[(AverageCrossoverDF[subject[i]]==1)&(AverageCrossoverDF['Limb_Dominant']==0),
                                   ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Width_Mean']]
    
    NarrowGait = AverageNarrowDF.loc[(AverageNarrowDF[subject[i]]==1)&(AverageNarrowDF['Limb_Dominant']==0),
                                   ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Width_Mean']]
    
    WideGait = AverageWideDF.loc[(AverageWideDF[subject[i]]==1)&(AverageWideDF['Limb_Dominant']==0),
                                   ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Width_Mean']]
    
    NewGait = NormalGait.loc[:,['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed']]
    NewGait = pd.concat([NewGait, CrossoverGait.loc[:, 'Stride_Width_Mean']], axis=1)
    
    
    poly = PolynomialFeatures(2, interaction_only=True)
    PolyFeatures = poly.fit_transform(NewGait)
    
    newKAM = ridgeReg.predict(PolyFeatures)
    print('Normal ' + str(newKAM))
    KAMarray = np.append(KAMarray, newKAM)
"""
----------------------------------------------------------------------------------------------------------------------------------------
One sample bootstrap hypothesis test
"""
def bootstrapTest(dataframe, testValue, samples=10000):
    difference = np.empty(0)
    meanDiffObs = abs(mean(dataframe) - testValue)
    newData = np.array(dataframe) - mean(dataframe) + float(testValue)
    
    for i in range(0, 10000):
        newSample = np.random.choice(newData, size=5)
        Diff = abs(mean(newSample) - float(testValue))
        difference = np.append(difference, float(Diff))
    
    p_value = len(difference[difference>float(meanDiffObs)])/(samples)
    
    return p_value
    
    
controlledDF = pd.read_csv("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/ControlledKFM.csv")
controlledDF = pd.get_dummies(controlledDF, columns=['Subject'])
controlledDF = controlledDF.loc[(controlledDF['Limb_Dominant']==0),:]
NormalDF = EncodedDF.loc[(EncodedDF['Condition_Normal']==1)&(EncodedDF['Limb_Dominant']==0),:]

subject = list(EncodedDF.columns[14:30])
crossoverPValue = np.empty(0)
narrowPValue = np.empty(0)
widePValue = np.empty(0)
for i in range(0, 16):
    subjectNormal = NormalDF.loc[NormalDF[subject[i]]==1, 'PKFM']
    
    #Perform bootstrap hypothesis test
    p_value = bootstrapTest(subjectNormal, controlledDF.loc[controlledDF[subject[i]]==1, 'Crossover'])
    crossoverPValue = np.append(crossoverPValue, p_value)
    
    p_value = bootstrapTest(subjectNormal, controlledDF.loc[controlledDF[subject[i]]==1, 'Narrow'])
    narrowPValue = np.append(narrowPValue, p_value)
    
    p_value = bootstrapTest(subjectNormal, controlledDF.loc[controlledDF[subject[i]]==1, 'Wide'])
    widePValue = np.append(widePValue, p_value)





























