# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:58:04 2019

Script for constructing a Group Regression Model

@author: Daniel
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

path = "C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/CleanData.csv"
data = pd.read_csv(path)

#One-hot encoding for categorical features
dataEncoded = pd.get_dummies(data, columns=['Subject', 'Condition', 'Limb'])

dataEncoded = dataEncoded.drop(columns=['Condition_Wide', 'Limb_NonDominant'])


KAM_dataEncoded = dataEncoded.drop(['FPAngleAverage', 'KAMImpulse', 'KneeAbdAngleAverage',
       'LimbPGRF1', 'LimbPGRF2', 'PKAM2', 'PKEM','PKFM'], axis=1)

#Subject-specific
subjectDF = KAM_dataEncoded.loc[(KAM_dataEncoded['Subject_15']==1)&(KAM_dataEncoded['Limb_Dominant']==1),:]




#Perform linear regression
X_features = subjectDF.loc[:, ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed',
       'Stride_Length_Mean', 'Stride_Width_Mean']]

poly = PolynomialFeatures(1)
PolyFeatures = poly.fit_transform(X_features)

Y = subjectDF.loc[:, 'PKAM1']

lassoReg = Lasso(alpha=0.00001)
lassoReg.fit(PolyFeatures, Y)
lassoReg.score(PolyFeatures, Y)
lassoReg.coef_


