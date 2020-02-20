# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:03:55 2019

@author: Daniel
"""

import pandas as pd
from pathlib import Path
from mlxtend.evaluate import permutation_test
from statistics import mean

"""
Functions
"""
def StandardHeading(dataframe):
    for i in range(len(dataframe.columns)):
        if (i)
    columnNames = dataframe.columns
    
    print(columnNames)

Directory = Path("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/Subject Data/Subject01/")

filePath = Directory / "Subject01Normal.csv"
Normal = pd.read_csv(filePath)

filePath = Directory / "Subject01Crossover.csv"
Crossover = pd.read_csv(filePath)

filePath = Directory / "Subject01Narrow.csv"
Narrow = pd.read_csv(filePath)

filePath = Directory / "Subject01WideDom.csv"
WideDom = pd.read_csv(filePath)

filePath = Directory / "Subject01WideNonDom.csv"
WideNonDom = pd.read_csv(filePath)

"""
Create dataframe for mean and standard deviation
"""
MeanDF = pd.DataFrame(columns=['Dom_Normal','Dom_Crossover','Dom_Narrow','Dom_Wide','NonDom_Normal','NonDom_Crossover','NonDom_Narrow','NonDom_Wide'])
MeanDF['Dom_Normal'] = list(['0','0','0','0','0','0','0','0','0'])
MeanDF.rename(index={0:'1st KAM peak',1:'KFM peak',2:'KAM angular impulse',3:'2nd KAM peak',4:'FPA',5:'Step Width',6:'Walking speed',7:'Trunk Sway',8:'Stride Length'},inplace=True)

StdDF = pd.DataFrame(columns=['Dom_Normal','Dom_Crossover','Dom_Narrow','Dom_Wide','NonDom_Normal','NonDom_Crossover','NonDom_Narrow','NonDom_Wide'])
StdDF['Dom_Normal'] = list(['0','0','0','0','0','0','0','0'])
StdDF.rename(index={0:'1st KAM peak',1:'KFM peak',2:'KAM angular impulse',3:'2nd KAM peak',4:'FPA',5:'Step Width',6:'Walking speed',7:'Trunk Sway',8:'Stride Length'},inplace=True)

"""
Calculate mean and SD for tables
"""
MeanDF['Dom_Normal']['1st KAM peak'] = str(mean(Normal['NormalDomPKAM1_MEAN.X']))
MeanDF['Dom_Normal']['KFM peak'] = str(mean(Normal['NormalDomPKEM_MEAN.X']))
#MeanDF['Dom_Normal']['KAM angular impulse'] = str(mean(Normal['NormalDomPKFM_MEAN.X']))
MeanDF['Dom_Normal']['2nd KAM peak'] = str(mean(Normal['NormalDomPKAM2_MEAN.X']))
MeanDF['Dom_Normal']['FPA'] = str(mean(Normal['NormalDomFPAngleAverage_MEAN.X']))
MeanDF['Dom_Normal']['Step Width'] = str(mean(Normal['Stride_Width_Mean.X']))
MeanDF['Dom_Normal']['Walking speed'] = str(mean(Normal['Speed.X']))
MeanDF['Dom_Normal']['Trunk Sway'] = str(mean(Normal['NormalDomTLAnglePKAM1_MEAN.X']))
MeanDF['Dom_Normal']['Stride Length'] = str(mean(Normal['Stride_Length_Mean.X']))



"""
Perform Pairwise Permutation test on difference of means with Bonferroni correction rejection p-value < alpha/N
"""
p_value = permutation_test(Crossover['CrossoverDomPKAM1_MEAN.X'], Normal['NormalDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

p_value1 = permutation_test(Narrow['NarrowDomPKAM1_MEAN.X'], Normal['NormalDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

p_value2 = permutation_test(WideDom['WideDomDomPKAM1_MEAN.X'], Normal['NormalDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

p_value3 = permutation_test(Crossover['CrossoverNonDomPKAM1_MEAN.X'], Normal['NormalNonDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

p_value4 = permutation_test(Narrow['NarrowNonDomPKAM1_MEAN.X'], Normal['NormalNonDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

p_value5 = permutation_test(WideDom['WideDomDomPKAM1_MEAN.X'], Normal['NormalNonDomPKAM1_MEAN.X'], method='approximate', num_rounds=10000, seed=0)

Dom_Normal = list(['123', '234', '456'])


Subject = pd.DataFrame(columns=['Dom_Normal','Dom_Crossover','Dom_Narrow','Dom_Wide','NonDom_Normal','NonDom_Crossover','NonDom_Narrow','NonDom_Wide'])
Subject.rename(index={0:'1st KAM peak',1:'KFM peak',2:'KAM angular impulse',3:'2nd KAM peak',4:'FPA',5:'Step Width',6:'Walking speed',7:'Trunk Sway'},inplace=True)