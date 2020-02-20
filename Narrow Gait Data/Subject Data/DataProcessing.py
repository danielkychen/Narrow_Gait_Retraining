# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:24:41 2019

Script for importing subject csv files from Gait study and Cleaning to create Table 1

@author: Daniel Chen
"""

import pandas as pd
from pathlib import Path
import numpy as np
import statistics as stat
import random
from itertools import combinations


"""
FUNCTIONS-----------------------------------------------------------------------------------------------------------------
"""
def VariableSelect(VariableName, DataFrame):
    return (DataFrame.loc[:,DataFrame.loc[0].str.find(VariableName)!=-1])

def MeanSD(DataFrame, Table, VariableName):
    for i in conditions:
        tempColumn = np.empty(0)
        for j in range(1,len(DataFrame.index)):
            Mean = round(stat.mean(DataFrame.loc[j, (DataFrame.columns.str.find(i)!=-1)].astype(float)),3)
            SD = round(stat.stdev(DataFrame.loc[j, (DataFrame.columns.str.find(i)!=-1)].astype(float)),3)
            tempColumn = np.append(tempColumn,(str(Mean) + " " + u"\u00B1" + " " + str(SD)))
        
        Table[Table.columns[(Table.columns.str.find(i)!=-1)&(Table.columns.str.find(VariableName)!=-1)]] = tempColumn
    return Table;
    
def BootstrapMeans(group1, group2, iterations):
    group1Mean = stat.mean(group1)
    group2Mean = stat.mean(group2)
    grandMean = stat.mean(np.append(group1, group2))
    Tobs = group1Mean - group2Mean
    
    #Generate new samples for group1 and group2
    new_group1 = [group1[i] - group1Mean + grandMean for i in range(0, len(group1))]
    new_group2 = [group2[i] - group2Mean + grandMean for i in range(0, len(group2))]
    
    T = np.empty(0)
    for i in range(0, iterations):
        sample1 = np.empty(0)
        sample2 = np.empty(0)
        for j in range(0, len(new_group1)):
            sample1 = np.append(sample1, new_group1[random.randint(0, len(new_group1)-1)])
            sample2 = np.append(sample2, new_group2[random.randint(0, len(new_group2)-1)])
    
        T = np.append(T, stat.mean(sample1) - stat.mean(sample2))
    
    return (1+sum((abs(T)>=abs(Tobs))))/(iterations+1)
"""
--------------------------------------------------------------------------------------------------------------------------
"""


subjectDataDir = Path("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/Subject Data/")

RawDataDF = pd.read_csv(subjectDataDir / "FiveTrialsRawData.csv")

"""
Keep only the columns with PKAM1, PKAM2, KFM, and KAMImpulse
"""
pKAM1_DF = VariableSelect("DomPKAM1", RawDataDF)
pKAM2_DF = VariableSelect("DomPKAM2", RawDataDF)
KFM_DF = VariableSelect("DomPKFM", RawDataDF)
Impulse_DF = VariableSelect("DomKAMImpulse", RawDataDF)

#Split into Dominant and Non-dominant dataframes
pKAM1_NonDomDF = VariableSelect("NonDom", pKAM1_DF)
pKAM1_DomDF = pKAM1_DF.drop(pKAM1_NonDomDF.columns, axis = 1)

pKAM2_NonDomDF = VariableSelect("NonDom", pKAM2_DF)
pKAM2_DomDF = pKAM2_DF.drop(pKAM2_NonDomDF.columns, axis = 1)

KFM_NonDomDF = VariableSelect("NonDom", KFM_DF)
KFM_DomDF = KFM_DF.drop(KFM_NonDomDF.columns, axis = 1)

Impulse_NonDomDF = VariableSelect("NonDom", Impulse_DF)
Impulse_DomDF = Impulse_DF.drop(Impulse_NonDomDF.columns, axis = 1)

"""
Generate Dataframe with mean+/-SD for every condition (Table 1)----------------------------------------------------------------------
"""
data = np.empty([16, 16])
DomTable1 = pd.DataFrame(data, columns = ["NormalDom_PKAM1","NormalDom_PKAM2", "NormalDom_KFM", "NormalDom_Impulse",
                                 "CrossoverDom_PKAM1","CrossoverDom_PKAM2", "CrossoverDom_KFM", "CrossoverDom_Impulse",
                                 "NarrowDom_PKAM1","NarrowDom_PKAM2", "NarrowDom_KFM", "NarrowDom_Impulse",
                                 "WideDom_PKAM1","WideDom_PKAM2", "WideDom_KFM", "WideDom_Impulse"])
    
NonDomTable1 = pd.DataFrame(data, columns = ["NormalNonDom_PKAM1","NormalNonDom_PKAM2", "NormalNonDom_KFM", "NormalNonDom_Impulse",
                                       "CrossoverNonDom_PKAM1","CrossoverNonDom_PKAM2", "CrossoverNonDom_KFM", "CrossoverNonDom_Impulse",
                                       "NarrowNonDom_PKAM1","NarrowNonDom_PKAM2", "NarrowNonDom_KFM", "NarrowNonDom_Impulse",
                                       "WideNonDom_PKAM1","WideNonDom_PKAM2", "WideNonDom_KFM", "WideNonDom_Impulse"])

#conditions = list(["Crossover", "Narrow", "Normal", "Wide"])
conditions = list(["Normal", "Crossover", "Narrow", "Wide"])

MeanSD(pKAM1_DomDF, DomTable1, "PKAM1")
MeanSD(pKAM2_DomDF, DomTable1, "PKAM2")
MeanSD(KFM_DomDF, DomTable1, "KFM")
MeanSD(Impulse_DomDF, DomTable1, "Impulse")

MeanSD(pKAM1_NonDomDF, NonDomTable1, "PKAM1")
MeanSD(pKAM2_NonDomDF, NonDomTable1, "PKAM2")
MeanSD(KFM_NonDomDF, NonDomTable1, "KFM")
MeanSD(Impulse_NonDomDF, NonDomTable1, "Impulse")
        
"""
-------------------------------------------------------------------------------------------------------------------------------------
"""
"""
Generate Dataframe with mean+/-SD for every condition (Table 2)
"""
DomTable2 = pd.DataFrame(data, columns = ["NormalDom_FPAngleAtPKAM1","NormalDom_SW","NormalDom_Speed","NormalDom_Trunk",
                                          "CrossoverDom_FPAngleAtPKAM1","CrossoverDom_SW", "CrossoverDom_Speed", "CrossoverDom_Trunk",
                                          "NarrowDom_FPAngleAtPKAM1","NarrowDom_SW", "NarrowDom_Speed", "NarrowDom_Trunk",
                                          "WideDom_FPAngleAtPKAM1","WideDom_SW", "WideDom_Speed", "WideDom_Trunk"])
    
NonDomTable2 = pd.DataFrame(data, columns = ["NormalNonDom_FPAngleAtPKAM1","NormalNonDom_SW","NormalNonDom_Speed","NormalNonDom_Trunk",
                                          "CrossoverNonDom_FPAngleAtPKAM1","CrossoverNonDom_SW", "CrossoverNonDom_Speed", "CrossoverNonDom_Trunk",
                                          "NarrowNonDom_FPAngleAtPKAM1","NarrowNonDom_SW", "NarrowNonDom_Speed", "NarrowNonDom_Trunk",
                                          "WideNonDom_FPAngleAtPKAM1","WideNonDom_SW", "WideNonDom_Speed", "WideNonDom_Trunk"])

"""
Keep only the columns with FPA, SW, Speed, Trunk Sway
"""
FPA_DF = VariableSelect("DomFPAngleAtPKAM1_MEAN", RawDataDF)
SW_DF = VariableSelect("Stride_Width", RawDataDF)
Speed_DF = VariableSelect("Speed", RawDataDF)
Trunk_DF = VariableSelect("DomTLAngleAtPKAM1_MEAN", RawDataDF)

#Split into Dominant and Non-dominant dataframes
FPA_NonDomDF = VariableSelect("NonDom", FPA_DF)
FPA_DomDF = FPA_DF.drop(FPA_NonDomDF.columns, axis = 1)

Trunk_NonDomDF = VariableSelect("NonDom", Trunk_DF)
Trunk_DomDF = Trunk_DF.drop(Trunk_NonDomDF.columns, axis = 1)

MeanSD(FPA_DomDF, DomTable2, "FPAngle")
MeanSD(FPA_NonDomDF, NonDomTable2, "FPAngle")

MeanSD(SW_DF, DomTable2, "SW")
MeanSD(SW_DF, NonDomTable2, "SW")

MeanSD(Speed_DF, DomTable2, "Speed")
MeanSD(Speed_DF, NonDomTable2, "Speed")

MeanSD(Trunk_DomDF, DomTable2, "Trunk")
MeanSD(Trunk_NonDomDF, NonDomTable2, "Trunk")

"""
----------------------------------------------------------------------------------------------------------------------------------------
Testing for statistical differences between different conditions and parameters

Bootstrap hypothesis test for equality of means
"""
ConditionListNames = list(["Normal", "Crossover", "Narrow", "Wide"])
PairNames = list(combinations(ConditionListNames, 2))
symbols = list(["a", "b", "c", "d", "e", "f"])
#DomDataFrames = list([pKAM1_DomDF, pKAM2_DomDF, KFM_DomDF, Impulse_DomDF])
#NonDomDataFrames = list([pKAM1_NonDomDF, pKAM2_NonDomDF, KFM_NonDomDF, Impulse_NonDomDF])
DomDataFrames = list([FPA_DomDF, SW_DF, Speed_DF, Trunk_DomDF])
NonDomDataFrames = list([FPA_NonDomDF, SW_DF, Speed_DF, Trunk_NonDomDF])
#VariableNames = list(["PKAM1", "PKAM2", "KFM", "Impulse"])
VariableNames = list(["FPA", "SW", "Speed", "Trunk"])
#OutputDataframe = DomTable1
#OutputDataframe = NonDomTable1
OutputDataframe = DomTable2
#OutputDataframe = NonDomTable2

for dataframe in range(0, len(DomDataFrames)):
    for subject in range(1, len(DomDataFrames[dataframe])):
        Normal = list(map(float,DomDataFrames[dataframe].loc[subject, DomDataFrames[dataframe].columns[(DomDataFrames[dataframe].columns.str.find("Normal")!=-1)]]))
        Crossover = list(map(float,DomDataFrames[dataframe].loc[subject, DomDataFrames[dataframe].columns[(DomDataFrames[dataframe].columns.str.find("Crossover")!=-1)]]))
        Narrow = list(map(float,DomDataFrames[dataframe].loc[subject, DomDataFrames[dataframe].columns[(DomDataFrames[dataframe].columns.str.find("Narrow")!=-1)]]))
        Wide = list(map(float,DomDataFrames[dataframe].loc[subject, DomDataFrames[dataframe].columns[(DomDataFrames[dataframe].columns.str.find("Wide")!=-1)]]))
        
        ConditionList = list([Normal, Crossover, Narrow, Wide])
        
        """
        Loop through each Dataframe and subject
        """
        Pairs = list(combinations(ConditionList, 2))
        
        significance = np.empty(0)
        alpha = 0.05
        for i in range(0, len(Pairs)):
            #Add "*" to significance array is pvalue is < 0.05/n (bonferroni correction)
            if BootstrapMeans(Pairs[i][0], Pairs[i][1], 10000) < (alpha/len(Pairs)):
                significance = np.append(significance, symbols[i])
            else:
                significance = np.append(significance, "")
        
        #add significance symbols to DomTable/NonDomTable dataframes to Columns which match 
        #variableName = "PKAM1"
        for pair in range(0, len(PairNames)):
            temp = OutputDataframe.columns[(OutputDataframe.columns.str.find(VariableNames[dataframe])!=-1)&((OutputDataframe.columns.str.find(PairNames[pair][0])!=-1)|(OutputDataframe.columns.str.find(PairNames[pair][1])!=-1))]
            OutputDataframe.loc[subject-1, temp[0]] = OutputDataframe.loc[subject-1, temp][0]+significance[pair]
            OutputDataframe.loc[subject-1, temp[1]] = OutputDataframe.loc[subject-1, temp][1]+significance[pair]

"""
Reformat and Export DataFrames to CSV
"""
DomTable1_mod = pd.DataFrame()
DomTable1_mod = pd.concat([DomTable1_mod, DomTable1[DomTable1.columns[DomTable1.columns.str.find("PKAM1")!=-1]]], axis=1)
DomTable1_mod = pd.concat([DomTable1_mod, DomTable1[DomTable1.columns[DomTable1.columns.str.find("PKAM2")!=-1]]], axis=1)
DomTable1_mod = pd.concat([DomTable1_mod, DomTable1[DomTable1.columns[DomTable1.columns.str.find("KFM")!=-1]]], axis=1)
DomTable1_mod = pd.concat([DomTable1_mod, DomTable1[DomTable1.columns[DomTable1.columns.str.find("Impulse")!=-1]]], axis=1)

NonDomTable1_mod = pd.DataFrame()
NonDomTable1_mod = pd.concat([NonDomTable1_mod, NonDomTable1[NonDomTable1.columns[NonDomTable1.columns.str.find("PKAM1")!=-1]]], axis=1)
NonDomTable1_mod = pd.concat([NonDomTable1_mod, NonDomTable1[NonDomTable1.columns[NonDomTable1.columns.str.find("PKAM2")!=-1]]], axis=1)
NonDomTable1_mod = pd.concat([NonDomTable1_mod, NonDomTable1[NonDomTable1.columns[NonDomTable1.columns.str.find("KFM")!=-1]]], axis=1)
NonDomTable1_mod = pd.concat([NonDomTable1_mod, NonDomTable1[NonDomTable1.columns[NonDomTable1.columns.str.find("Impulse")!=-1]]], axis=1)

DomTable2_mod = pd.DataFrame()
DomTable2_mod = pd.concat([DomTable2_mod, DomTable2[DomTable2.columns[DomTable2.columns.str.find("FPA")!=-1]]], axis=1)
DomTable2_mod = pd.concat([DomTable2_mod, DomTable2[DomTable2.columns[DomTable2.columns.str.find("SW")!=-1]]], axis=1)
DomTable2_mod = pd.concat([DomTable2_mod, DomTable2[DomTable2.columns[DomTable2.columns.str.find("Speed")!=-1]]], axis=1)
DomTable2_mod = pd.concat([DomTable2_mod, DomTable2[DomTable2.columns[DomTable2.columns.str.find("Trunk")!=-1]]], axis=1)

NonDomTable2_mod = pd.DataFrame()
NonDomTable2_mod = pd.concat([NonDomTable2_mod, NonDomTable2[NonDomTable2.columns[NonDomTable2.columns.str.find("FPA")!=-1]]], axis=1)
NonDomTable2_mod = pd.concat([NonDomTable2_mod, NonDomTable2[NonDomTable2.columns[NonDomTable2.columns.str.find("SW")!=-1]]], axis=1)
NonDomTable2_mod = pd.concat([NonDomTable2_mod, NonDomTable2[NonDomTable2.columns[NonDomTable2.columns.str.find("Speed")!=-1]]], axis=1)
NonDomTable2_mod = pd.concat([NonDomTable2_mod, NonDomTable2[NonDomTable2.columns[NonDomTable2.columns.str.find("Trunk")!=-1]]], axis=1)