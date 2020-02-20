# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:36:47 2019

This script is for cleaning and processing the raw data files for the Narrow SW gait study

@author: Daniel
"""

"""
Combine RawData1.csv and RawData2.csv into one file
"""

import pandas as pd
import numpy as np

RawData1 = pd.read_csv("C:\\Users\\danth\\Documents\\Post Doc\\Gait Retraining\\Narrow Gait Data\\RawData1.csv")
RawData2 = pd.read_csv("C:\\Users\\danth\\Documents\\Post Doc\\Gait Retraining\\Narrow Gait Data\\RawData2.csv")

"""
Cleaning up and formatting dataframe for RawData1.csv
"""
#Remove rows 1 and 2 from RawData1.csv
cleanRawData1 = RawData1.drop([RawData1.index[1],RawData1.index[2]])

#Rename column labels for cleanRawData1
#columns 0 to 94 don't need axis
#extract the desired column names between 0 to 94 and store in array
columnNames = list(cleanRawData1.iloc[0,0:95])
columnNames95Row0 = np.asarray(cleanRawData1.iloc[0,95:],dtype=np.str)
columnNames95Row1 = np.asarray(cleanRawData1.iloc[1,95:],dtype=np.str)

#Use zip function to combine columnNames95Row0 and columnNames95Row1
columnNames2 = [i + j for i, j in zip(columnNames95Row0,columnNames95Row1)]

columnsLabels = list(columnNames) + list(columnNames2)

cleanRawData1.columns = columnsLabels

cleanRawData1 = cleanRawData1.drop([cleanRawData1.index[0],cleanRawData1.index[1]])
cleanRawData1.index = range(0,len(cleanRawData1.index))

"""
Cleaning up and formatting dataframe for RawData2.csv
"""
cleanRawData2 = RawData2.drop(columns = ["Sex", "Condition order"])

"""
Combine cleanRawData1 and cleanRawData2
"""
GaitDF = pd.concat([cleanRawData1, cleanRawData2], axis=1)

"""
Export .csv file
"""
GaitDF.to_csv(path_or_buf="C:\\Users\\danth\\Documents\\Post Doc\\Gait Retraining\\Narrow Gait Data\\GaitData.csv", index=False)