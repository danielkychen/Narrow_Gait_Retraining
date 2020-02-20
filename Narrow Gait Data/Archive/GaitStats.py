# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:02:48 2019

This script reads GaitData.csv and explores the data

@author: Daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GaitDF = pd.read_csv("C:\\Users\\danth\\Documents\\Post Doc\\Gait Retraining\\Narrow Gait Data\\GaitData.csv")

"""
Compare PKAM between different conditions for each subject individually
"""
