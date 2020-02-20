# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:14:19 2019

Script for Import CleanData.csv and plotting Scatter Matrices to explore trends

@author: Daniel
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
%matplotlib qt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from scipy.cluster.hierarchy import dendrogram, linkage
%config InlineBackend.figure_format='retina'
plt.style.use('ggplot')

plt.rcParams.update({'font.size': 2})
matplotlib.rcParams['lines.linewidth'] = 0.1

data = pd.read_csv("C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/CleanData.csv")

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
-----------------------------------------------------------------------------------------------------------------------------------------------------
Calculate change of each trial per subject with respect to baseline Normal average values
"""
deltaDF = pd.DataFrame(columns = data.columns)
Conditions = list(['Crossover', 'Narrow', 'Wide'])
Limb = list(['Dominant', 'NonDominant'])

for condition in Conditions:
    for limb in Limb:
        for subject in range(0, 16):
            tempDF = pd.DataFrame()
            partialDF = data.loc[(data['Condition']==condition)&(data['Subject']==subject)&(data['Limb']==limb), :]
            
            tempNorm = AverageNormDF.loc[(AverageNormDF['Subject']==subject)&(AverageNormDF['Limb']==limb), :]
            
            #Take difference between partialDF and tempNorm
            tempDF = partialDF.loc[:, :'Stride_Width_Mean'] - tempNorm.values[0][0:14]
            tempDF = pd.concat([tempDF, partialDF.loc[:, 'Subject':]], axis = 1)
            
            deltaDF = pd.concat([deltaDF, tempDF], axis = 0)
            
deltaDF[deltaDF.columns[0:15]] = deltaDF.loc[:, deltaDF.columns[0:15]].apply(pd.to_numeric)
deltaDF.index = range(0, len(deltaDF))

"""
Perform one hot encoding for categorical features
"""
EncodedDF = pd.get_dummies(deltaDF, columns=['Subject', 'Condition', 'Limb'])

#Select specific subject
subject = list(EncodedDF.columns[14:30])

i = 0
SubjectCrossover = EncodedDF.loc[(EncodedDF[subject[i]]==1)&(EncodedDF['Condition_Crossover']),:]
SubjectNarrow = EncodedDF.loc[(EncodedDF[subject[i]]==1)&(EncodedDF['Condition_Narrow']),:]
SubjectWide = EncodedDF.loc[(EncodedDF[subject[i]]==1)&(EncodedDF['Condition_Wide']),:]

"""
Drop 'non-dominant' and 'wide' columns to avoid dummy variable trap
"""
#SubjectDF = SubjectDF.drop(columns=['Condition_Wide', 'Limb_NonDominant'])

Features = SubjectDF.loc[:, ['FPAngleAtPKAM1', 'TLAngleAtPKAM1', 'Speed', 'Stride_Length_Mean', 'Stride_Width_Mean', 'Limb_Dominant']]
                            # 'Condition_Crossover', 'Condition_Narrow', 'Limb_Dominant']]

#Polynomial features order 2 then LASSO
poly = PolynomialFeatures(2, interaction_only=True)
PolyFeatures = poly.fit_transform(Features)

Target = SubjectDF.loc[:, 'KAMImpulse']

regression = Lasso(alpha=0.0002, max_iter=100000).fit(PolyFeatures, Target)
regression.score(PolyFeatures, Target)
regression.coef_

"""
Hierarchical clustering using PKAM1, KFM, Impulse
"""
ClusterFeatures = EncodedDF.loc[:, ['PKAM1', 'PKFM', 'KAMImpulse']]

link = linkage(ClusterFeatures)

dendrogram(link,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)

plt.figure()
plt.show()





"""---------------------------------------------------------------------------------------------------------------------------------------------------
Cluster deltaDF to identify trends and groups
"""
"""
Perform one hot encoding for categorical features
"""
EncodedDF = pd.get_dummies(deltaDF, columns=['Subject', 'Condition', 'Limb'])

#Normalize values in each column (except categorical features) to have mean of 0 and variance of 1-----------------------------------------------------
deltaDF_norm = StandardScaler().fit_transform(deltaDF.loc[:,:'Stride_Width_Mean'])
deltaDF_norm = pd.DataFrame(deltaDF_norm)

deltaDF_norm = pd.concat([deltaDF_norm, EncodedDF.loc[:, 'Subject_0':]], axis = 1)
deltaDF_norm.columns = EncodedDF.columns
#-------------------------------------------------------------------------------------------------------------------------------------------------------
"""
PCA
"""
pca = PCA(n_components=5)
pComponents = pca.fit_transform(Features)

#Plot explained variance
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

PCA_components = pd.DataFrame(pComponents)

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

"""
K Means clustering of Principle Components
"""
ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#Use 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(PCA_components.iloc[:,:3])

#Combine PCA_components dataframe with categorical columns
PCA_DF = pd.concat([PCA_components.iloc[:, :3], EncodedDF.loc[:, 'Subject_0':]], axis=1)
Labels = pd.DataFrame(kmeans.labels_, columns=['Label'])
PCA_DF = pd.concat([PCA_DF, Labels], axis=1)

"""
Can plot PCA above at: https://chart-studio.plot.ly/create
"""
"""
Hierarchical clustering (dendrogram)
"""
#Segment into Dominant/Nondominant and then conditions
DomDF = deltaDF_norm.loc[deltaDF_norm["Limb_Dominant"]==1, :]
NonDomDF = deltaDF_norm.loc[deltaDF_norm["Limb_NonDominant"]==1, :]

#Scatter Matrices
pd.scatter_matrix(DomDF)

CrossoverDomDF = DomDF.loc[DomDF["Condition_Crossover"]==1, :]
NarrowDomDF = DomDF.loc[DomDF["Condition_Narrow"]==1, :]
WideDomDF = DomDF.loc[DomDF["Condition_Wide"]==1, :]

link = linkage(deltaDF_norm)

dendrogram(link,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.savefig('C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/Hierarchical.png', dpi=4000)
plt.show()

"""
Plot correlation/scatter matrices of just FPA, SW, Speed, and Trunk Sway (TL)
"""
subjectList = list(DomDF.columns[14:30])

correlationMatrix = pd.DataFrame()
for subject in subjectList:
    kinematicDF = DomDF.loc[DomDF[subject]==1,['FPAngleAtPKAM1','TLAngleAtPKAM1','Speed','Stride_Length_Mean','Stride_Width_Mean']]
    #Subject = kinematicDF.loc[kinematicDF["Subject_0"]==1,:"Stride_Width_Mean"]
    Corr = kinematicDF.corr()
    correlationMatrix = pd.concat([correlationMatrix, Corr], axis = 0)

correlationMatrix = correlationMatrix.loc[correlationMatrix.index=='Stride_Width_Mean',:]
correlationMatrix = correlationMatrix.drop(['FPAngleAtPKAM1'], axis=1)


#Matrix = pd.scatter_matrix(Subject, s=0.2)
#for ax in Matrix.ravel():
#    ax.set_xlabel(ax.get_xlabel(), fontsize = 5, rotation = 90)
#    ax.set_ylabel(ax.get_ylabel(), fontsize = 5, rotation = 0)
#    
#    
##plt.tight_layout()
#plt.gcf().subplots_adjust(bottom=0.50, left=0.15)
#plt.savefig('C:/Users/danth/Documents/Post Doc/Gait Retraining/Narrow Gait Data/DomScatter.png', dpi=4000)
#plt.show()




#"""
#Separate scatter matrices for Limb
#"""
#Dom = EncodedDF.loc[(EncodedDF["Limb_Dominant"] == 1), :]
#NonDom = EncodedDF.loc[(EncodedDF["Limb_NonDominant"] == 1), :]
#
#DomNormal = Dom.loc[(Dom["Condition_Normal"] == 1), :]
#DomCrossover = Dom.loc[(Dom["Condition_Crossover"] == 1), :]
#DomNarrow = Dom.loc[(Dom["Condition_Narrow"] == 1), :]
#DomWide = Dom.loc[(Dom["Condition_Wide"] == 1), :]
#
#NonDomNormal = NonDom.loc[(NonDom["Condition_Normal"] == 1), :]
#NonDomCrossover = NonDom.loc[(NonDom["Condition_Crossover"] == 1), :]
#NonDomNarrow = NonDom.loc[(NonDom["Condition_Narrow"] == 1), :]
#NonDomWide = NonDom.loc[(NonDom["Condition_Wide"] == 1), :]
#

