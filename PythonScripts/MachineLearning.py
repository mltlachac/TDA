#!/usr/bin/env python
# coding: utf-8

# In[322]:


#Code written by ML Tlachac in 2020
#For paper titled Topological Data Analysis to Engineer Features from Audio Signals for Depression Detection
#If this code is used, please cite paper

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from scipy import stats
import collections
import operator
import argparse
import random
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from statistics import mean 
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn import utils
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.decomposition import PCA
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

randoms = [481998864, 689799321, 796360373, 325345551, 781053364, 425410490, 448592531, 477651899, 256556897, 950476446, 439161956, 617662138, 919221369, 372092462, 978558065, 915406186, 914758288, 270769509, 348581119, 620469471, 968622238, 493269528, 923889165, 187902000, 768516562, 656996274, 204570914, 478659400, 591118631, 455578751, 523453979, 904238395, 870935338, 65160836, 469733522, 301035177, 843432976, 931667506, 283989232, 77803117, 371210776, 231366353, 454473430, 335714437, 233937007, 131940380, 267081710, 208764677, 225578708, 684893704, 93911936, 333598779, 253843993, 390054067, 432395108, 730697387, 988951427, 963369310, 983748712, 206214635, 607442488, 783641874, 444298574, 799459448, 736269849, 222259535, 501043573, 914806112, 780691269, 993143254, 900823730, 946288505, 776711331, 393086523, 366784871, 181714875, 239540123, 101370413, 417433780, 288079126, 205915691, 73435964, 248074219, 582671864, 635043553, 338657949, 330517223, 804096498, 667716642, 995598949, 504427080, 778739823, 245211208, 96486247, 541502147, 5680657, 590309190, 5062322, 921199528, 188694207]

split = 10
smile = "betti"
level = "sub"
nFeatureList = list(np.arange(5,101,5))
modelTypelist = ["SVC1", "kNN3", "RF"]


# In[323]:


#load Data
df = pd.read_csv("ME" + level + ".csv")
#df2 = pd.read_csv("D" + level + ".csv")
#df = df.append(df2).reset_index(drop=True)
cnames = df.columns
print(df.shape)


# In[325]:


df0 = pd.DataFrame()
for c in cnames:
    df0[c] = df[c].fillna(0)

data = df0
data[data.columns[-1]] = np.where(data[data.columns[-1]] >= split, 1, 0)

#split data
if smile == "smile":
    featureSubsetS = data[data.columns[101:-2]] #include simle
elif smile == "betti":
    featureSubsetS = data[data.columns[1:101]] #include TDA
elif smile == "both":
    featureSubsetS = data[data.columns[1:-2]] #include smile, TDA
else:
    print("invalid smile option")

#featureSubsetT = data[data.columns[-101:-1]]
gender = list(data[data.columns[-2]].to_numpy())
featureSubsetS["gender"] = gender
target = data[data.columns[-1]]


# In[327]:


#create index
indexes = data[data.columns[0]]
data["pq"] = indexes
pq = data[data.columns[-1]]

#scale features
min_max_scaler = preprocessing.MinMaxScaler()                   #NEED TO SCALE BEFORE FEATURE SELECTION!
np_scaled = min_max_scaler.fit_transform(featureSubsetS)
featureSubset2 = pd.DataFrame(np_scaled)

#top 1 to 100 pca of features
pcaDF = []
for f in nFeatureList:
    pca = PCA(n_components=f)
    pca.fit(featureSubset2)
    X_pca = pca.transform(featureSubset2)
    newDF = pd.DataFrame(X_pca)
    #labeldata
    newDF = newDF.assign(pq = pq)#.assign(target = target)
    pcaDF.append(newDF)


# In[328]:


#parameters
mlist = []
flist = []
randomseed = []
#indexes in train and test sets
pqtrain = []
pqtest = []
#results
predictions = []
realvalues = []
#metrics
f1List = []
accList = []
aucList = []

for f in range(0, len(nFeatureList)):
    print("Feature" + str(nFeatureList[f]))
    fDF = pcaDF[f]
    
    for r in randoms:
        random.seed(r)

        #split into train (x) vs test (y)
        xdata, ydata, xtarget, ytarget = train_test_split(fDF, target, test_size=0.3, shuffle = True, random_state = r)
        ydataF = ydata[ydata.columns[:-1]] #remove test index for features
        yIndex = ydata[ydata.columns[-1]] #save test index
        
        #balence classes for training data
        featureSubset = xdata.assign(target = xtarget) #assign target to training data 

        if xtarget[xtarget == 0].shape[0] != xtarget[xtarget == 1].shape[0]:

            targetClassCount = collections.Counter(featureSubset[featureSubset.columns[-1]])
            majorityKey = max(targetClassCount, key=targetClassCount.get)
            majorityCount = targetClassCount[majorityKey]
            minorityKey = min(targetClassCount,  key=targetClassCount.get)
            minorityCount = targetClassCount[minorityKey]
            featureSubset_majority = featureSubset[featureSubset[featureSubset.columns[len(featureSubset.columns)-1]] == majorityKey]
            featureSubset_minority = featureSubset[featureSubset[featureSubset.columns[len(featureSubset.columns)-1]] == minorityKey]

            
            #downsampling
            featureSubset_majority_downsampled = resample(featureSubset_majority, replace=False, n_samples=minorityCount, random_state=r) # reproducible results
            featureSubset_downsampled = pd.concat([featureSubset_majority_downsampled, featureSubset_minority])
            featureSubset_downsampled = featureSubset_downsampled.sample(frac=1).reset_index(drop=True)
            featureSubset = featureSubset_downsampled
            token = ""
            
            #print(minorityCount)
            
        #create datasets from balenced data
        xtargetb = featureSubset[featureSubset.columns[-1]] #balenced target
        #print(len(xtargetb)/2)
        xIndex = featureSubset[featureSubset.columns[-2]] #save training index
        xdataFb = featureSubset[featureSubset.columns[:-2]] #remove target and index for balenced training data
        
        for modelType in modelTypelist:
            #select model
            if modelType == "SVC1":
                clf = svm.SVC(kernel='rbf', random_state=r)
            #elif modelType == "SVC2":
            #    clf = svm.SVC(kernel='linear', random_state=r)
            elif modelType == "RF":
                #clf = RandomForestClassifier(criterion="gini", max_depth=3, random_state=r)
                clf = RandomForestClassifier(random_state=r)
            elif modelType == "kNN3":
                clf = KNeighborsClassifier(n_neighbors=3)
              
            #fit model and make predictions
            model = clf.fit(xdataFb, xtargetb) #balenced training data
            result = model.predict(ydataF) #make predictions from testing data 
            
            #evaluate predictions - compare predictions (result) to real data (ytarget)
            auc = roc_auc_score(ytarget, result) 
            f1 = f1_score(ytarget, result)
            acc = accuracy_score(ytarget, result)
            
            #add to lists for df
            mlist.append(modelType)
            flist.append(nFeatureList[f])
            randomseed.append(r)
            pqtrain.append(xIndex.tolist())
            pqtest.append(yIndex.tolist())
            predictions.append(result.tolist())
            realvalues.append(ytarget.tolist())
            f1List.append(f1)
            accList.append(acc)
            aucList.append(auc)
            
#make df
newDF2 = pd.DataFrame()
newDF2["method"] = mlist
newDF2["nFeatures"] = flist
newDF2["randomseed"] = randomseed
newDF2["trainIndex"] = pqtrain
newDF2["testIndex"] = pqtest
newDF2["preds"] = predictions
newDF2["real"] = realvalues
newDF2["F1"] = f1List
newDF2["Accuracy"] = accList
newDF2["AUC"] = aucList

newDF2.to_csv("ICMLA/ME" + token + str(split) + smile + "all" + level + ".csv")


# In[ ]:




