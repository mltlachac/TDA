#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Code written by ML Tlachac in 2020
#For paper titled Topological Data Analysis to Engineer Features from Audio Signals for Depression Detection
#If this code is used, please cite paper

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


# In[273]:


split = 10
modelTypelist = ["SVC1", "kNN3", "RF"]
scorelist = ["F1", "AUC", "Accuracy"]
dataset = "MEs"
level = "sub"

noTDA = []
withTDA = []
onlyTDA = []
ttest = []
pvalue = []
ttest2 = []
pvalue2 = []
scoreSave = []
modelSave = []
flist = []
flistW = []
flistT = []

for score in scorelist:
    
    avW = []
    av = []
    avT = []
    
    for modelType in modelTypelist:
        
        modelSave.append(modelType)
        scoreSave.append(score)

        resultsDFw = pd.read_csv("ICMLA/" + dataset + str(split) + "bothall" + level + ".csv")

        df = pd.DataFrame()
        averagesW = []
        for parameter in sorted(set(resultsDFw.nFeatures)):
            resultsDF2 = resultsDFw[resultsDFw.method == modelType]
            df[str(parameter)] = list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])
            #print(len(list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])))
            averagesW.append(round(sum(resultsDF2[resultsDF2["nFeatures"] == parameter][score])/100,3))
        print(max(averagesW))
        avW.append(averagesW)

        plt.figure(figsize = (12, 5))
        plt.xlabel("Number of Features", fontsize = 15)
        plt.ylabel(score, fontsize = 15)
        plt.title(modelType+ " with TDA", fontsize = 20)
        df.boxplot()
        plt.ylim(0.5, 1)
        #plt.show()
        #plt.savefig("pcaViz/" + smile + str(split) + score + ".png")
        plt.close()

        #smile features only

        resultsDF = pd.read_csv("ICMLA/" + dataset + str(split) + "smile" + "all.csv")

        df = pd.DataFrame()
        averages = []
        for parameter in sorted(set(resultsDF.nFeatures)):
            resultsDF2 = resultsDF[resultsDF.method == modelType]
            df[str(parameter)] = list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])
            #print(len(list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])))
            averages.append(round(sum(resultsDF2[resultsDF2["nFeatures"] == parameter][score])/100,3))
        print(max(averages))
        av.append(averages)

        plt.figure(figsize = (12, 5))
        plt.xlabel("Number of Features", fontsize = 15)
        plt.ylabel(score, fontsize = 15)
        plt.title(modelType+ " without TDA", fontsize = 20)
        df.boxplot()
        plt.ylim(0.5, 1)
        #plt.show()
        #plt.savefig("pcaViz/smile" + str(split) + score + ".png")
        plt.close()
        
        
        resultsDFt = pd.read_csv("ICMLA/" + dataset + str(split) + "bettiall" + level + ".csv")
        
        df = pd.DataFrame()
        averages = []
        for parameter in sorted(set(resultsDF.nFeatures)):
            resultsDF2 = resultsDFt[resultsDFt.method == modelType]
            df[str(parameter)] = list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])
            #print(len(list(resultsDF2[resultsDF2["nFeatures"] == parameter][score])))
            averages.append(round(sum(resultsDF2[resultsDF2["nFeatures"] == parameter][score])/100,3))
        print(max(averages))
        avT.append(averages)

        plt.figure(figsize = (12, 5))
        plt.xlabel("Number of Features", fontsize = 15)
        plt.ylabel(score, fontsize = 15)
        plt.title(modelType + " without TDA", fontsize = 20)
        df.boxplot()
        plt.ylim(0.5, 1)
        #plt.show()
        #plt.savefig("pcaViz/smile" + str(split) + score + ".png")
        plt.close()
        
        
    featureCount = list(np.arange(5,101,5))

    featureList = []
    for model in av: #numpy array
        m = max(model) #best value for each method
        for a in range(0, len(model)):
            if model[a] == m: #check if value is max
                featureList.append(featureCount[a]) #if value is max, add to list
                break

    featureListW = []
    for model in avW:
        m = max(model)
        for a in range(0, len(model)):
            if model[a] == m:
                featureListW.append(featureCount[a])
                break
                
    featureListT = []
    for model in avT:
        m = max(model)
        for a in range(0, len(model)):
            if model[a] == m:
                featureListT.append(featureCount[a])
                break

    for m in range(0, len(modelTypelist)):
        flist.append(featureList[m])
        flistW.append(featureListW[m])
        flistT.append(featureListT[m])

        rvs1 = resultsDF[(resultsDF.method == modelTypelist[m]) & (resultsDF["nFeatures"] == featureList[m])][score]
        rvs2 = resultsDFw[(resultsDFw.method == modelTypelist[m]) & (resultsDFw["nFeatures"] == featureListW[m])][score]
        rvs3 = resultsDFt[(resultsDFt.method == modelTypelist[m]) & (resultsDFt["nFeatures"] == featureListT[m])][score]

        #print(modelTypelist[m] + " with split " + str(split))
        #print(score + " without TDA: " + str(round(sum(rvs1)/100,4)))
        #print(score + " with TDA: " + str(round(sum(rvs2)/100,4)))
        #print(stats.ttest_ind(rvs1,rvs2))


        noTDA.append(str(round(sum(rvs1)/100,3)))
        withTDA.append(str(round(sum(rvs2)/100,3)))
        onlyTDA.append(str(round(sum(rvs3)/100,3)))

        
        test = stats.ttest_ind(rvs1,rvs2)
        ttest.append(test[0])
        pvalue.append(test[1])
        test = stats.ttest_ind(rvs1,rvs3)
        ttest2.append(test[0])
        pvalue2.append(test[1])

newDF = pd.DataFrame()
newDF["model"] = modelSave
newDF["score"] = scoreSave
newDF["noTDA"] = noTDA
newDF["withTDA"] = withTDA
newDF["ttest"] = ttest
newDF["pvalue"] = pvalue
newDF["onlyTDA"] = onlyTDA
newDF["ttest2"] = ttest2
newDF["pvalue2"] = pvalue2
newDF["featureN"] = flist
newDF["featureW"] = flistW
newDF["featureT"] = flistT

newDF.to_csv("ICMLA/results" + dataset + str(split) + level + ".csv")
print(newDF)


# In[285]:


#distribution of PHQ plot
df = pd.read_csv("MEup.csv")
plt.figure(figsize = (5, 5))
plt.hist(df.score, bins = 26, color = 'k', edgecolor= "w")
plt.xlabel("PHQ-9 Score", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Moodable/EMU", fontsize = 15)
#plt.title("DAIC-WOZ", fontsize = 15)
plt.ylim(0,23)
plt.savefig("Ddist.png")

