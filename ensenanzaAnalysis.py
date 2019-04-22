# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:04:42 2019

@author: Steven
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Keras library
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 
#Serialization
import pickle;


#rms = sqrt(mean_squared_error(Y_test,y_pred1))

def isAnswerCorrect(df):
    df['respuesta'] = df['respuesta'].apply(str)
    df['respuestaJ'] = df['respuestaJ'].apply(str)
    isCorrect = (df['respuesta'] == df['respuestaJ']);
    return 1*isCorrect;
    
def convertSchool(user):
    if(user['escuela']=='Hartford International School'):
        return 1
    else:
        return 0

def giveScoreToAnswer(answer, timesMedian):
    time = answer['tiempo'];
    timeLimit = int(timesMedian[answer['loID'],answer['grado']]) + 2;
    if(time < timeLimit and answer['isCorrect'] == 1):
        return 20-(time/3);
    else:
        return 0;


def calculateIntensities(userPerformance):
    return (1-userPerformance.loc['score']/20);

def categorizeIntensity(currentUser):
    for i in range(0,5):
        col = 'LOIN'+str(i);
        if( currentUser[col] < 0.4 ):
            currentUser[col]= 0;
        elif( currentUser[col] >= 0.4 and currentUser[col] < 0.7 ):
            currentUser[col] = 1;
        else:
            currentUser[col] = 2;
            
    return currentUser;

def generalGroupBy(data):
    return (data.groupby(['loID','grado']).mean())

def generalGroupByTimeMEAN(data):
    return (data.groupby(['loID','grado'])['tiempo'].mean())

def generalGroupByTimeMEDIAN(data):
    return (data.groupby(['loID','grado'])['tiempo'].median())

def dataComparison(schools,data,row,col,figIndex,titulo):
    for i in range(1,6):
        plt.figure(figIndex)
        plt.subplot(2,3,i)
        for j in range(0,len(schools)):
            plt.plot(data[j].loc[lambda x: x.loID == i-1][row],data[j].loc[lambda x: x.loID == i-1][col], label=schools[j]['escuela'][1])
        
        plt.xlabel = 'Grado'
        if(col=='isCorrect'):
            plt.ylim(0,1)
            plt.ylabel = '% de correctas'
        elif(col=='answerChangedCount'):
            plt.plot([1,2,3,4,5],[1,1,1,1,1], label='Ideal')
            plt.ylabel = 'Promedio de selección'
            plt.ylim(0,2)
        else:
            plt.ylabel = 'Tiempo'
        if(i==5):
            plt.suptitle(titulo)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Objetivo de aprendizaje #"+str(i))
                  

def predictionsAnalysis(testValues, predictions, size):
    LO0 = []
    LO1 = []
    LO2 = []
    LO3 = []
    LO4 = []
    for i in range(0,size):
        LO0.append(predictions[i][0])
        LO1.append(predictions[i][1])
        LO2.append(predictions[i][2])
        LO3.append(predictions[i][3])
        LO4.append(predictions[i][4])
    
    for i in range(1,6):
        plt.figure(5)
        plt.subplot(2,3,i)
        if(i == 1):
            testArray = testValues['LOIN0']
            LO = LO0;
        elif(i == 2):
            testArray = testValues['LOIN1']
            LO = LO1;
        elif(i == 3):
            testArray = testValues['LOIN2']
            LO = LO2;
        elif(i == 4):
            testArray = testValues['LOIN3']
            LO = LO3;
        else:
            testArray = testValues['LOIN4']
            LO = LO4;
            
        plt.scatter(list(range(0,size)),testArray)
        plt.scatter(list(range(0,size)),LO)
        plt.plot(list(range(0,size)),testArray,label='Actual value')
        plt.plot(list(range(0,size)),LO,label='Prediction')
        plt.xlabel=('Observation')
        plt.ylabel=('Value')
        plt.ylim(0,1)
        if(i == 5):
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def getIntensityDF(intensityDF):
    LO0Array = []
    LO1Array = []
    LO2Array = []
    LO3Array = []
    LO4Array = []
    IDArray = []
    for i in range(0, 1135, 5):
        total = intensityDF.loc[i].intensity+intensityDF.loc[i+1].intensity+intensityDF.loc[i+2].intensity+intensityDF.loc[i+3].intensity+intensityDF.loc[i+4].intensity        
        IDArray.append(intensityDF.loc[i].userID)
        LO0Array.append(intensityDF.loc[i].intensity/total)
        LO1Array.append(intensityDF.loc[i+1].intensity/total)
        LO2Array.append(intensityDF.loc[i+2].intensity/total)
        LO3Array.append(intensityDF.loc[i+3].intensity/total)
        LO4Array.append(intensityDF.loc[i+4].intensity/total)
    d = {'userID':IDArray, 'LOI0': LO0Array,'LOI1': LO1Array,'LOI2': LO2Array,'LOI3': LO3Array,'LOI4': LO4Array}
    return pd.DataFrame(data=d)


############################
### Reading the datasets ###
hartfortDF = pd.read_csv('datasets/firstModel/datos_hartfort.csv')
hartfortDF = hartfortDF.drop(labels=['Unnamed: 0'],axis=1)
ensenanzaDF = pd.read_csv('datasets/firstModel/datos_la_ensenanza.csv');
ensenanzaDF = ensenanzaDF.drop(labels=['Unnamed: 0'],axis=1)
#hartfortDF = hartfortDF.loc[lambda x: x.genero == 1]
schoolsDF = []
schoolsDF.append(hartfortDF)
schoolsDF.append(ensenanzaDF)

###########################
####### Analysis ##########
schoolsMeansGB = []
schoolsMediansGB = []
schoolsGeneralGroupBy = []
schoolsMeansGB.append(generalGroupByTimeMEAN(hartfortDF).reset_index())
schoolsMeansGB.append(generalGroupByTimeMEAN(ensenanzaDF).reset_index())
schoolsMediansGB.append(generalGroupByTimeMEDIAN(hartfortDF).reset_index())
schoolsMediansGB.append(generalGroupByTimeMEDIAN(ensenanzaDF).reset_index())
schoolsGeneralGroupBy.append(generalGroupBy(hartfortDF).reset_index())
schoolsGeneralGroupBy.append(generalGroupBy(ensenanzaDF).reset_index())
#Time analysis MEAN
dataComparison(schoolsDF,schoolsMeansGB, 'grado', 'tiempo', 1,'Promedio de tiempos por colegio')
#Time analysis MEDIAN (most accurate)
dataComparison(schoolsDF,schoolsMediansGB, 'grado', 'tiempo', 2, 'Mediana de tiempos por colegio')
#is Correct % analysis
dataComparison(schoolsDF,schoolsGeneralGroupBy, 'grado', 'isCorrect', 3, 'Preguntas correctas por colegio')
#changed Answers analysis
#dataComparison(schoolsDF,schoolsGeneralGroupBy, 'grado', 'answerChangedCount', 4)

#Adding score to each school
hartfortDF['score'] = hartfortDF.apply(lambda x: giveScoreToAnswer(x, generalGroupByTimeMEDIAN(hartfortDF)), axis=1)
ensenanzaDF['score'] = ensenanzaDF.apply(lambda x: giveScoreToAnswer(x, generalGroupByTimeMEDIAN(ensenanzaDF)), axis=1)

#sns.distplot(hartfortDF['tiempo'])
#sns.distplot(ensenanzaDF['tiempo'])

#Merging datasets
ensenanzaDF = ensenanzaDF.drop(labels=['answerChangedCount','problema'],axis=1)
hartfortDF = hartfortDF.drop(labels=['problema'],axis=1)
mergedDF = pd.concat([hartfortDF, ensenanzaDF]).reset_index(drop=True)

#Grouping users
scoresMeans = mergedDF.groupby(['userID','loID'])['score'].mean().reset_index();
scoresMeans['intensity'] = scoresMeans.apply(calculateIntensities,axis=1);
scoresMeans = scoresMeans.round({'intensity':1})

#General user information
mergedDF['escuela'] = mergedDF.apply(convertSchool,axis=1)
generalUserInfo = mergedDF.groupby(['userID']).mean()
generalUserInfo = generalUserInfo.drop(labels=['loID','tiempo','isCorrect','score'],axis=1);

LOIntensitiesPerUser = getIntensityDF(scoresMeans);
LOIntensitiesPerUser = LOIntensitiesPerUser.set_index('userID');
LOIntensitiesPerUser = LOIntensitiesPerUser.round(2);
generalUserInfo[['LOIN0','LOIN1','LOIN2','LOIN3','LOIN4']] = LOIntensitiesPerUser[['LOI0','LOI1','LOI2','LOI3','LOI4']];

#Correlation analysis
correlationData = generalUserInfo.corr(method='pearson')


#Dataset preparation
X = generalUserInfo.drop(labels=['LOIN0','LOIN1','LOIN2','LOIN3','LOIN4'],axis=1);
Y = generalUserInfo[['LOIN0','LOIN1','LOIN2','LOIN3','LOIN4']].copy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
sc = StandardScaler() #Must be scaled to avoid variable domination
#Transforming dataset
#pickle.dumps(X.describe(),'normalizingData')
#X = preprocessing.normalize(X,axis=1)
trainedScaler = sc.fit(X_train);
joblib.dump(sc, 'scaler.joblib');
X_train = trainedScaler.transform(X_train)
X_test = trainedScaler.transform(X_test)


classifier = Sequential();
classifier.add(Dense(input_dim=4, activation='relu', kernel_initializer='normal', units=5));
classifier.add(Dense(kernel_initializer='uniform',units=5))
classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse','mae']);
modelHistory = classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test);

predictionsAnalysis(y_test, y_pred, len(y_pred))

classifier.save('NewPlayerNeuralNet.h5')





#print(modelHistory.history.keys())

#plt.figure(20)
#plt.plot(modelHistory.history['mean_squared_error'])
#plt.plot(modelHistory.history['mean_absolute_error'])
#plt.title('Precisión del modelo')
#plt.ylabel = "Precisión2"
#plt.xlabel = "Epoch"
#plt.legend(['Train', 'Test'], loc='upper left')

#loss = keras.losses.mean_squared_error(y_test,y_pred)

#import tensorflow as tf
#sess = tf.InteractiveSession()
#plt.figure(21)
#plt.plot(modelHistory.history['loss'])
#plt.plot(loss.eval())
#plt.title('Model Loss Values')
#plt.ylabel = 'Loss'
#plt.xlabel = 'Epoch'
#plt.legend(['Train', 'Test'], loc='upper left')

