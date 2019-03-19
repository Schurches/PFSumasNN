# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:29:01 2019

@author: Steven
"""

#Data analysis of our dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#Reading the data
hartfortDF = pd.read_csv('datos_hartfort_modificado.csv');
#Ignoring ID
data = hartfortDF.iloc[:,1:];
data = data.drop(labels=["escuela","tipo"], axis=1);
isCorrectColumn = data['respuesta'] == data['respuestaJ'];
correctAnswersDF = data.drop(labels=['respuesta', 'respuestaJ'], axis=1);
correctAnswersDF['correct'] = 1*isCorrectColumn;


#Analyzing gender
malesPerformance = correctAnswersDF.loc[correctAnswersDF['genero']==0,['loID','correct']]
femalesPerformance = correctAnswersDF.loc[correctAnswersDF['genero']==1,['loID','correct']]
#plot1 = sns.FacetGrid(malesPerformance, col='loID');
#plot1.map(plt.hist, "correct");
#plot2 = sns.FacetGrid(femalesPerformance, col='loID');
#plot2.map(plt.hist, "correct");
timesPerformance = correctAnswersDF[['edad','genero','grado','loID','tiempo']];
timesPerformance = timesPerformance.sort_values(by=['genero','edad','grado','loID']);
#plot3= sns.FacetGrid(timesPerformance, col='genero');
#plot3.map(sns.distplot,"tiempo")
#plot4= sns.FacetGrid(timesPerformance, col='edad');
#plot4.map(sns.distplot,"tiempo")
#plot5= sns.FacetGrid(timesPerformance, col='grado');
#plot5.map(sns.distplot,"tiempo")
#plot6= sns.FacetGrid(timesPerformance, col='loID');
#plot6.map(sns.distplot,"tiempo")
#Scatter plot the data
#currentAge1 = timesPerformance.loc[timesPerformance['grado']==1,['edad','tiempo']]
#currentAge2 = timesPerformance.loc[timesPerformance['grado']==2,['edad','tiempo']]
#currentAge3 = timesPerformance.loc[timesPerformance['grado']==3,['edad','tiempo']]
#currentAge4 = timesPerformance.loc[timesPerformance['grado']==4,['edad','tiempo']]
#currentAge5 = timesPerformance.loc[timesPerformance['grado']==5,['edad','tiempo']]
sns.pairplot(timesPerformance)
#sns.distplot(timesPerformance['tiempo'])
#
#plt.subplot(3, 2, 1)
#plt.scatter(x=currentAge1['edad'],y=currentAge1['tiempo'])
#plt.subplot(3, 2, 2)
#plt.scatter(x=currentAge2['edad'],y=currentAge2['tiempo'])
#plt.subplot(3, 2, 3)
#plt.scatter(x=currentAge3['edad'],y=currentAge3['tiempo'])
#plt.subplot(3, 2, 4)
#plt.scatter(x=currentAge4['edad'],y=currentAge4['tiempo'])
#plt.subplot(3, 2, 5)
#plt.scatter(x=currentAge5['edad'],y=currentAge5['tiempo'])


#score = timesPerformance['tiempo']
#plt.scatter(x=timesPerformance['tiempo'],y=timesPerformance['grado'])
#plt.scatter(x=timesPerformance['tiempo'],y=timesPerformance['grado'])

#Linear regression to calculate score

def scoreR(tiempo):
    if tiempo >= 0 and tiempo <= 60:
        return -1*(tiempo / 3) + 20
    else:
        return 0

score = timesPerformance['tiempo'];
score = score.apply(scoreR);
score.columns = ['score'];
timesPerformance['score'] = score;
normalizedTime = [timesPerformance['tiempo']];
normalizedAge = [timesPerformance['edad']];
normalizedData = timesPerformance;
normalizedTime = preprocessing.normalize(normalizedTime,axis=1);
normalizedAge = preprocessing.normalize(normalizedAge,axis=1);
normalizedTime = np.transpose(normalizedTime);
normalizedAge = np.transpose(normalizedAge);
normalizedData['tiempo'] = normalizedTime;
normalizedData['edad'] = normalizedAge;


normalizedData['grado'] = normalizedData['grado'].apply(str);
dummyGrades = pd.get_dummies(normalizedData['grado']);
dummyGrades = dummyGrades.drop('5',1);
dummyGrades.columns= ['grado1','grado2','grado3','grado4'];
normalizedData = normalizedData.drop('grado',1);
normalizedData[['grado1','grado2','grado3','grado4']] = dummyGrades[['grado1','grado2','grado3','grado4']];

X1 = normalizedData[['edad','grado1','grado2','grado3','grado4','tiempo']];
Y = normalizedData['score'];


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y, test_size=0.2, random_state=101)

linearModel1 = LinearRegression()

linearModel1.fit(X_train1,y_train1);

predictions1 = linearModel1.predict(X_test1)

plt.scatter(y_test1,predictions1)

print(linearModel1.coef_)
print(linearModel1.intercept_)

linearModel1.score(X1,Y)



#Initiating the ANN dataset
neuralDataset = correctAnswersDF[['loID','correct','grado','tiempo']];
#Adding score
score = neuralDataset['tiempo'];
score = score.apply(scoreR);
score.columns = ['score'];
#neuralDataset['score'] = score;
#Saving normal category (for further test)
normalLOs = correctAnswersDF['loID'];
#Generating dummy variables
dummyGrades = pd.get_dummies(correctAnswersDF['grado']);
dummyGrades.columns = ['grado1','grado2','grado3','grado4','grado5'];
dummyGrades = dummyGrades.drop('grado5',1);
#dummyLOs = pd.get_dummies(correctAnswersDF['loID']);
#dummyLOs.columns = ['LO0','LO1','LO2','LO3','LO4']
#dummyLOs = dummyLOs.drop('LO4',1);
#Normalizing values
neuralTimeNormalized = preprocessing.normalize([neuralDataset['tiempo']],axis=1);
neuralScoreNormalized = preprocessing.normalize([score],axis=1);
neuralTimeNormalized = np.transpose(neuralTimeNormalized);
neuralScoreNormalized = np.transpose(neuralScoreNormalized);
neuralIDNormalized = preprocessing.normalize([neuralDataset['loID']],axis=1);
neuralIDNormalized = np.transpose(neuralIDNormalized);
neuralGradoNormalized = preprocessing.normalize([neuralDataset['grado']],axis=1);
neuralGradoNormalized = np.transpose(neuralGradoNormalized);

datasetWithDummies = neuralDataset[['loID']].copy();
datasetWithDummies['loID'] = neuralIDNormalized;
datasetWithDummies['grado'] = correctAnswersDF[['grado']].copy();
datasetWithDummies['grado'] = neuralGradoNormalized;
#datasetWithDummies['correct'] = neuralDataset[['correct']].copy();
#datasetWithDummies[['grado1','grado2','grado3','grado4']] = dummyGrades.copy();
datasetWithDummies['tiempo'] = correctAnswersDF['tiempo'].copy();

#Dataset to use for training

#Not-nornamlized dataset
#dataset1 = datasetWithDummies.copy();
#outputV1 = score;
#Normalized dataset
datasetN1 = timesPerformance.drop(labels=['score'],axis=1)#datasetWithDummies.copy();
#datasetN1['tiempo'] = normalizedT;

#datasetN1['tiempo'] = neuralTimeNormalized;
normalizedT = preprocessing.normalize([timesPerformance['score']],axis=1);
normalizedT = np.transpose(normalizedT);

outputV2 = timesPerformance[['score']].copy();
outputV2['score'] = normalizedT;
#outputV2 = neuralScoreNormalized;


X_train, X_test, Y_train, Y_test = train_test_split(datasetN1, outputV2, test_size=0.2, random_state=101)
sc = StandardScaler() #Must be scaled to avoid variable domination

#Transforming not-normalized dataset
#X_train1 = sc.fit_transform(Xt)
#X_test1 = sc.transform(Xtst)
#Transforming normalized dataset
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Adding input layer and a hidden layer
nLayers = [3, 5, 10, 15]
for pos in range(0,4):
    #Artificial Neural Network
    classifier = Sequential() #Defined ANN as sequence of layers
    classifier.add(Dense(input_dim=5, activation='relu', kernel_initializer='uniform', units=nLayers[pos])) #relu is rectifier function
    classifier.add(Dense(kernel_initializer='uniform', units=1))
    #compiling ANN
    classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse','mae']) #Adam is the stochastic gradient descent
    #Fitting the ANN
    #nBatch = [10,20,30,40,50]
    #nEpoch = [25, 50, 75, 100]
    print('NEURAL NETWORK WITH HIDDEN LAYER OF SIZE: ', str(nLayers[pos]));
    classifier.fit(X_train, Y_train, batch_size=10, epochs=50, verbose=2)

y_pred1 = classifier.predict(X_test);


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test,y_pred1))