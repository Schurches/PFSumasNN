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
from keras.models import Sequential
from keras.layers import Dense
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt, log10


#rms = sqrt(mean_squared_error(Y_test,y_pred1))

def isAnswerCorrect(df):
    df['respuesta'] = df['respuesta'].apply(str)
    isCorrect = (df['respuesta'] == df['respuestaJ']);
    return 1*isCorrect;

def dropDataFromTables(df):
    df.drop(labels=['id','escuela','problema','tipo','respuesta','respuestaJ'])

def scoreR(tiempo):
    if tiempo >= 0 and tiempo <= 60:
        return -1*(tiempo / 3) + 20
    else:
        return 0

def generateProblemID():
    problemID = []
    cycles = 0;
    for index in range(0,2875):
        if(index%115==0):
            cycles = cycles+1;
        problemID.append(cycles);
    return problemID;

def obtainProblemInfo(problemArray):
    isN1GreaterThan = []
    isN1Equal = []
    isN1LessThan = []
    digitsOfN1 = [];
    digitsOfN2 = [];
    digitsOfAnswer = [];
    for index in range(0,2875):
        currentProblem = problemArray.loc[index];
        N1 = currentProblem.split('+');
        N2 = N1[1].split('=?');
        N1 = int(N1[0]);
        N2 = int(N2[0]);
        isN1GreaterThan.append(1*(N1 > N2));
        isN1Equal.append(1*(N1==N2));
        isN1LessThan.append(1*(N1<N2))
        if(N1==0):
            digitsOfN1.append(1);
        else:
            digitsOfN1.append(int(log10(N1))+1);
        if(N2==0):
            digitsOfN2.append(1);
        else:
            digitsOfN2.append(int(log10(N2))+1);
        if((N1+N2)==0):
            digitsOfAnswer.append(1);
        else:
            digitsOfAnswer.append(int(log10(N1+N2))+1);
    d = {'isN1GreaterThan': isN1GreaterThan,
            'isN1Equal': isN1Equal,
            'isN1LessThan': isN1LessThan,
            'N1Length': digitsOfN1,
            'N2Length': digitsOfN2,
            'AnsLength': digitsOfAnswer}
    return pd.DataFrame(data=d);

#Reading the data
hartfortDF = pd.read_csv('datasets/datos_hartford_marzo.csv');
isCorrectColumn = isAnswerCorrect(hartfortDF);
hartfortDF['isCorrect'] = isCorrectColumn;
hartfortDF = hartfortDF.drop(labels=['id','escuela','respuesta','respuestaJ','tipo'],axis=1)

#Cleaning data

hartfortDF = hartfortDF.loc[lambda x: x.edad > 4]
uncompleteData = hartfortDF.groupby(['userID']).count().loc[lambda x: x.edad < 25]
hartfortDF = hartfortDF.loc[lambda x: x.userID != '-LaQlc1fC7y-HDX9BHFG']
gradesAnalysis = hartfortDF.groupby(['grado','genero','edad','loID']).mean()
hartfortDF = hartfortDF.reset_index()
problemFeatures = obtainProblemInfo(hartfortDF['problema'])
#Plotting data
plot1 = sns.FacetGrid(hartfortDF,row='grado',col='edad').map(sns.distplot, "tiempo")

#Normalizing data
normalizedTime = hartfortDF['tiempo'].copy();
normalizedTime = [normalizedTime]
normalizedTime = preprocessing.normalize(normalizedTime,axis=1);
normalizedTime = np.transpose(normalizedTime);
hartfortDF['tiempo'] = normalizedTime;

#normalizedData['grado'] = normalizedData['grado'].apply(str);
#dummyGrades = pd.get_dummies(normalizedData['grado']);
#dummyGrades = dummyGrades.drop('5',1);
#dummyGrades.columns= ['grado1','grado2','grado3','grado4'];
#normalizedData = normalizedData.drop('grado',1);
#normalizedData[['grado1','grado2','grado3','grado4']] = dummyGrades[['grado1','grado2','grado3','grado4']];

#Preparing data
hartfortDF = hartfortDF.sort_values(['loID','problema'])
nNDataser = hartfortDF.drop(labels=['userID','isCorrect'],axis=1)
problemID = generateProblemID();
hartfortDF['problemaID'] = problemID;
hartfortDF[['isN1GreaterThan','isN1Equal','isN1LessThan','N1Length','N2Length','AnsLength']] = problemFeatures[['isN1GreaterThan','isN1Equal','isN1LessThan','N1Length','N2Length','AnsLength']];

X = hartfortDF.drop(labels=['userID','problema','isCorrect'],axis=1)
Y = hartfortDF['isCorrect'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

sc = StandardScaler() #Must be scaled to avoid variable domination

#Transforming dataset
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Adding input layer and a hidden layer
#nLayers = [3, 5, 10, 15]
#for pos in range(0,4):
#    #Artificial Neural Network
#    classifier = Sequential() #Defined ANN as sequence of layers
#    classifier.add(Dense(input_dim=5, activation='relu', kernel_initializer='uniform', units=nLayers[pos])) #relu is rectifier function
#    classifier.add(Dense(kernel_initializer='uniform', units=1))
#    #compiling ANN
#    classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse','mae']) #Adam is the stochastic gradient descent
#    #Fitting the ANN
#    #nBatch = [10,20,30,40,50]
#    #nEpoch = [25, 50, 75, 100]
#    print('NEURAL NETWORK WITH HIDDEN LAYER OF SIZE: ', str(nLayers[pos]));
#    classifier.fit(X_train, Y_train, batch_size=10, epochs=50, verbose=2)

#Building the NN
classifier = Sequential();
classifier.add(Dense(input_dim=13,activation='relu',kernel_initializer='uniform', units=20)); #Hidden layer with 5 neurons
classifier.add(Dense(kernel_initializer='uniform', activation='relu',units=1)); #Output layer
#Compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']);

classifier.fit(X_train, y_train, batch_size=20, epochs=50)

y_pred = classifier.predict(X_test);
y_pred_bool = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test,y_pred_bool)

















#Initiating the ANN dataset
#neuralDataset = correctAnswersDF[['loID','correct','grado','tiempo']];
##Adding score
#score = neuralDataset['tiempo'];
#score = score.apply(scoreR);
#score.columns = ['score'];
##neuralDataset['score'] = score;
##Saving normal category (for further test)
#normalLOs = correctAnswersDF['loID'];
##Generating dummy variables
#dummyGrades = pd.get_dummies(correctAnswersDF['grado']);
#dummyGrades.columns = ['grado1','grado2','grado3','grado4','grado5'];
#dummyGrades = dummyGrades.drop('grado5',1);
##dummyLOs = pd.get_dummies(correctAnswersDF['loID']);
##dummyLOs.columns = ['LO0','LO1','LO2','LO3','LO4']
##dummyLOs = dummyLOs.drop('LO4',1);
##Normalizing values
#neuralTimeNormalized = preprocessing.normalize([neuralDataset['tiempo']],axis=1);
#neuralScoreNormalized = preprocessing.normalize([score],axis=1);
#neuralTimeNormalized = np.transpose(neuralTimeNormalized);
#neuralScoreNormalized = np.transpose(neuralScoreNormalized);
#neuralIDNormalized = preprocessing.normalize([neuralDataset['loID']],axis=1);
#neuralIDNormalized = np.transpose(neuralIDNormalized);
#neuralGradoNormalized = preprocessing.normalize([neuralDataset['grado']],axis=1);
#neuralGradoNormalized = np.transpose(neuralGradoNormalized);
#
#datasetWithDummies = neuralDataset[['loID']].copy();
#datasetWithDummies['loID'] = neuralIDNormalized;
#datasetWithDummies['grado'] = correctAnswersDF[['grado']].copy();
#datasetWithDummies['grado'] = neuralGradoNormalized;
##datasetWithDummies['correct'] = neuralDataset[['correct']].copy();
##datasetWithDummies[['grado1','grado2','grado3','grado4']] = dummyGrades.copy();
#datasetWithDummies['tiempo'] = correctAnswersDF['tiempo'].copy();
#
##Dataset to use for training
#
##Not-nornamlized dataset
##dataset1 = datasetWithDummies.copy();
##outputV1 = score;
##Normalized dataset
#datasetN1 = timesPerformance.drop(labels=['score'],axis=1)#datasetWithDummies.copy();
##datasetN1['tiempo'] = normalizedT;
#
##datasetN1['tiempo'] = neuralTimeNormalized;
#normalizedT = preprocessing.normalize([timesPerformance['score']],axis=1);
#normalizedT = np.transpose(normalizedT);
#
#outputV2 = timesPerformance[['score']].copy();
#outputV2['score'] = normalizedT;
##outputV2 = neuralScoreNormalized;



