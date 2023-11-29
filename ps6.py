import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.linear_model import LogisticRegression
import tabulate
import math

#loading in matlab file data
data3 = scipy.io.loadmat("input\hw4_data3.mat")
# print(data3.keys())

X = np.array(data3["X_train"])
y = np.array(data3["y_train"])
xtest = np.array(data3["X_test"])
y_test = np.array(data3["y_test"])
# print(X.shape)

#spliting X_train into three subsets: one for each class
class1Indices = []
class2Indices = []
class3Indices = []
#get indices for each class label
for i in range(0, len(y)):
    if y[i] == 1:
        class1Indices.append(i)
    elif y[i] == 2:
        class2Indices.append(i)
    else:
        class3Indices.append(i)
#divide X_train into subclasses using indices obtained
X_train_1 = []
X_train_2 = []
X_train_3 = []
for j in range(0, len(X)):
    if j in class1Indices:
        X_train_1.append(X[j])
    elif j in class2Indices:
        X_train_2.append(X[j])
    else:
        X_train_3.append(X[j])
#convert to numpy arrays
X_train_1 = np.array(X_train_1)
X_train_2 = np.array(X_train_2)
X_train_3 = np.array(X_train_3)
#display size of classes
# print("Size of X_train_1: ", X_train_1.shape)
# print("Size of X_train_2: ", X_train_2.shape)
# print("Size of X_train_3: ", X_train_3.shape)



########### Naive-Bayes Classifier ###########
# find the mean of each feature for each class
class1Mean = np.mean(X_train_1, axis=0) #mean of each feature/col in class 1
class2Mean = np.mean(X_train_2, axis=0)
class3Mean = np.mean(X_train_3, axis=0)
classIMean = []
classIMean.append(class1Mean)
classIMean.append(class2Mean)
classIMean.append(class3Mean)

# find the standard deviation for each feauture for each class
class1StdDev = np.std(X_train_1, axis=0)
class2StdDev = np.std(X_train_2, axis=0)
class3StdDev = np.std(X_train_3, axis=0)
classIStdDev = []
classIStdDev.append(class1StdDev)
classIStdDev.append(class2StdDev)
classIStdDev.append(class3StdDev)

# print(classIStdDev[0][1])
# print("|                   | Mean             | Standard Deviation |")
# print("| Class 1 Feature 1 |",class1Mean[0], "|",class1StdDev[0], "|")
# print("| Class 1 Feature 2 |",class1Mean[1], "|",class1StdDev[1], "|")
# print("| Class 1 Feature 3 |",class1Mean[2], "|",class1StdDev[2], "|")
# print("| Class 1 Feature 4 |",class1Mean[3], "|",class1StdDev[3], "|")
# print("| Class 2 Feature 1 |",class2Mean[0], "|",class2StdDev[0], "|")
# print("| Class 2 Feature 2 |",class2Mean[1], "|",class2StdDev[1], "|")
# print("| Class 2 Feature 3 |",class2Mean[2], "|",class2StdDev[2], "|")
# print("| Class 2 Feature 4 |",class2Mean[3], "|",class2StdDev[3], "|")
# print("| Class 3 Feature 1 |",class3Mean[0], "|",class3StdDev[0], "|")
# print("| Class 3 Feature 2 |",class3Mean[1], "|",class3StdDev[1], "|")
# print("| Class 3 Feature 3 |",class3Mean[2], "|",class3StdDev[2], "|")
# print("| Class 3 Feature 4 |",class3Mean[3], "|",class3StdDev[3], "|")

#for each feature j, calculate p(xj|wi)
p_wi = (1/3)
#posterior probability matrix
postProb = np.zeros((xtest.shape[0], 3))
#log summation matrix
classProb = np.zeros((xtest.shape[0], 3))
#for testing sample k
for k in range(0, xtest.shape[0]): #25 test samples
    #for class i
    for i in range(0, 3): # 3 classes
        #for feature j, calculate p(xj|wi)
        for j in range(0, X_train_1.shape[1]): #4 features
            #normal distribution equation for exponent
            expo = (-((xtest[k][j] - classIMean[i][j])**2)/(2*classIStdDev[i][j]**2))
            # p(xj | wi)
            probTemp = (1/(math.sqrt(2*math.pi)*classIStdDev[i][j]))*math.exp(expo)
            # p(x|wi), independent values
            classProb[k][i] += np.log(probTemp)
        # ln(p(wi | x)), posterior probability
        postProb[k][i] = classProb[k][i] + np.log(p_wi)
# add 1 cause indexed at zero
estClassifier = np.argmax(postProb, axis=1) + 1 #col with max (class with max)
estClassifierFixed = [[i] for i in estClassifier] #add [] around entries to match ytest

# Getting accuracy of estimation
accuracyCount = 0
for i in range(0, len(y_test)):
    # print(estClassifierFixed[i], y_test[i])
    if estClassifierFixed[i] == y_test[i]:
        accuracyCount += 1
accuracy = (accuracyCount/len(y_test))*100
print("Accuracy: ", accuracy, "%")








####################################################################

# estimate covariacne matrix for each class, rows=samples
sigma_1 = np.cov(X_train_1, rowvar=False)
sigma_2 = np.cov(X_train_2, rowvar=False)
sigma_3 = np.cov(X_train_3, rowvar=False)

# print("X_train_1 covariance:")
# print(sigma_1.shape)
# print(sigma_1)
# print("X_train_2 covariance:")
# print(sigma_2.shape)
# print(sigma_2)
# print("X_train_3 covariance:")
# print(sigma_3.shape)
# print(sigma_3)

#compute mean vectors 
# print("Mean vector of class 1: ", class1Mean)
# print("Size of mean vector 1: ", class1Mean.shape)
# print("Mean vector of class 2: ", class2Mean)
# print("Size of mean vector 2: ", class2Mean.shape)
# print("Mean vector of class 3: ", class3Mean)
# print("Size of mean vector 3: ", class3Mean.shape)

# compute g1(x), g2(x), g3(x)
# class1Mean = class1Mean.reshape(-1, 1)
# g_x_part1 = (-1/2) * np.dot(np.dot((xtest[i, :] - class1Mean).T, np.linalg.inv(sigma_1)), (xtest[i, :] - class1Mean))
# g_x_part2 = g_x_part1 + np.log(p_wi)

# class2Mean = class2Mean.reshape(-1, 1)
# g_x2_part1 = (-1/2) * np.dot(np.dot((xtest[i, :] - class2Mean).T, np.linalg.inv(sigma_2)), (xtest[i, :] - class2Mean))
# g_x2_part2 = g_x2_part1 + np.log(p_wi)

# class3Mean = class3Mean.reshape(-1, 1)
# g_x3_part1 = (-1/2) * np.dot(np.dot((xtest[i, :] - class3Mean).T, np.linalg.inv(sigma_3)), (xtest[i, :] - class3Mean))
# g_x3_part2 = g_x3_part1 + np.log(p_wi)



discriminantValues = []
classCovariance = [sigma_1, sigma_2, sigma_3]
for i in range(xtest.shape[0]):
    discriminants = np.zeros(3)
    for j in range(3):
        # covariance matrix inverse
        inverseCov = np.linalg.inv(classCovariance[j])
        # get x - ui
        xMinusAvg = xtest[i] - classIMean[j]
        xMinusAvg = np.array(xMinusAvg)
        tempTerm = np.dot(inverseCov, xMinusAvg.T)
        # (x-iu).T * inverse cov matrix *(x-ui)
        firstTerm = np.dot(tempTerm, xMinusAvg)
        # covariance matrix determinant
        covDeterminant = np.linalg.det(classCovariance[j]) 
        # log of determinant of covariance matrix
        logDet = np.log(covDeterminant)
        #log of prior prob
        logPrior = np.log(p_wi) 
        # Calculate the discriminant function for class j
        discriminants[j] = (-0.5)*firstTerm + logPrior + (-0.5)*np.log(2*math.pi) - (0.5)*logDet
    # Find the class with the highest discriminant value
    classEst = np.argmax(discriminants) + 1
    discriminantValues.append(classEst)
discriminantValuesListed = [[i] for i in discriminantValues]

accuracyCount = 0
for i in range(0, len(y_test)):
    if discriminantValues[i] == y_test[i]:
        accuracyCount += 1
accuracy = accuracyCount/len(y_test) *100
print("Accuracy:", accuracy, "%")


