#Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#to load matlab mat files
from scipy.io import loadmat
import pathlib
import sklearn.datasets
from sklearn.model_selection import train_test_split

import copy
from PIL import Image

import glob

import pathlib

from utils import *


data_dir = pathlib.Path('./dataset')


image_count = len(list(data_dir.glob('*/*.*')))
print("Total no of images =", image_count)

images = []
labels = []
classes = set()

image_width = 30
image_height = 30

for filename in data_dir.glob('*/*.*'):
    class_name = str(filename).split("/")[1]
    classes.add(class_name)
    im=copy.deepcopy(Image.open(filename).resize((image_width,image_height)).convert('LA'))
    #print(np.array(im).shape)
    images.append(np.array(im)[...,:1].reshape((image_width*image_height*1,)))
    #print(images[0].shape)
    labels.append(class_name)
    im.close()

classes = sorted(list(classes))
labels = [i for x in labels for i in range(len(classes)) if x == classes[i]]





dataX = np.array(images).reshape((len(images), images[0].shape[0]))

#dataX[dataX <  128] = 0
#dataX[dataX >= 128] = 1
dataX = 1 - dataX / 255

# plt.imshow(dataX[0].reshape(image_width, image_height), cmap="gray")

dataY = np.array(labels).reshape((len(labels),1))





train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

#print(x_train, x_val, x_test)


#Optimization hyper-parameters 
alpha_lst = [0.1]
num_iters_lst = [10, 50, 100, 400, 1000]
Lambda_lst = [0.1, 0.5 , 1, 2 ,3 , 5, 10]

train_pred = [[],[],[],[],[]]
val_pred = [[],[],[],[],[]]
test_pred = [[],[],[],[],[]]

for alpha in alpha_lst:
    for num_iters in num_iters_lst:
        for Lambda in Lambda_lst:
            print(f"alpha {alpha}  num_iters {num_iters}  Lambda {Lambda}")

            #Inicialize vector theta =0
            initial_theta = np.zeros((x_train.shape[1]+1,1))

            all_theta, all_J = oneVsAll(x_train, y_train, initial_theta, alpha, num_iters, Lambda, len(classes))

            pred = predictOneVsAll(all_theta, x_val)
            m = len(y_val)
            #Check that pred.shape  = (5000,) => rank 1 array. You need to reshape it !!!
            pred= pred.reshape((len(x_val),1))
            val_acc = sum(np.equal(pred,y_val))[0]/m*100
            print("Validation Set Accuracy:", val_acc,"%")
            val_pred[0].append(val_acc)
            val_pred[1].append(Lambda)
            val_pred[2].append(alpha)
            val_pred[3].append(num_iters)
            val_pred[4].append(all_J)   

            pred = predictOneVsAll(all_theta, x_train)
            m = len(y_train)
            #Check that pred.shape  = (5000,) => rank 1 array. You need to reshape it !!!
            pred= pred.reshape((len(x_train),1))
            train_acc = sum(np.equal(pred,y_train))[0]/m*100
            print("Training Set Accuracy:", train_acc,"%")
            train_pred[0].append(train_acc)
            train_pred[1].append(Lambda)
            train_pred[2].append(alpha)
            train_pred[3].append(num_iters)
            train_pred[4].append(all_J) 

            pred = predictOneVsAll(all_theta, x_test)
            m = len(y_test)
            #Check that pred.shape  = (5000,) => rank 1 array. You need to reshape it !!!
            pred= pred.reshape((len(x_test),1))
            test_acc = sum(np.equal(pred,y_test))[0]/m*100
            print("Test Set Accuracy:", test_acc,"%")
            test_pred[0].append(test_acc)
            test_pred[1].append(Lambda)
            test_pred[2].append(alpha)
            test_pred[3].append(num_iters)
            test_pred[4].append(all_J)  
            print()


fp = open("results.txt", 'w')

fp.write(str(val_pred)+'\n')
fp.write(str(train_pred)+'\n')
fp.write(str(test_pred)+'\n')

fp.close()
