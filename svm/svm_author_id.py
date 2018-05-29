#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# end=len(features_train)/100
# features_train = features_train[:end]
# labels_train = labels_train[:end]

from sklearn.svm import SVC
#clf=SVC(kernel="linear")
clf=SVC(kernel="rbf",C=10000)

print "starting fit..."
t0=time()
clf.fit(features_train,labels_train)
print "fit completed in ", round(time()-t0,3), "s"

print "starting predict..."
t1=time()
pred=clf.predict(features_test)
print "predict completed in ", round(time()-t1,3), "s"
print "pred 10:", pred[10]
print "pred 26:", pred[26]
print "pred 50:", pred[50]

count_chris=0
for elem in pred:
    if(elem==1):
        count_chris=count_chris+1
print count_chris
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,labels_test))

#########################################################


