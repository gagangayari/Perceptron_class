# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:04:12 2020

@author: Gagan
"""


from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#%%

data=load_breast_cancer()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
X=df
y=data.target
#%% Train Test split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y, train_size=0.8,stratify=y)


#%%

class perceptron:
    def __init__(self):
        self.w=None
        self.b=None
        
    def pred_single(self,x):
        """
        Predicts the class of a single data point

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION:1 or 0 depending on the predicted class.
            Its a binary classification.

        """
        return 1 if(np.dot(self.w,x)>=self.b) else 0
    
    def predict(self,X):
        """
        Predicts the classes of a given set of data.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        Y=[]
        for x in X:
            prediction=self.pred_single(x)
            Y.append(prediction)
        return np.array(Y)
            
    
        
        
    def fit(self,X_train,Y_train,epoch=1):
        """
        Train or fit a model.Sets the weights and bias.

        Parameters
        ----------
        X_train : TYPE :2D array
            DESCRIPTION: Feature matrix/values
        Y_train : TYPE :1D array
            DESCRIPTION: Output labels corresponding to the features
        epoch : TYPE, int
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.w=np.ones(X_train.shape[1])
        self.b=0
        max_acc=0
        accuracy=[]
        weights=[]
        for _ in range(epoch):
            for x,y in zip(X_train,Y_train):
                y_pred=self.pred_single(x)
                if(y==0 and y_pred==1):
                    self.w=self.w-x
                    self.b=self.b-1
                    
                elif(y==1 and y_pred==0):
                    self.w=self.w+x
                    self.b=self.b+1
            
            acc1=accuracy_score(self.predict(X_train),Y_train)
            accuracy.append(acc1)
            
            ##Checks the maximum accuracy achived.Also saves the parameters for max_accuracy
            if(acc1>max_acc):
                max_acc=acc1
                w_argmax=self.w
                b_argmax=self.b
            weights.append(self.w)
        
        self.w=w_argmax
        self.b=b_argmax
        
        ##Plot the change in weights
        for w in weights:
            ax=plt.subplot(111)
            ax.plot(w)
            ax.set_ylim([-6000,15000])
            plt.pause(0.5)
        
        ##See the changer in accuracy
        # plt.plot(accuracy)
        # plt.show()
            
#%% Creating an instance

p=perceptron()

p.fit(X_train.values,Y_train,100)

predictions=p.predict(X_test.values)

accuracy=accuracy_score(Y_test, predictions)

print(accuracy)


    