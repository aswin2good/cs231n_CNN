'''k Nearest Neighbor Classifier (training)'''
import numpy as np

class NearestNeighbor: 
  def __init__(self):
    pass

  def train(self,X,y):
   #X is N x D where each row is an example
   #Y is 1D of size N
    self.Xtr = X
    self.ytr = y
  def predict(self,X):
   #X is N x D where each row is an example we wish to predict the label for-
    num_test = X.shape[0]
    #let output match the input
    Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

    #now, for each test image, we need to find the closest train image
    #and predict labesl of the nearest image
    #so, loop over all test rows
    for i in xrange(num_test):
        #find the nerest training image to the ith test image
        #using the L1 (Manhattan) distance (sum of absolute value differnce between test and training)
        distances = np.ssum(np.abs(self.Xtr - X[i,: ]), axis =1)
        min_index = np.argmin(distances) #get the index with smallest distance
        Ypred[i] = self.ytr[min_index] #predict the label of the nearest example
    return Ypred
