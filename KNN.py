
# coding: utf-8

# In[ ]:


import numpy as np
from collections import Counter

trainingdata = open("C:/Users/Abhishek/Desktop/SML/KNN/mnist_train.csv").read()
trainingdata = trainingdata.split("\n")[1:-1]
trainingdata = [i.split(",") for i in trainingdata]

X_train = np.array([[int(i[j]) for j in range(1,len(i))] for i in trainingdata])
y_train = np.array([int(i[0]) for i in trainingdata])

testingdata = open("C:/Users/Abhishek/Desktop/SML/KNN/mnist_test.csv").read()
testingdata = testingdata.split("\n")[1:-1]
testingdata = [i.split(",") for i in testingdata]
X_test = np.array([[int(i[j]) for j in range(1,len(i))] for i in testingdata])

print ( X_train, y_train, X_test)


class kNN():
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def ypred(self, X, k=1):
        L2Distance = self.Euclidian(X)
        testingno = L2Distance.shape[0]
        predicted_y = np.zeros(testingno)
        for i in range(testingno):
            Nearestk = []
            Traininglabel = self.y_train[np.argsort(L2Distance[i,:])].flatten()
            Nearestk = Traininglabel[:k]
            count = Counter(Nearestk)
            predicted_y[i] = count.most_common(1)[0][0]
        return(predicted_y)

    def Euclidian(self, X):
        testingno = X.shape[0]
        trainingno = self.X_train.shape[0]
        dotproduct = np.dot(X, self.X_train.T)
        squareoftrain = np.square(self.X_train).sum(axis = 1)
        squareoftest = np.square(X).sum(axis = 1)
        L2Distance = np.sqrt(-2 *dotproduct+squareoftrain+np.matrix(squareoftest).T)

        return(L2Distance)

#for each K calculate the prediction and flush it onto the file
# k = 1,3, 5, 10, 30, 50, 70, 80, 90, 100
k = 1
object_for_kNN = kNN()
object_for_kNN.train(X_train, y_train)

y = []


batch_size = 2000
for i in range(int(len(X_test)/(2*batch_size))):
    y_p = object_for_kNN.ypred(X_test[i * batch_size:(i+1) * batch_size], k)
    y = y + list(y_p)


for i in range(int(len(X_test)/(2*batch_size)), int(len(X_test)/batch_size)):
    predts = object_for_kNN.ypred(X_test[i * batch_size:(i+1) * batch_size], k)
    y = y + list(y_p)
print("Done with Testing")



op = open("C:/Users/Abhishek/Desktop/SML/KNN/predictions.csv", "w")
op.write("Id,Label\n")
for x in range(len(y)):
    op.write(str(x+1)+","+str(int(y[x]))+"\n")
op.close()





