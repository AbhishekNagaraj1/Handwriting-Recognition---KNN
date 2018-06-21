
# coding: utf-8

# In[1]:




# In[76]:

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
df1 = pd.read_csv("C:/Users/Abhishek/Desktop/SML/KNN/predictions10.csv", sep=',')
#df_data = open('C:/Users/Abhishek/Desktop/SML/KNN/predictions1.csv').read()
#df = df_data[0:].values
#cols = ['Label']
#print(df1.ix[:,1])
#print(df2)
df2 = pd.read_csv('C:/Users/Abhishek/Desktop/SML/KNN/mnist_test.csv')
#df.head
print("-----------")
#df2.ix[:,0]
y_pred = df1.ix[:,1]
df3 = df2.ix[:,0]
y_true = df3[:8000]

accuracy_score(y_true, y_pred)
#0.5
#accuracy_score(y_true, y_pred, normalize=False)


# In[ ]:




# In[ ]:



