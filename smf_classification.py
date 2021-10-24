#!/usr/bin/env python
# coding: utf-8

# In[2]:


#first performing ravel and then appending it
import numpy as np
import os
x=[]
y=[]
for i in os.listdir('/mnt/fs/Splitted_data/train/'):    
    file_names=os.listdir(os.path.join('/mnt/fs/Splitted_data/train/',i))
    for k in file_names:
        values=np.load(os.path.join('/mnt/fs/Splitted_data/train/',i,k))
        flattened_val = values.ravel()
        #values.reshape(1,-1)
        #print(len(flattened_val))
        x.append(flattened_val)
        y.append(i)


# In[3]:


X_train =np.array(x)
Y_train =np.array(y)


# In[5]:


#first performing ravel and then appending it
import numpy as np
import os
x=[]
y=[]
for i in os.listdir('/mnt/fs/Splitted_data/test/'):    
    file_names=os.listdir(os.path.join('/mnt/fs/Splitted_data/test/',i))
    for k in file_names:
        values=np.load(os.path.join('/mnt/fs/Splitted_data/test/',i,k))
        flattened_val = values.ravel()
        #values.reshape(1,-1)
        #print(len(flattened_val))
        x.append(flattened_val)
        y.append(i)


# In[7]:


X_test =np.array(x)
Y_test =np.array(y)


# In[8]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[13]:


#Knn algorithm
# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
print("KNN ALGORITHM COMPLETED")


# In[14]:


#Accuracy knn
from sklearn.metrics import confusion_matrix, accuracy_score
matrix= confusion_matrix(Y_test, pred)
score = accuracy_score(Y_test, pred)
print(matrix)
print('Accuracy Score :',score)
print("Done with Accuracy of Knn")


# In[15]:


#Randomforest algorithm
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
pred1 = rf.predict(X_test)
print("$$$$ RandomForest Completed $$$$$$$")


# In[16]:


#Accuracy randomforest
matrix= confusion_matrix(Y_test, pred1)
score = accuracy_score(Y_test, pred1)
print(matrix)
print('Accuracy Score :',score)
print("$$$Done with Accuracy of RandomForest$$$")


# In[18]:


from xgboost import XGBClassifier
#xgboost algorithm
xg = XGBClassifier()
xg.fit(X_train,Y_train)
pred2 = xg.predict(X_test)
print("$$$XGBoost Completed$$$$")


# In[19]:


#Accuracy xgboost
matrix= confusion_matrix(Y_test, pred2)
score = accuracy_score(Y_test, pred2)
print(matrix)
print('Accuracy Score :',score)
print("Done with Accuracy of XGboost")


# In[20]:


import pickle
with open('knn_algo.pkl', 'wb') as f:
    pickle.dump(knn, f)


# In[21]:


import pickle
with open('randomforest_algo.pkl', 'wb') as f:
    pickle.dump(rf, f)


# In[25]:


import pickle
with open('xgboost_algo.pkl', 'wb') as f:
    pickle.dump(xg, f)
print("PROGRAM COMPLETED")


# In[23]:




