#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("$$$Script Started$$$")
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
print("$$$Done with Appending train values$$$")        


# In[ ]:


X_train =np.array(x)
Y_train =np.array(y)
print("$$$Train values Converted to array$$$")


# In[ ]:


print("$$$Starting with test values$$$")
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
print("$$$Done with Appending test values$$$")         


# In[ ]:


X_test =np.array(x)
Y_test =np.array(y)
print("$$$Test values Converted to array$$$")


# In[ ]:


print("$$$Loading sklearn packages$$$")
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


print("$$$Fitting Knn Model$$$")
#Knn algorithm
# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
print("$$$KNN ALGORITHM COMPLETED$$$")


# In[ ]:


print("$$$Measuring Accuracy of knn$$$")
#Accuracy knn
from sklearn.metrics import confusion_matrix, accuracy_score
matrix= confusion_matrix(Y_test, pred)
score = accuracy_score(Y_test, pred)
print("$$$Printing Confusion matrix$$$")
print(matrix)
print("$$$Printing Accuracy$$$")
print('Accuracy Score :',score)
print("Done with Accuracy of Knn")


# In[ ]:


print("$$$Dumping Model$$$")
import pickle
with open('knn_algo.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("$$$Script Done$$$")

