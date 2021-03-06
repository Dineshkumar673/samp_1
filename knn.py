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
        print("In Progress")
print("$$$Done with Appending train values$$$")        


# In[ ]:


X_train =np.array(x)
Y_train =np.array(y)
print("$$$Train values Converted to array$$$")



print("$$$Loading sklearn packages$$$")
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


print("$$$Fitting Knn Model$$$")
#Knn algorithm
# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
print("$$$KNN ALGORITHM COMPLETED$$$")



print("$$$Dumping Model$$$")
import pickle
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("$$$Script Done$$$")

