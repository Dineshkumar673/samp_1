#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
        print("In Progress")
print("$$$Done with Appending test values$$$")   


# In[5]:


X_test =np.array(x)
Y_test =np.array(y)
print("$$$Test values Converted to array$$$")


# In[ ]:





# In[7]:


print("$$$Script started$$$")
import pickle 
pickled_model = pickle.load(open('/mnt/fs/Splitted_data/samp_1/knn_model.pkl', 'rb'))
print("$$$Model loaded$$$")
pred = pickled_model.predict(X_test)
print("$$$Measuring Accuracy of knn$$$")
#Accuracy knn
from sklearn.metrics import confusion_matrix, accuracy_score
matrix= confusion_matrix(Y_test, pred)
score = accuracy_score(Y_test, pred)
print("$$$Printing Confusion matrix$$$")
#print(matrix)
print("$$$Printing Accuracy$$$")
print('Accuracy Score :',score)
print("$$$Done with Accuracy of Knn$$$")
print("$$$Script Done$$$")
import pandas as pd
import csv
pd.DataFrame(matrix).to_csv("/mnt/fs/Splitted_data/samp_1/con_mat_knn.csv")


# In[ ]:





# In[ ]:




