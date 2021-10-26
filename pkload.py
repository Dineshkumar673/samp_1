#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Script started")
import pickle 
pickled_model = pickle.load(open('/mnt/fs/Splitted_data/samp_1/knn_algo.pkl', 'rb'))
print("Model loaded")

