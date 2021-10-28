

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
        print('in progress')
print("$$$Done with Appending train values$$$")        


# In[ ]:


X_train =np.array(x)
Y_train =np.array(y)
print("$$$Train values Converted to array$$$")



print("$$$Loading sklearn packages$$$")
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


print("$$$Fitting Random Forest Model$$$")
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train,Y_train)
print("$$$RANDOM FOREST ALGORITHM COMPLETED$$$")



print("$$$Dumping Model$$$")
import pickle
with open('/mnt/fs/Splitted_data/randomforest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("$$$Script Done$$$")
