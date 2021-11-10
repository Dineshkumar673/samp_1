#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
dir_name='/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Datasets/Absdiff_base_dir/segmented_dataset/Arya_chudidhar_augs_0000/'
dir_val=os.listdir(dir_name)
#print(dir_val)
classes_dir = ['after-30', 'afternoon-36', 'age-36',  'any-35',
                   'baby-34', 'brother-35', 'birthday-34','eat-34', 'sorry-31', 'sister-28']    
dest='/mnt/fs/Splitted_data/Arya_chudidhar_0000'
for i in dir_val:
    os.mkdir(dest+'/'+i+'/')
    for j in classes_dir:
        os.mkdir(dest+'/'+i+'/'+j)
for i in dir_val:
    for j in classes_dir:
        src=os.listdir(dir_name +'/'+i+'/'+j+'/')
        for k in src:
            success=shutil.copy(dir_name +'/'+i+'/'+j+'/'+k,dest+'/'+i+'/'+j)

