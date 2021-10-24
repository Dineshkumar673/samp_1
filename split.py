#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import shutil



def train_test_split():
    print("########### Train Test Val Script started ###########")
    
    root_dir = '/mnt/fs/Splitted_data/train/'
    root_dir1 = '/mnt/fs/Splitted_data/test/'
    root_dir2 = '/mnt/fs/Splitted_data/val/'
    classes_dir = ['second(time)-35','sell-33','she-30',
                   'shoe-31','shop-36','show-28','sign-27','sister-28','sit-24','sleep-36','small-21','smile-22','some-25','son-32',
                   'song-33','soon-25','sorry-31','stand-21','start-20','stay-23','stop-31','student-30','study-26','sun-22','table-26',
                   'take-18','talk-31','tea-35','teeth-33','tell-24','thank-28','that-29','then-30','there-28','they-33','thing-35','think-34',
                   'this-35','time-30','tired-31','to-26','today-35','tomorrow-34','traffic-56','train-35','tv-29','uncle-36','under-36',
                   'understand-35','unwell-30','up-34','use-35','wait-35','water-28','we-35','week-27','welcome-35','what-34','when-34',
                   'where-35','which-36','who-36','whose-36','why-33','wife-36','with-35','work-33','wrong-35','year-29','yes-35','yesterday-37','you-24']
   
    #    classes_dir.append(name)

    processed_dir = '/mnt/fs/Data_ML/Small_synthetic_dataset_features'

    val_ratio = 0.20
    test_ratio = 0.20

    for cls in classes_dir:
        # Creating partitions of the data after shuffeling
        print("$$$$$$$ Class Name " + cls + " $$$$$$$")
        src = processed_dir +"/" + cls  # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                                   int(len(allFileNames) * (1 - val_ratio)),
                                                                   ])

        train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
        val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

        print('Total images: '+ str(len(allFileNames)))
        print('Training: '+ str(len(train_FileNames)))
        print('Validation: '+  str(len(val_FileNames)))
        print('Testing: '+ str(len(test_FileNames)))

         # Creating Train / Val / Test folders (One time use)
        os.makedirs(root_dir + cls)
        os.makedirs(root_dir2 + cls)
        os.makedirs(root_dir1 + cls)

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir + cls)

        for name in val_FileNames:
            shutil.copy(name, root_dir2 + cls)

        for name in test_FileNames:
            shutil.copy(name, root_dir1 + cls)

    print("########### Train Test Val Script Ended ###########")

train_test_split()

