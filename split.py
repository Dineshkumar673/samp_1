#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import shutil



def train_test_split():
    print("########### Train Test Val Script started ###########")
    
    root_dir = '/home/ubuntu/mnt/fs/Splitted_data/train/'
    root_dir1 = '/home/ubuntu/mnt/fs/Splitted_data/test/'
    root_dir2 = '/home/ubuntu/mnt/fs/Splitted_data/val/'
    classes_dir = ['address-31','after-30', 'afternoon-36', 'age-36', 'all-23', 'any-35',
                   'baby-34', 'bad-26', 'before-34', 'big-34', 'birthday-34','book-30','boy-35','bring-23',
                   'brother-35','bus_stop-54','busy-34','but-30','bye-27','call-35','close-34','coffee-30',
                   'cold-26','come-28','complete-34','correct-21','count-35','dark-36','date-27','daughter-36',
                   'day-36','deaf-32','do-23','doctor-28','door-36','down-28','dress-34','drink-29','early-33','easy-25',
                   'eat-34','email-34','end-34','enough-33','evening-28','ever-36','family-35','far-28','father-26',
                   'few-21','find-36','fire-36','food-30','forget-34','free-18','friend-35','from-28','fruit-36','full-35',
                   'get-27','girl-36','give-36','go-35','good-28','happy-35','have-28','he-32','hello-23','help-32','here-20',
                   'his-38','hit-25','hot-31','hour-36','house-21','how-22','hungry-28','hurry-35','if-30','in-34','injury-25',
                   'internet-29','it-35','job-28','keep-26','know-25','language-48','last-24','late-26','leave-22','leg-31','light-25',
                   'like-27','live-23','long-32','look-33','love-33','make-23','many-35','market-26','me-24','medicine-35','meet-21',
                   'minute-25','mobile-31','money-36','month-25','morning-30','mother-30','move-35','must-20','name-32','near-28',
                   'need-26','next-36','night-35','no-33','not-34','now-35','number-32','office-35','ok-29','on-31','open-32','other-28',
                   'out-22','over-30','person-29','phone-36','play-35','please-26','pull-26','push-21','question-33','railwaystation-36',
                   'read-29','ready-28','remember-27','road-36','run-34','sad-28','school-31',"'second(time)-35'",'sell-33','she-30',
                   'shoe-31','shop-36','show-28','sign-27','sister-28','sit-24','sleep-36','small-21','smile-22','some-25','son-32',
                   'song-33','soon-25','sorry-31','stand-21','start-20','stay-23','stop-31','student-30','study-26','sun-22','table-26',
                   'take-18','talk-31','tea-35','teeth-33','tell-24','thank-28','that-29','then-30','there-28','they-33','thing-35','think-34',
                   'this-35','time-30','tired-31','to-26','today-35','tomorrow-34','traffic-56','train-35','tv-29','uncle-36','under-36',
                   'understand-35','unwell-30','up-34','use-35','wait-35','water-28','we-35','week-27','welcome-35','what-34','when-34',
                   'where-35','which-36','who-36','whose-36','why-33','wife-36','with-35','work-33','wrong-35','year-29','yes-35','yesterday-37','you-24']
   
    #    classes_dir.append(name)

    processed_dir = '/home/ubuntu/mnt/fs/Data_ML/Small_synthetic_dataset_features'

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

