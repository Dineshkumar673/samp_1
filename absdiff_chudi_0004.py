#!/usr/bin/env python
# coding: utf-8

# In[ ]:


####### DECLARE VARIABLES ###########################################################################################
import numpy as np
import cv2
import random
import os
import shutil
import sys
import time
segments_dir = '/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Datasets/Absdiff_base_dir/segmented_dataset/Arya_chudidhar_augs_0004/'
absdiff_dir = '/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Absdiff/Arya_chudidhar_0004/'
percentage = 0
prev_inf_time = 0
img_nm = 0
pixel_val = 10
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
n_seg = 13
segments_dir_lst = os.listdir(segments_dir)
segments_dir_lst.sort()
print(segments_dir_lst)


# In[ ]:


Seg_dir_lst = os.listdir(segments_dir)
Seg_dir_lst.sort()

if not os.path.exists(absdiff_dir):
    
    os.mkdir(absdiff_dir)

Var_nmber = 0

for Aug_dir in Seg_dir_lst:
    Aug_dir_lst = os.listdir(segments_dir+'/'+Aug_dir)
    Aug_dir_lst.sort()
    os.mkdir(absdiff_dir+'/'+Aug_dir)
    for Sign_dir in Aug_dir_lst:
        Sign_dir_lst = os.listdir(segments_dir+'/'+Aug_dir+'/'+Sign_dir)
        Sign_dir_lst.sort()
        Var_dir = str(Var_nmber).zfill(5)
        if not os.path.exists(absdiff_dir+'/'+Sign_dir):
            os.mkdir(absdiff_dir+'/'+Aug_dir+'/'+Sign_dir)
        
        i = 1
        for Image in Sign_dir_lst:
            current_frame = cv2.imread(os.path.join(segments_dir,Aug_dir,Sign_dir,Sign_dir_lst[i]),0)
            previous_frame = cv2.imread(os.path.join(segments_dir,Aug_dir,Sign_dir,Sign_dir_lst[i-1]),0)
            current_frame = cv2.resize(current_frame,(256,256))
            previous_frame = cv2.resize(previous_frame,(256,256))
           
            name = str(i)
            name = name.zfill(4)
            begin = time.time()
            absdiff = cv2.absdiff(previous_frame,current_frame)
            absdiff = cv2.GaussianBlur(absdiff,(3,3),cv2.BORDER_DEFAULT)
            absdiff = cv2.morphologyEx(absdiff, cv2.MORPH_OPEN, kernel)
            absdiff = cv2.morphologyEx(absdiff, cv2.MORPH_CLOSE, kernel)
            absdiff[absdiff < pixel_val] = 0
            absdiff = cv2.equalizeHist(absdiff)
            end = time.time()
            inf_time = end - begin
            avg_inf_time = inf_time + prev_inf_time
            prev_inf_time = avg_inf_time
            img_nm += 1
            cv2.imwrite(absdiff_dir+'/'+Aug_dir+'/'+Sign_dir+'/'+"absdiff"+name+".jpg",absdiff)
            i = i + 1
            if i == len(Sign_dir_lst):
                break 
        Var_nmber += 1
avg_inf_time = avg_inf_time / img_nm
print(avg_inf_time)
print("Done with absdiff creation of Arya_chudi_0004")

    ####### DONE ###############################################################################################

