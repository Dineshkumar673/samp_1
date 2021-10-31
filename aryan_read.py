
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
import pandas as pd 
from csv import writer


from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


# In[ ]:


import os
path = '/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Datasets/Absdiff_base_dir/segmented_dataset/Aryan_augs_0000/'
#The immediate file path

directory_contents = os.listdir(path)
print(directory_contents)


# In[ ]:





# In[ ]:


for i in range (0,len(directory_contents)):
    #os.path.join(path,directory_contents[i])
    class_name = os.listdir(path+'/'+directory_contents[i]+'/')
    
    
    for j in range (0,len(class_name)):
        top_list=[]
        #df = pd.DataFrame()
        img_name = os.listdir(path+'/'+ directory_contents[i]+'/'+class_name[j]+'/')
        #print(img_name)
        for k in range (0,len(img_name)):
            input_frame = cv2.imread(path+'/'+directory_contents[i]+'/'+class_name[j]+'/'+img_name[k])
            
            #print(input_frame)
            with mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5) as pose:
                pose_result = pose.process(image=input_frame)
                pose_landmarks = pose_result.pose_landmarks

            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = cv2.resize(input_frame,(256,256))
            #input_frame = cv2.flip(input_frame,1)

            
            if pose_landmarks is not None:
                left_shoulder_x =0
                left_shoulder_y =0
                right_shoulder_x=0
                right_shoulder_y=0
                left_elbow_x =0
                left_elbow_y =0
                right_elbow_x =0
                right_elbow_y =0
                left_wrist_x =0
                left_wrist_y =0
                right_wrist_x =0
                right_wrist_y =0
                left_pinky_x =0
                left_pinky_y =0
                right_pinky_x =0
                right_pinky_y =0
                left_thumb_x =0
                left_thumb_y =0
                right_thumb_x =0
                right_thumb_y =0
            #else:                 
                
                left_shoulder_x = pose_landmarks.landmark[11].x
                left_shoulder_y = pose_landmarks.landmark[11].y
                right_shoulder_x = pose_landmarks.landmark[12].x
                right_shoulder_y = pose_landmarks.landmark[12].y

                left_elbow_x = pose_landmarks.landmark[13].x
                left_elbow_y = pose_landmarks.landmark[13].y
                right_elbow_x = pose_landmarks.landmark[14].x
                right_elbow_y = pose_landmarks.landmark[14].y

                left_wrist_x = pose_landmarks.landmark[15].x
                left_wrist_y = pose_landmarks.landmark[15].y
                right_wrist_x = pose_landmarks.landmark[16].x
                right_wrist_y = pose_landmarks.landmark[16].y

                left_pinky_x = pose_landmarks.landmark[17].x
                left_pinky_y = pose_landmarks.landmark[17].y
                right_pinky_x = pose_landmarks.landmark[18].x
                right_pinky_y = pose_landmarks.landmark[18].y

                left_index_x = pose_landmarks.landmark[19].x
                left_index_y = pose_landmarks.landmark[19].y
                right_index_x = pose_landmarks.landmark[20].x
                right_index_y = pose_landmarks.landmark[20].y

                left_thumb_x = pose_landmarks.landmark[21].x
                left_thumb_y = pose_landmarks.landmark[21].y
                right_thumb_x = pose_landmarks.landmark[22].x
                right_thumb_y = pose_landmarks.landmark[22].y
            #left_hip_x = pose_landmarks.landmark[23].x
            #left_hip_y = pose_landmarks.landmark[23].y
            #print(right_thumb_x)
            #print(right_thumb_y)
            top_list.append([directory_contents[i], class_name[j], left_shoulder_x,left_shoulder_y,right_shoulder_x,right_shoulder_y,left_elbow_x,left_elbow_y,right_elbow_x,right_elbow_y,left_wrist_x,left_wrist_y,right_wrist_x,right_wrist_y,left_pinky_x,left_pinky_y,right_pinky_x,right_pinky_y,left_index_x,left_index_y,right_index_x,right_index_y,left_thumb_x,left_thumb_y,right_thumb_x,right_thumb_y]) 
        with open(r'/mnt/fs/Splitted_data/samp_1/aryan_final.csv', 'a',newline='') as f:
            top_list = np.array(top_list)
            top_list=top_list.ravel()
            writer_obj = writer(f, delimiter = ',')
            writer_obj.writerow(top_list)
            f.close()

