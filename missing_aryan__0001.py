#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
import pandas as pd 
import shutil


from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import hands as mp_hands


# In[ ]:


path = '/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Datasets/Absdiff_base_dir/segmented_dataset/Aryan_augs_0001/'
#The immediate file path

directory_contents = os.listdir(path)
print(directory_contents)
dest='/mnt/fs/Synthetic_dataset_creation/Absdiff_dataset/Datasets/Absdiff_base_dir/Aryan_missingpose/Aryan_0001/'


# In[ ]:


#original code
count=0
top_list=[]
for i in range (0,len(directory_contents)):
    #os.path.join(path,directory_contents[i])
    class_name = os.listdir(path+'/'+directory_contents[i]+'/')
    os.mkdir(dest+'/'+directory_contents[i]+'/')
    
    for j in range (0,len(class_name)):
          
        #df = pd.DataFrame()
        img_name = os.listdir(path+'/'+ directory_contents[i]+'/'+class_name[j]+'/')
        os.mkdir(dest+'/'+ directory_contents[i]+'/'+class_name[j]+'/')
        
        #print(img_name)
        for k in range (0,len(img_name)):
            
            img=(path+'/'+directory_contents[i]+'/'+class_name[j]+'/'+img_name[k])
            input_frame = cv2.imread(img)
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = cv2.resize(input_frame,(256,256))
            #print(input_frame)
            with mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.6) as pose:
                pose_result = pose.process(image=input_frame)
                pose_landmarks = pose_result.pose_landmarks
            with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.2) as hands:
                hand_result = hands.process(image=input_frame)
                hand_landmarks = hand_result.multi_hand_landmarks
                multi_handedness = hand_result.multi_handedness
            #input_frame = cv2.flip(input_frame,1)

            
            if pose_landmarks is None:
        
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
                success=shutil.copy(path+'/'+directory_contents[i]+'/'+class_name[j]+'/'+img_name[k],dest+'/'+ directory_contents[i]+'/'+class_name[j])
                count +=1
                print(count)
                #top_list.append(['dinnu' for value_0 in range(0,26) ])
               
            else:                 
                
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
            #print(hand_result.multi_handedness)
            if hand_landmarks is None:
                
                #for hand_landmark in hand_landmarks:
                wrist_x=0
                wrist_y=0
                thumb_cmc_x=0
                thumb_cmc_y=0
                thumb_mcp_x=0
                thumb_mcp_y=0
                thumb_ip_x=0
                thumb_ip_y=0
                thumb_tip_x=0
                thumb_tip_y=0
                index_finger_mcp_x=0
                index_finger_mcp_y=0
                index_finger_pip_x=0
                index_finger_pip_y=0
                index_finger_dip_x=0
                index_finger_dip_y=0
                index_finger_tip_x=0
                index_finger_tip_y=0
                middle_finger_mcp_x=0
                middle_finger_mcp_y=0
                middle_finger_pip_x=0
                middle_finger_pip_y=0
                middle_finger_dip_x=0
                middle_finger_dip_y=0
                middle_finger_tip_x=0
                middle_finger_tip_y=0
                ring_finger_mcp_x=0
                ring_finger_mcp_y=0
                ring_finger_pip_x=0
                ring_finger_pip_y=0
                ring_finger_dip_x=0
                ring_finger_dip_y=0
                ring_finger_tip_x=0
                ring_finger_tip_y=0
                pinky_finger_mcp_x=0
                pinky_finger_mcp_y=0
                pinky_finger_pip_x=0
                pinky_finger_pip_y=0
                pinky_finger_dip_x=0
                pinky_finger_dip_y=0
                pinky_finger_tip_x=0
                pinky_finger_tip_y=0
                len_hands=(0 if (multi_handedness) is None else len(multi_handedness))
                    #top_list.append(['dinnu' for value_0 in range(27,110) ])
                        #hand_path.append([img,hand_result.multi_handedness])
                
                
                        #hand_path.append([img,hand_result.multi_handedness])
                        
                        #print(hand_result.multi_handedness)   
                        #writer_obj = writer(f, delimiter = ',')
                        #writer_obj.writerow([directory_contents[i], class_name[j]+'/'+img_name[k],hand_result.multi_handedness])
                        #f.close()
                        #writer_obj = writer(f, delimiter = ',')
                        #writer_obj.writerow(hand_path)
                        #f.close()
            else:
                for hand_landmark in hand_landmarks:        
                    wrist_x = hand_landmark.landmark[0].x
                    wrist_y = hand_landmark.landmark[0].y
                    thumb_cmc_x = hand_landmark.landmark[1].x
                    thumb_cmc_y = hand_landmark.landmark[1].y
                    thumb_mcp_x = hand_landmark.landmark[2].x
                    thumb_mcp_y = hand_landmark.landmark[2].y
                    thumb_ip_x = hand_landmark.landmark[3].x
                    thumb_ip_y = hand_landmark.landmark[3].y
                    thumb_tip_x = hand_landmark.landmark[4].x
                    thumb_tip_y = hand_landmark.landmark[4].y

                    index_finger_mcp_x = hand_landmark.landmark[5].x
                    index_finger_mcp_y = hand_landmark.landmark[5].y
                    index_finger_pip_x = hand_landmark.landmark[6].x
                    index_finger_pip_y = hand_landmark.landmark[6].y
                    index_finger_dip_x = hand_landmark.landmark[7].x
                    index_finger_dip_y = hand_landmark.landmark[7].y
                    index_finger_tip_x = hand_landmark.landmark[8].x
                    index_finger_tip_y = hand_landmark.landmark[8].y

                    middle_finger_mcp_x = hand_landmark.landmark[9].x
                    middle_finger_mcp_y = hand_landmark.landmark[9].y
                    middle_finger_pip_x = hand_landmark.landmark[10].x
                    middle_finger_pip_y = hand_landmark.landmark[10].y
                    middle_finger_dip_x = hand_landmark.landmark[11].x
                    middle_finger_dip_y = hand_landmark.landmark[11].y
                    middle_finger_tip_x = hand_landmark.landmark[12].x
                    middle_finger_tip_y = hand_landmark.landmark[12].y

                    ring_finger_mcp_x = hand_landmark.landmark[13].x
                    ring_finger_mcp_y = hand_landmark.landmark[13].y
                    ring_finger_pip_x = hand_landmark.landmark[14].x
                    ring_finger_pip_y = hand_landmark.landmark[14].y
                    ring_finger_dip_x = hand_landmark.landmark[15].x
                    ring_finger_dip_y = hand_landmark.landmark[15].y
                    ring_finger_tip_x = hand_landmark.landmark[16].x
                    ring_finger_tip_y = hand_landmark.landmark[16].y

                    pinky_mcp_x = hand_landmark.landmark[17].x
                    pinky_mcp_y = hand_landmark.landmark[17].y
                    pinky_pip_x = hand_landmark.landmark[18].x
                    pinky_pip_y = hand_landmark.landmark[18].y
                    pinky_dip_x = hand_landmark.landmark[19].x
                    pinky_dip_y = hand_landmark.landmark[19].y
                    pinky_tip_x = hand_landmark.landmark[20].x
                    pinky_tip_y = hand_landmark.landmark[20].y
                    len_hands=(0 if (multi_handedness) is None else len(multi_handedness))
            top_list.append([img,directory_contents[i],class_name[j],img_name[k],len_hands])
            
print(top_list[0])
output_df=pd.DataFrame(top_list,columns=['path','Aug','class_name','Misssing_image','No_of_hands']) 

print(output_df.head())
output_df.to_csv('/mnt/fs/Splitted_data/samp_1/aryan_hand_0001.csv', index=False)                
print("$$$Done With Aryan_0001")

