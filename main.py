
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cv2

# img_array = cv2.imread('WIN_20201222_10_54_49_Pro.jpg')
# resized = cv2.resize(gray , (100,100))
# normalized = resized / 255
# cv2.imshow('live',gray)


model = keras.models.load_model('mask_detection')


face_clsfr=cv2.CascadeClassifier('haarcascade.xml')

source=cv2.VideoCapture(0)

labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}



while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,3))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        # print(label)

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],1)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key == ord('q')):
        break
        
cv2.destroyAllWindows()
source.release()