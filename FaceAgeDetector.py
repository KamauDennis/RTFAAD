import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cam_device_index',type=int)
parser.add_argument('--img_path',type=str)
args = parser.parse_args()

cam_device_index = args.cam_device_index
img_path = args.img_path

#loading model
model_path = './age_detector_model/0001' 
age_detector_model = tf.saved_model.load(model_path)

def preprocess(img):
  img_res = cv2.resize(img,(200,200))
  img_tensor = tf.convert_to_tensor(img_res,dtype=tf.float32)
  img_exp = tf.expand_dims(img_tensor,0)
  return img_exp

  #detecting faces and detecting age
def detect(x):
    class_names = np.array(['0-1','2-5','6-12','13-19','20-29','30-39','40-59','60-120'])
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    #if the camera device index is given as an argument
    if type(x) is int:
      camera = cv2.VideoCapture(x)
      while (True):
          ret,frame = camera.read()
          gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#face detection happens in grayscale format
          faces = face_cascade.detectMultiScale(gray,1.3,5)
          for (x,y,w,h) in faces:
              img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
              roi = frame[y:y+h,x:x+w]  
              prep_roi = preprocess(roi)
              #adding age detector model
              pred = age_detector_model(prep_roi)
              label = class_names[tf.argmax(pred,1)]
              label = label[0]
              cv2.putText(img,label,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)                              
          cv2.imshow("camera",frame)
          if cv2.waitKey(84) & 0xff == ord("q"):
              break
      camera.release() 
      cv2.destroyAllWindows() 

    else:       
      img = cv2.imread(x)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
      faces = face_cascade.detectMultiScale(gray,1.3,3)
      for (x,y,w,h) in faces:
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          roi = img[y:y+h,x:x+w]
          prep_roi = preprocess(roi)
          pred = age_detector_model(prep_roi)
          label = class_names[tf.argmax(pred,1)]
          label = label[0]
          cv2.putText(img,label,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      cv2.imwrite('PredImage.jpg',img)    
      cv2.imshow('Image',img)
      cv2.waitKey()
      cv2.destroyAllWindows()
    
if img_path:
  detect(img_path)
else:
  detect(cam_device_index)
