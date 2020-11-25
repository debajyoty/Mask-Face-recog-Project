import os
import cv2
import face_recognition
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import math
import os
import sys
from threading import Timer
import shutil
import time

img_path = os.getcwd() + "//Recog_Train"

images = []
class_names = []
encode_list = []
encode_list_cl = []
myList = os.listdir(img_path)

#print(myList)

for subdir in os.listdir(img_path):
    path = img_path + '/' + subdir
    path = path + '/'
    for img in os.listdir(path):
        img_pic = path + img
        class_names.append(subdir)
        cur_img = cv2.imread(img_pic)
        cur_img = cv2.cvtColor(cur_img , cv2.COLOR_BGR2RGB)
        images.append(cur_img)
        
   
#plt.imshow()

#for img in images:
    

def detect_and_predict_mask(frame, faceNet, maskNet,threshold):
	# grab the dimensions of the frame and then construct a blob
	# from it
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
            
			# add the face and bounding boxes to their respective
			# lists
			locs.append((startX, startY, endX, endY))
			#print(maskNet.predict(face)[0].tolist())
			preds.append(maskNet.predict(face)[0].tolist())
	return (locs, preds)
# SETTINGS
MASK_MODEL_PATH="C:\masksdetection-master\masksdetection-master\model\mask_model.h5"
FACE_MODEL_PATH="C:\masksdetection-master\masksdetection-master\face_detector"
SOUND_PATH="C:\masksdetection-master\masksdetection-master\sounds\alarm.wav" 
THRESHOLD = 0.5

# Load Sounds
#mixer.init()
#sound = mixer.Sound(SOUND_PATH)
from os.path import dirname, join

protoPath = join(dirname(__file__), "deploy.prototxt")
weightsPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
#prototxtPath = "C:\masksdetection-master\masksdetection-master\face_detector\deploy.prototxt.txt"
#weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(protoPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)
        
#for cl in myList :
 #   cur_img = cv2.imread(f'{img_path} / {cl}' )
 #   images.append([cur_img])
        
def find_encodings(images) :
    #for names in images :
        for img in images : 
            encodings = face_recognition.face_encodings(img)[0]
            encode_list.append(encodings)
       
        return encode_list
    
encodeListKnown = find_encodings(images)
 
cap = cv2.VideoCapture(0)

while True : 
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)
    (locs, preds) = detect_and_predict_mask(imgs, faceNet, maskNet,THRESHOLD)
    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame  = face_recognition.face_encodings(imgs,facesCurFrame)
    
    for encodeFace , faceLoc , pred in zip(encodeCurFrame,facesCurFrame ,preds):
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        matches = face_recognition.compare_faces(encode_list, encodeFace)
        faceDis = face_recognition.face_distance(encode_list, encodeFace)
        matchIndex = np.argmin(faceDis)
        name = class_names[matchIndex]
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
 
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4 , x2*4 , y2*4 , x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),color, cv2.FILLED)
        cv2.putText(img, name, (x1+6 , y2 - 6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2 )
        cv2.putText(img, label, (x1 , y2+10) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,0,255) , 2)
        
        cv2.imshow('Project' , img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
cv2.destroyAllWindows()
vs.stop()
        
        

        
    
    
