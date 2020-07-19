# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco
# import the necessary packages
from imutils.video import *
import numpy as np
import argparse
import imutils
import time
import cv2
import os
#import face_recognition
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import sys

angry_cnt = int(0)
surprised_cnt = int(0)
sad_cnt = int(0)
scared_cnt = int(0)
happy_cnt = int(0)
neutral_cnt = int(0)
disgust_cnt = int(0)

pic_number = int(1)
fans = int(0)
fans_total = int(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help="path to input video")
ap.add_argument("-o", "--output", required=False,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# load model for Expression detection
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#cv2.namedWindow('your_face')
preds = ""

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

folder = "U:\\MAIN_PRX\\jaffedbase\\"

with open("testfile.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for x_line in content:
	print("This is IMG no: " + str(pic_number))
	pic_number = pic_number + 1
	filename = x_line
	print(x_line)
	img = cv2.imread(os.path.join(folder,filename))
	cropped = img
	try:
		frame1 = imutils.resize(cropped,width=300)
	except Exception as e:
		continue
	gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
		
	canvas = np.zeros((250, 300, 3), dtype="uint8")
	frameClone = frame1.copy()
	if len(faces) > 0:
		faces = sorted(faces, reverse=True,
		key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		(fX, fY, fW, fH) = faces
					# Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
			# the ROI for classification via the CNN
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (64, 64))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
		
		
		preds = emotion_classifier.predict(roi)[0]
		emotion_probability = np.max(preds)
		label = EMOTIONS[preds.argmax()]
		
		print("------------>>  " + label + " : " + x_line[3:5])
		
		ans = x_line[3:5]
		
		EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
		if(label=="angry" and ans=="AN"):
			fans = fans + 1
			angry_cnt = angry_cnt + 1
		elif(label=="scared" and ans=="FE"):
			fans = fans + 1
			scared_cnt = scared_cnt + 1
		elif(label=="disgust" and ans=="DI"):
			fans = fans + 1
			disgust_cnt = disgust_cnt + 1
		elif(label=="happy" and ans=="HA"):
			fans = fans + 1
			happy_cnt = happy_cnt + 1
		elif(label=="sad" and ans=="SA"):
			fans = fans + 1
			sad_cnt = sad_cnt + 1
		elif(label=="surprised" and ans=="SU"):
			fans = fans + 1
			surprised_cnt = surprised_cnt + 1
		elif(label=="neutral" and ans=="NE"):
			fans = fans + 1
			neutral_cnt = neutral_cnt + 1

		fans_total = fans_total + 1
			
print("Happy detected : " + str(happy_cnt))
print("Sad detected : " + str(sad_cnt))
print("Scared detected : " + str(scared_cnt))
print("disgust detected : " + str(disgust_cnt))
print("surprised detected : " + str(surprised_cnt))
print("neutral detected : " + str(neutral_cnt))
print("angry detected : " + str(angry_cnt))
print("Correct Detected : " + str(fans))
print("Total : " + str(fans_total))


