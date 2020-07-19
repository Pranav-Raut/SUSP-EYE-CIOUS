#  /* 
#	* 		DIRCTORY :: 	(tensorflow_env) C:\Users\HP PC\Documents\Tensorflow\workspace\training_demo>
#	*		
#	*		RUNNING FILE COMMAND :: 	python MAIN_VIDEO_INPUT_FILE_RUNNING_LATEST_MODIFIED_14_5_19.py --input path_to_input/input_file_name --output path_to_output/output_file_name 
#	*/


from imutils.video import *
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import sys
import tensorflow as tf
import glob

flg = int(0)
flg_weapon = int(0)
flg_shoot = int(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

rcnn_coco = "rcnn-coco"

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# load the COCO class labels our  model was trained on
labelsPath = os.path.sep.join([rcnn_coco, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# load model for Expression detection
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# derive the paths to the weights and model configuration
weightsPath = os.path.sep.join([rcnn_coco, "cnnv3.weights"])
configPath = os.path.sep.join([rcnn_coco, "cnnv3.cfg"])

# load our object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from it
print("[INFO] loading MODEL from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'trained-inference-graphs'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'annotations','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

print("[INFO] Loading video from stream...")
vs = cv2.VideoCapture(args["input"])
(W, H) = (None, None)
writer = None

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

preds = ""

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

frno_psr = int(0)
frame_no = int(0)

exp_lis = []
# loop over frames from the video file stream
while True:
	print("Frame number : "+str(frno_psr))
	frno_psr = frno_psr + 1
	frame_no = frame_no + 1
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			
			if(classID!=0):
				continue

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		label = ""
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			
			
			
			
			if(classIDs[i]==1):
				cropped = frame[y:y+h, x:x+w]
				img=cropped
				
				#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				try:
					frame1 = imutils.resize(cropped,width=300)
				except Exception as e:
					continue
				# frame1 = cv2.resize(cropped, (300, 300), interpolation = cv2.INTER_AREA)
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
					
					# exp_lis.append(label)
					# lbl = label
					# cnt = int(0)
					# if(frame_no > 10):
					# 	exp_lis.pop(0)
					# 	mp = {}
					# 	for i in lis:
					# 		try:
					# 			exp_lis[label] = exp_lis[label] + 1
					# 		except:
					# 			exp_lis[label] = 1
						
					# 	for ky in exp_lis:
					# 		if(exp_lis[ky] > cnt):
					# 			cnt = exp_lis[ky]
					# 			lbl = ky
					# 		if(exp_lis[ky] == cnt && (ky == "angry" || ky == "scared" || ky == "sad")):
					# 			cnt = exp_lis[ky]
					# 			lbl = ky
					
					if(label=="scared" or label=="sad" or label=="angry"):
						flg = 1
				cv2.putText(frame, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					
				
				

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
	image_expanded = np.expand_dims(frame, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes_ssd, scores_ssd, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})
		
	vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.squeeze(boxes_ssd),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores_ssd),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=5,
		min_score_thresh=0.50)

	for i in range(scores_ssd.shape[1]):
		if scores_ssd[0,i]>0.5:
			if(str(category_index.get(classes[0,i])['name'])=="gun" or str(category_index.get(classes[0,i])['name'])=="knife"):
				flg_weapon = 1
		if scores_ssd[0,i]>0.85:
			if(str(category_index.get(classes[0,i])['name'])=="gun"):
				flg_shoot = 1
			
	# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
	
	to_print = ""
	if(flg==1 and flg_weapon==1 and flg_shoot == 1):
		to_print = "Highly SUSP\"EYE\"CIOUS"
	elif(flg_shoot == 1):
		to_print = "On Gun Point"
	elif(flg_weapon==1):
		to_print = "Alert : WEAPON DETECTED"
	else:
		to_print = "No threat"
	flg = 0
	flg_shoot = 0
	flg_weapon = 0
	color = []
	if(to_print=="No threat"):
		color = [int(c) for c in COLORS[1]]
	else:
		color = [int(c) for c in COLORS[3]]
	cv2.putText(frame, to_print, (W-600, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
	
	if writer is None:
	# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
		(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
	writer.write(frame)
	

# release the file pointers
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.release()