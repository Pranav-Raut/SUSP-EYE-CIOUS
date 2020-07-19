# Import packages
# To Run python test_model.py "C:\Users\HP PC\Documents\Tensorflow\workspace\training_demo\11.jpg" --textual --with_image
import tensorflow as tf
import sys
import os
import cv2
import numpy as np
import glob
import argparse

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'trained-inference-graphs_[[fstr_rcnn]]'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'annotations','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = args.image_path

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

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
for imno in range(1,96):
	if(imno==18):
		continue
	imno_pic = str("C:\\Users\\HP PC\\Documents\\Tensorflow\\workspace\\training_demo\\test\\g ("+str(imno)+").jpg")
	image = cv2.imread(imno_pic)
	image_expanded = np.expand_dims(image, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})

	flg_psr = int(0)
	for i in range(scores.shape[1]):
		if scores[0,i]>0.5:
			if(str(category_index.get(classes[0,i])['name']) == "gun"):
				print("0, ",end='')
				flg_psr = 1
				break
			elif(str(category_index.get(classes[0,i])['name']) == "knife"):
				print("1, ",end='')
				flg_psr = 1
				break
	if(flg_psr==0):
		print("2, ",end='')
		
for imno in range(1,90):
	imno_pic = str("C:\\Users\\HP PC\\Documents\\Tensorflow\\workspace\\training_demo\\test\\k ("+str(imno)+").jpg")
	image = cv2.imread(imno_pic)
	image_expanded = np.expand_dims(image, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})

	flg_psr = int(0)
	for i in range(scores.shape[1]):
		if scores[0,i]>0.5:
			if(str(category_index.get(classes[0,i])['name']) == "gun"):
				print("0, ",end='')
				flg_psr = 1
				break
			elif(str(category_index.get(classes[0,i])['name']) == "knife"):
				print("1, ",end='')
				flg_psr = 1
				break
	if(flg_psr==0):
		print("2, ",end='')
		
for imno in range(1,92):
	imno_pic = str("C:\\Users\\HP PC\\Documents\\Tensorflow\\workspace\\training_demo\\test\\o ("+str(imno)+").jpg")
	image = cv2.imread(imno_pic)
	image_expanded = np.expand_dims(image, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})

	flg_psr = int(0)
	for i in range(scores.shape[1]):
		if scores[0,i]>0.5:
			if(str(category_index.get(classes[0,i])['name']) == "gun"):
				print("0, ",end='')
				flg_psr = 1
				break
			elif(str(category_index.get(classes[0,i])['name']) == "knife"):
				print("1, ",end='')
				flg_psr = 1
				break
	if(flg_psr==0):
		print("2, ",end='')

# Clean up
cv2.destroyAllWindows()