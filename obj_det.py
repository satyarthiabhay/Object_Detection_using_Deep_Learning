#usr/bin//env python
''' 		Object Detection with Deep Learning
'''
#generic/Built-in
import argparse
import os
import sys
import time
#other lib
import numpy
from cv2 import cv2 

__author__ 		= "Abhay Satyarthi", "Akansha", "Kirti"
__credits__ 	= ["Abhay, Akansha, Kirti"]
__version__ 	= "1.0"
__maintainer__ 	= "Abhay Satyarthi"
__email__ 		= "satyarthiabhay@gmail.com"
__status__ 		= "Development"

#construct the argument parse and parse the arguments
def parse_cmd_line():
	
	ap = argparse.ArgumentParser(description = 'Object Detection with Deep Learning',
							  	epilog ='Enjoy Object Detection!')

	ap.add_argument('-img', '--image'     ,
					type = str  ,
					help = 'Image / Directory containing image')

	ap.add_argument('-vid', '--video'     ,
					type = str  ,
					help = 'Video / Directory containing image')

	ap.add_argument('-y'  , '--yolo'      , default = 'yolo-coco',
					type = str  ,
					help = 'Base path to YOLO directory')

	ap.add_argument('-o'  , '--output_dir', default = 'Output/'	 ,
					type = str  ,
					help = 'Directory to store Output Image')

	ap.add_argument('-l'  , '--given_list', default = [],
					type = str  ,
					help = 'Detect given set of objects')

	ap.add_argument('-i'  , '--ignore'	  , default = [],
					type = str	,
					help = 'Ignore set of objects')

	ap.add_argument('-c'  , '--confidence', default = 0.5,
					type = float,
					help = 'Minimum probability to filter weak detections')

	ap.add_argument('-t'  , '--threshold' , default = 0.3,
					type = float,
					help = 'Threshold when applying non-maxima suppression')

	ap.add_argument('-ht' , '--height'	  , default = 720,
					type = int	,
					help = 'Height of image')

	ap.add_argument('-wt' , '--width'	  ,
				type = int	,
				help = 'Width of image')
	
	return ap.parse_args()

def load_yolo():
	#load all the COCO class object_names
	object_name_path = os.path.sep.join([yolo_path, 'coco.names'])
	object_names	 = open(object_name_path).read().strip().split('\n')

	#initialize a list of colors to represent each possible class label
	numpy.random.seed(42)
	colors = numpy.random.randint(0, 255, size = (len(object_names), 3), dtype="uint8")

	#the YOLO net weights file
	weights_path = os.path.sep.join([yolo_path, 'yolov3.weights'])

	#the neural network configuration
	config_path	 = os.path.sep.join([yolo_path, 'yolov3.cfg'])

	#load our YOLO network trained on COCO dataset (80 classes)
	print('[INFO]Loading YOLO network from disk...')
	net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

	# get all the layer names
	layers_name = net.getLayerNames()
	# determine only the *output* layer names that we need from YOLO
	output_layers = [layers_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	return object_names, colors, net, output_layers

def load_image(image):
	# load our input image
	image = cv2.imread(image)

	#grab its spatial dimensions
	(h, w) = image.shape[:2]

	#image resize
	if wt is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r	= ht / float(h)
		dim = (int(w * r), ht)
		# otherwise, the height is None
	else:
		dim = (wt,ht)
	image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

	#grab its new spatial dimensions after resizing
	(h, w) = image.shape[:2]
	return image, h, w

def start_webcam():
	cap = cv2.VideoCapture(0)
	return cap

def detect_objects(image, net, output_layers):
	# construct a blob from the input image
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	# sets the blob as the input of the network
	net.setInput(blob)

	# and then perform a forward pass of the YOLO object detector,
	# giving us our bounding boxes and associated probabilities
	# measure how much it took in seconds
	start = time.time()
	layer_outputs = net.forward(output_layers)
	end = time.time()
 
	# yolo_time = time taken by YOLO
	yolo_time = end - start
 
	return yolo_time, layer_outputs

def get_box_dimensions(layer_outputs, h, w):
	boxes, confidences, class_ids = [], [], []

	# loop over each of the layer outputs
	for output in layer_outputs:
		# loop over each of the detections
		for detection in output:
			# extract the class id (label) and confidence (as a probability) of
			# the current object detection
			scores = detection[5:]
			class_id = numpy.argmax(scores)
			confidence = scores[class_id]
			# discard out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > confi:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[:4] * numpy.array([w, h, w, h])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	return boxes, confidences, class_ids

def show_detections(boxes, confidences, colors, class_ids, object_names, image):
	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, confi, thres)
	det_objs = []
	# ensure at least one detection exists
	if len(indexes) > 0:
		# loop over the indexes we are keeping
		
		for i in indexes.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#if given_list is empty detect all object
			#AND If given_list is non-empty, detect object from given_list
			if len(given_list)!= 0 and object_names[class_ids[i]] not in given_list:
				continue
			#ignore items from ignore
			if object_names[class_ids[i]] in ignore_list:
				continue
			#Draw a colored bounding box rectangle and label on the image
			color = [int(c) for c in colors[class_ids[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = '{}: {:.2%}'.format(object_names[class_ids[i]].capitalize(), confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			det_objs.append(object_names[class_ids[i]])
	
	return det_objs , image
 
def obj_det_table(det_objs):
	# Get unique values, thier frequnecy count & first index position
	(obj_name , frequency)	= numpy.unique(det_objs, return_counts=True)
	print('_'*80)
	print('\nThere are total {} objects and {} distinct objects'.format(sum(frequency),len(obj_name)))
	#table of object name with frequency on console
	print('_'*26)
	print('{:^15}{}{:^10}{}'.format('Object Name','|','Frequency','|'))
	print('-'*26+'|')
	# Iterate over the ziped object and display each unique value along with frequency count
	for obj,freq in zip(obj_name , frequency):
		print('{:^15}{}{:^10}{}'.format(obj.capitalize(),'|',freq,'|'))
	print('_'*26+'|')
	print('_'*80)

def image_detect(image):
	object_names, colors, net, output_layers = load_yolo()
	try:
		image, h, w = load_image(image)
	except:
		raise 'Image cannot be loaded!\n Please check the path provided!'
	finally:
		starting_time = time.time()
		yolo_time, layer_outputs = detect_objects(image, net, output_layers)
		print ("[INFO] YOLOv3 took {:6f} seconds".format(yolo_time))
		boxes, confidences, class_ids = get_box_dimensions(layer_outputs, h, w)
		#store object detected and image in det_objs and op_img respectively
		det_objs, op_img = show_detections(boxes, confidences, colors, class_ids, object_names, image)
		elapsed_time = time.time() - starting_time
		#show object detection image : op_img
		cv2.imshow('Output Image',op_img)
		#show obj_det table
		obj_det_table(det_objs)
		
		print('Time taken by model to detect objects: {:3f}'.format(elapsed_time))
		while True:
			key = cv2.waitKey(0)  & 0xFF
			if key == 27:         # wait for ESC key to exit
				break
			elif key == ord('s'): # wait for 's' key to save and exit
				#store filename and extension of image in filename and ext respectively
				fname_ext	 = os.path.basename(ip_img)
				(fname, ext) = fname_ext.split('.')
				op_img_name	 = fname +' _yolov3.'+ext
				#save image 
				cv2.imwrite(op_dir+op_img_name,op_img)
				break

def webcam_detect():
	object_names, colors, net, output_layers = load_yolo()
	try:
		cap = start_webcam()
	except:
		raise 'Webcam can not be strted !\n Please check the webcam!'
	finally:
		starting_time = time.time()
		frame_id = 0

		while True:
			_, frame = cap.read()
			frame_id += 1
			height, width  = frame.shape[:2]
			_, layer_outputs = detect_objects(frame, net, output_layers)
			boxes, confidences, class_ids = get_box_dimensions(layer_outputs, height, width)
			#store frame in  op_frame 
			_, op_frame = show_detections(boxes, confidences, colors, class_ids, object_names, frame)
			#show object detection image : op_img
			
			cv2.imshow('Real-Time',op_frame)
			key = cv2.waitKey(1)  & 0xFF
			if key == 27:         # wait for ESC key to exit
				elapsed_time = time.time() - starting_time
				break
		fps = frame_id / elapsed_time
		print('Time taken by model to detect objects per frame: {:3f} seconds'.format(1/fps))
		cap.release()
		
def video_detect(video):
	object_names, colors, net, output_layers = load_yolo()
	try:
		cap = cv2.VideoCapture(video)
		height, width = None, None
		op_vid = None
	except:
		raise 'Video cannot be loaded!\n Please check the path provided!'
	finally:
		starting_time = time.time()
		frame_id = 0
		while True:
			grabbed, frame = cap.read()
			frame_id += 1
			# Checking if the complete video is read
			if not grabbed:
				elapsed_time = time.time() - starting_time
				break
			if width is None or height is None:
				height, width = frame.shape[:2]

			_, layer_outputs = detect_objects(frame, net, output_layers)
			boxes, confidences, class_ids = get_box_dimensions(layer_outputs, height, width)
			_, op_frame = show_detections(boxes, confidences, colors, class_ids, object_names, frame)
			cv2.imshow('Output Video',op_frame)
			
			if op_vid is None:
				fourcc = cv2.VideoWriter_fourcc(*'MPEG')
				op_vid = cv2.VideoWriter(op_dir+'yolov3.mp4', fourcc, 30, 
									(frame.shape[1], frame.shape[0]), True)
				op_vid.write(frame)
			key = cv2.waitKey(1)  & 0xFF
			if key == 27:         # wait for ESC key to exit
				elapsed_time = time.time() - starting_time
				break
		fps = frame_id / elapsed_time
		print('Time taken by model to detect objects per frame: {:3f} seconds'.format(1/fps))		
		print ("[INFO] Cleaning up...")
		cap.release()
		op_vid.release()

if __name__ == '__main__':
	
	args	= parse_cmd_line()
	ht		= args.height
	wt		= args.width
	out_dir	= args.output_dir
	confi	= args.confidence
	thres	= args.threshold
	ip_img	= args.image
	ip_vid	= args.video
	yolo_path = args.yolo
	op_dir	  = args.output_dir
	#store given_list in GIVEN_LIST
	if args.given_list:
		given_list = [str(item) for item in args.given_list.split(',')]
	else:
		given_list = []

	#store ignore in ignore_list
	if args.ignore:
		ignore_list = [str(item) for item in args.ignore.split(',')]
	else:
		ignore_list = []

	# If both image and video files are given then raise error
	if ip_img is None and ip_vid is None:
		print ('Neither path to an image or path to video provided')

	# Do inference with given image
	if ip_img:
		image_detect(ip_img)

	elif ip_vid:
		# Read the video
		video_detect(ip_vid)

	else:
		print('---- Starting Web Cam object detection ----')
		webcam_detect()

	cv2.destroyAllWindows()

#last line of the program