import cv2
import time
import imutils
import argparse
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream

#Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = 'deploy.prototxt')
ap.add_argument("-m", "--model", default = 'res10_300x300_ssd_iter_140000.caffemodel')
ap.add_argument("-c", "--confidence", type = float, default = 0.7)
args = vars(ap.parse_args())

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#Initializing Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while(True):

	#Reading camera feed
	frame = vs.read()
	frame = imutils.resize(frame, width = 600)
	(h, w) = frame.shape[:2]

	#Converting image to blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	#Passing blob through network
	nn.setIntput(blob)
	detections = nn.forward()

	#Loop over detections
	for i in range(0, detections.shape[2]):
		
		#Extracting confidence of detections
		confidence = detections[0, 0, i, 2]

		#Filtering out weak detections
		if confidence < args["confidence"]:
			continue

		#Bounding box coordinates
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		#Drawing bounding box with confidence of detection
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	#Showing output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	#Exit using 'q'
 	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()