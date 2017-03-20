from __future__ import print_function
import numpy as np
import argparse
import cv2
import math
import time
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

start_monitor = False

# keep looping
while True:
	if cv2.waitKey(1) & 0xFF == ord("s"):
		start_monitor = True

	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width = 300)

	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#image = cv2.imread(args["image"])[300:-1, 70:540]
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	(T, threshInv) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
	edged = cv2.Canny(threshInv, 30, 200)

	# (_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# frameClone = cv2.drawContours(frame.copy(), cnts, 0, (0, 255, 0), 2)

	# cv2.imshow("Frame", edged)

	(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	if start_monitor:
		same_row_rects = {}
		colors = [(0,255,0),(255,0,0),(0,0,255)]
		same_row_rects_colors = {}
		colors_used = 0
		for idx, cnt in enumerate(cnts):
			area = cv2.contourArea(cnt)

			print(area)

			if area > 100:
				print(area)

				#x,y,w,h = cv2.boundingRect(cnt)
				#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

				rect = cv2.minAreaRect(cnt)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				
				#print(cnt)
				print(box[:,1], np.sum(box[:,1])/4)

				boxKey = np.sum(box[:,1])/4;

				key_found = False
				for key in same_row_rects:
					print(key, 'corresponds to', same_row_rects[key])

					if math.fabs(key-boxKey) < 13:
						key_found = True
						same_row_rects[key].append(box)
						cv2.drawContours(image,[box],0, same_row_rects_colors[key], 3)

				if key_found == False:
					same_row_rects[boxKey] = []
					same_row_rects[boxKey].append(box)
					same_row_rects_colors[boxKey] = colors[colors_used]
					cv2.drawContours(image,[box],0, colors[colors_used], 3)
					colors_used += 1

	print(start_monitor)

	cv2.imshow("Frame", image)

	time.sleep(0.05)

	# detect faces in the image and then clone the frameso that we can draw on it
	#faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
	#	minSize = (30, 30))
	#frameClone = frame.copy()

	# loop over the face bounding boxes and draw them
	#for (fX, fY, fW, fH) in faceRects:
	#	cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	# show our detected faces
	#cv2.imshow("Face", frameClone)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
