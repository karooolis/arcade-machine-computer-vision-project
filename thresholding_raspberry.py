from __future__ import print_function
import RPi.GPIO as GPIO
import numpy as np
import argparse
import cv2
import math
import time
import heapq

# setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.OUT)

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

BUTTON_PRESSED = False

total_keys = 0

# keep looping
while True:
	input_state = GPIO.input(18)
	if input_state == False:
		print("Button Pressed")
		BUTTON_PRESSED = not BUTTON_PRESSED
		time.sleep(0.1)

	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # [-130-(total_keys*10):-50,10:-30]

	height,width = image.shape
	circle_img = np.zeros((height,width), np.uint8)
	cv2.rectangle(circle_img,(20,20),(width-30,height-30),(255,0,125),thickness=-1)

	masked_data = cv2.bitwise_and(image, image, mask=circle_img)

	cv2.imshow("Mask", masked_data)

	image = masked_data

	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
	edged = cv2.Canny(threshInv, 180, 255)

	cv2.imshow("thresInv", threshInv)
	cv2.imshow("Frame edged", edged)

	(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	if BUTTON_PRESSED:
		colors = [(0,255,0),(255,0,0)] #,(0,0,255)
		same_row_rects_colors = {}
		same_row_rects = {}
		colors_used = 0

		for idx, cnt in enumerate(cnts):
			area = cv2.contourArea(cnt)
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)

			width = math.fabs(box[0][0]-box[2][0])
			height = math.fabs(box[0][1]-box[2][1])

			# print(box, width, height, area)

			if area > 0.2 and area < 550 and width > 5 and height > 5 and width < 50 and height < 50:
				boxKey = np.sum(box[:,1])/4;

				cv2.drawContours(image,[box],0, colors[0], 3)

				key_found = False
				for key in same_row_rects:
					if math.fabs(key-boxKey) < 5:
						key_found = True
						boxKey = key
						same_row_rects[boxKey].append(box)

				if key_found == False:
					same_row_rects[boxKey] = []
					same_row_rects[boxKey].append(box)

		totals = []
		same_row_rects_keys = list(same_row_rects.keys())
		total_keys = len(same_row_rects_keys)
		for idx, key in enumerate(sorted(same_row_rects_keys)):
			for box in same_row_rects[key]:
				cv2.drawContours(frame,[box],0, colors[idx%2], 3)

			val = same_row_rects[key]
			np_array = np.array(val)
			x_totals = [np.sum(arr[:,0]) for arr in np_array]
			totals.append(min(x_totals))

                print(totals)
                if len(totals) > 2:
                        totals = totals[0:2]
                
		if len(totals) == 2 and math.fabs(totals[0]-totals[1]) < 20:
                        GPIO.output(23,GPIO.HIGH)
                        print("PRESS NOW!")

	# print("-----------------------------")
        
	cv2.imshow("Frame", frame)
	GPIO.output(23,GPIO.LOW)
	time.sleep(0.01)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
