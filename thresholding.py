from __future__ import print_function
import numpy as np
import argparse
import cv2
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])[300:-1, 70:540]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

#(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
#cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)

#cv2.imshow("Threshold Binary Inverse", threshInv)

#cv2.imshow("Coins", cv2.bitwise_and(image, image, mask = threshInv))




# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# help = "Path to the image")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
# cv2.imshow("Image", image)

edged = cv2.Canny(threshInv, 30, 200)
#cv2.imshow("Edges", edged)

(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

same_row_rects = {}
colors = [(0,255,0),(255,0,0),(0,0,255)]
same_row_rects_colors = {}
colors_used = 0
for idx, cnt in enumerate(cnts):
	area = cv2.contourArea(cnt)

	if area > 900:
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

			if math.fabs(key-boxKey) < 30:
				key_found = True
				same_row_rects[key].append(box)
				cv2.drawContours(image,[box],0, same_row_rects_colors[key], 3)

		if key_found == False:
			same_row_rects[boxKey] = []
			same_row_rects[boxKey].append(box)
			same_row_rects_colors[boxKey] = colors[colors_used]
			cv2.drawContours(image,[box],0, colors[colors_used], 3)
			colors_used += 1

		#same_row_rects.append(box[:,1])

		#if idx > 0:
		#	cv2.drawContours(image,[box],0, (0, 255, 122), 3)
			#print(np.subtract(box[:,1],same_row_rects[idx-1]))


# print("I count {} coins in this image".format(len(cnts)))

#coins = image.copy()
#cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Coins", image)
# cv2.waitKey(0)

# cv2.drawContours(coins, cnts, 0, (0, 255, 0), 2)
# cv2.drawContours(coins, cnts, 1, (0, 255, 0), 2)
# cv2.drawContours(coins, cnts, 2, (0, 255, 0), 2)

# for (i, c) in enumerate(cnts):
# 	(x, y, w, h) = cv2.boundingRect(c)

# print("Coin #{}".format(i + 1))
# coin = image[y:y + h, x:x + w]
# cv2.imshow("Coin2", coin)

# mask = np.zeros(image.shape[:2], dtype = "uint8")

# ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
# cv2.circle(mask, (int(centerX), int(centerY)), int(radius),255, -1)
# mask = mask[y:y + h, x:x + w]
# cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
# cv2.waitKey(0)

cv2.waitKey(0)