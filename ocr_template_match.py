# PART I: preparation
# (1.1) import necessary packages
from imutils import contours
import imutils
import numpy as np
import argparse
import cv2
import myutils

# (1.2) set parameters which are imported later
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image", required=True)
ap.add_argument("-t", "--template", help="path to template OCR-A image", required=True)
args = vars(ap.parse_args())

# (1.3) assign credit card types
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"}

# (1.4) visualization function
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# PART II: operations for template image
# (2.1) read and show template image
img = cv2.imread(args["template"])
cv_show('img',img)

# (2.2) convert color template image to gray image
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)

# (2.3) convert template to binary image
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# (2.4) compute external contours with function: cv2.findContours()
# (2.4) parameter 1: ref.copy(): binary image (black-white)
# (2.4) parameter 2: cv2.RETR_EXTERNAL: only check external contours
# (2.4) parameter 3: cv2.CHAIN_APPROX_SIMPLE: only save terminal coordinate
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # return contour of each number in template image
refCnts = refCnts[1] if imutils.is_cv3() else refCnts[0]                             # check opencv version
# print('refCnts:',refCnts)
cv2.drawContours(img, refCnts, -1, (0,0,255), 3)                                     # -1: means it draws all contours
cv_show('img',img)
print('shape of refCnts:', np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]  				 # sort every contour by left-to-right order

# (2.5) traverse template contours list and save it in a dictionary
digits = {}
# traverse contour of each number in template image
for (i, c) in enumerate(refCnts):
	# compute external rectangle and resize it
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y+h, x:x+w]
	roi = cv2.resize(roi, (57,88))
	# map each contour with number (0---0, 1---1, ......) with a dictionary
	digits[i] = roi


# PART III: operations for test image
# (3.1) initialize convolutional kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# (3.2) pre-processing for test image: read and show
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

# (3.3) morphological operation: top-hat to highlight the brighter area
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

# (3.4) morphological operation: Sobel Operator
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)       # ksize=-1相当于用3*3的
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
print('shape of gradX:',np.array(gradX).shape)
cv_show('gradX',gradX)

# (3.5) closed operation (expand first, erode then) to connect numbers together
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX',gradX)

# (3.7) THRESH_OTS: filter useless background by set threshold == 0, which can automatically set threshold (适合双峰)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

# (3.8) closed operation again
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh',thresh)

# (3.9) compute contour of test image
threshCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
threshCnts = threshCnts[1] if imutils.is_cv3() else threshCnts[0]
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0,0,255), 3)
cv_show('img',cur_img)

# (3.10) traverse contours of test image and get contours in the numbers area
locs = []
for (i, c) in enumerate(cnts):
	# compute external rectangle
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# select numbers area and filter other areas like (Name, Visa, Master, Expiration Date) by width / height ratio
	if ar > 2.5 and ar < 4.0:
		if (w > 40 and w < 55) and (h > 10 and h < 20):
			# save the appropriate area
			locs.append((x, y, w, h))

# (3.11) sort all contours from left to right
locs = sorted(locs, key=lambda x: x[0])

# (3.12) traverse contour of each number of each appropriate area
output = []
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []

	# select each big area (include 4 numbers in every single area, like 4359 5862 1254 5789)
	group = gray[gY-5 : gY+gH+5, gX-5 : gX+gW+5]
	cv_show('group',group)

	# pre-processing for group: convert it into a binary image
	group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)

	# compute contour of each area
	digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = digitCnts[1] if imutils.is_cv3() else digitCnts[0]
	digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

	# compute each number of each group (like 4, 3, 5, 9 in 4359)
	for c in digitCnts:
		# compute external contour of each number and resize it
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y+h, x:x+w]
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)

		# compute matching score for each number in template image
		scores = []
		for (digit, digitROI) in digits.items():
			# template match
			result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# append the number with max score
		groupOutput.append(str(np.argmax(scores)))

	# show image and add some corresponding text on the image
	cv2.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0,0,255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# update output
	output.extend(groupOutput)


# PART IV: print, show and save result
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imwrite('./images/check_05.png', image)