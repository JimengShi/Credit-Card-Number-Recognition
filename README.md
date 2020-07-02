# Credit-Card-Number-Recognition
A computer vision project about number recognition of credit card is achieved by OpenCV.


## Framework
<div align="center">
<img src="https://github.com/JimengShi/Credit-Card-Number-Recognition/blob/master/images/Framework.png" alt="Framework" >
</div>

### PART I: preparation
(1.1) import necessary packages

(1.2) set parameters which are imported later

(1.3) assign credit card types

(1.4) visualization function


### PART II: operations for template image
(2.1) read and show template image

(2.2) convert color template image to gray image

(2.3) convert template to binary image

(2.4) compute external contours with function: cv2.findContours()

(2.5) traverse template contours list and save it in a dictionary


### PART III: operations for test image
(3.1) initialize convolutional kernel

(3.2) pre-processing for test image: read and show

(3.3) morphological operation: top-hat to highlight the brighter area

(3.4) morphological operation: Sobel Operator

(3.5) closed operation (expand first, erode then) to connect numbers together

(3.7) filter useless background by set threshold

(3.8) closed operation again

(3.9) compute contour of test image

(3.10) traverse contours of test image and get contours in the numbers area

(3.11) sort all contours from left to right

(3.12) traverse contour of each number of each appropriate area


### PART IV: print, show and save result
