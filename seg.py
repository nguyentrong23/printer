import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import color

image = cv2.imread('img/snake1.bmp')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

gradient = cv2.morphologyEx(blurred_image, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
thershold = cv2.threshold(gradient ,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
distance_transform = cv2.distanceTransform(thershold, cv2.DIST_L2, 5)

markers = np.zeros_like(gray_image)
markers[gray_image < 100] = 1
markers[gray_image > 150] = 2
local_max = peak_local_max(distance_transform, footprint=np.ones((3, 3)), labels=markers)
markers[local_max[:, 0], local_max[:, 1]] = 3

labels = watershed(-distance_transform, markers, mask=(gradient > 0))

cv2.imshow("gra",gradient)
# cv2.imshow("thershold",thershold)
# cv2.imshow("distance_transform",distance_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()
