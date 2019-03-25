import cv2
import numpy as np
img1 = cv2.imread('Groundtruthhere')  ##replace with Original image here
img2 = cv2.imread('Sketchlocationhere') ## replace with the sketch here
vis = np.concatenate((img1, img2), axis=1)
cv2.imwrite('out.jpg', vis) ##mention the directory where you want the concatenated image to be saved
