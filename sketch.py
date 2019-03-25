import imageio
img="imagelocation"  ###enter your file location here!!!
start_img = imageio.imread(img)

import numpy as np
def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
gray_img = grayscale(start_img)
inverted_img = 255-gray_img
import scipy.ndimage
blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=30) ### if you want a darker sketch increase the value of sigma
def dodge(front,back):
    result=front*255/(255-back) 
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')
final_img= dodge(blur_img,gray_img)
import matplotlib.pyplot as plt
plt.imshow(final_img, cmap="gray")
plt.imsave('outputlocation', final_img, cmap='gray', vmin=0, vmax=255) ###enter the directory in which u want the image to be saved
