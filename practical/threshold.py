import cv2
import numpy as np

img=cv2.imread("images/cat.jpg",0)

height,width=img.shape

minval=np.min(img)
maxval=np.max(img)

t=0.6*(minval+maxval)

thresholded=np.zeros(shape=(height,width),dtype=np.uint8)

for i in range(height):
    for j in range(width):
        if img[i,j]>t:
            thresholded[i,j]=255
        else:
            thresholded[i,j]=0
            
cv2.imshow("Original",img)

cv2.imshow("Thresholded",thresholded)

cv2.waitKey(0)

cv2.destroyAllWindows()
