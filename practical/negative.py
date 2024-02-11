import cv2
import numpy as np

img=cv2.imread("images/cat.jpg")
height,width=img.shape

neg=np.zeros(shape=(height,width),dtype=np.uint8)

for i in range(height):
  for j in range(width):
    neg[i,j]=255-img[i,j]

cv2.imshow("Original",img)

cv2.imshow("Negative",neg)

cv2.waitKey(0)