#r=0.114,g=0.587,b=0.299

import cv2
import numpy as np

img=cv2.imread("images/cat.jpg")
height,width,ch=img.shape

grey=np.zeros(shape=(height,width),dtype=np.uint8)

simplegrey=np.zeros(shape=(height,width),dtype=np.uint8)

for i in range(height):
  for j in range(width):
    grey[i,j]=int(0.114*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,2])
    
    
for i in range(height):
    for j in range(width):
        simplegrey[i,j]=(int(img[i,j,0])+int(img[i,j,1])+int(img[i,j,2]))//3

    
cv2.imshow("original",img)

cv2.imshow("grey",grey)

cv2.imshow("Simplegrey",simplegrey)

cv2.waitKey(0)