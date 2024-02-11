import cv2
import numpy as np

img=cv2.imread("images/say1.jpg")

height,width,ch=img.shape

neg=np.zeros_like(img)

for i in range(height):
    for j in range(width):
        neg[i,j,0]=255-img[i,j,0]
        neg[i,j,1]=255-img[i,j,1]
        neg[i,j,2]=255-img[i,j,2]
        
cv2.imshow("Original",img)
cv2.imshow("Negative",neg)

cv2.waitKey(0)
cv2.destroyAllWindows()