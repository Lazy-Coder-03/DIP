import cv2
import numpy as np

img=cv2.imread("images/cat.jpg",0)

height,width=img.shape

gamma=2.2

gamma_corrected=np.zeros(shape=(height,width),dtype=np.uint8)

for i in range(height):
    for j in range(width):
        gamma_corrected[i,j]=255*(img[i,j]/255)**gamma
        
        
cv2.imshow("Original",img)

cv2.imshow("Gamma Corrected",gamma_corrected)

cv2.waitKey(0)

cv2.destroyAllWindows()

