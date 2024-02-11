import cv2
import numpy as np

img=cv2.imread("images/cat.jpg",0)

height,width=img.shape

# c=255/np.log(1+np.max(img))
c=50

log_transformed=np.zeros(shape=(height,width),dtype=np.uint8)

for i in range(height):
    for j in range(width):
        log_transformed[i,j]=int(c*np.log(1+img[i,j]))
        
        log_transformed[i,j]=np.clip(log_transformed[i,j],0,255)
cv2.imshow("Original",img)

cv2.imshow("Log Transformed",log_transformed)

cv2.waitKey(0)

cv2.destroyAllWindows()

