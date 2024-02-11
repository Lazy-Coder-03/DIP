import cv2
import numpy as np

def rotateImg(img, angle):
    height,width=img.shape
    rotated=np.zeros(shape=(height,width),dtype=np.uint8)
    angle=np.deg2rad(angle)
    
    for i in range(height):
        for j in range(width):
            x=int((i-height/2)*np.cos(angle)-(j-width/2)*np.sin(angle)+height/2)
            y=int((i-height/2)*np.sin(angle)+(j-width/2)*np.cos(angle)+width/2)
            
            if x>=0 and x<height and y>=0 and y<width:
                rotated[i,j]=img[x,y]
            else:
                rotated[i,j]=0

    return rotated

img=cv2.imread("images/cat.jpg",0)

rotated=rotateImg(img,-45)

cv2.imshow("Original",img)

cv2.imshow("Rotated",rotated)

cv2.waitKey(0)

cv2.destroyAllWindows()


