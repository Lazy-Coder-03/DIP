import cv2
import numpy as np
import matplotlib.pyplot as plt

# bgrimg=cv2.imread("images/rainbow.jpg")
# img=cv2.cvtColor(bgrimg,cv2.COLOR_BGR2RGB)
img=cv2.imread("images/say1.jpg")
height,width,ch=img.shape

blueimg=np.zeros_like(img)
greenimg=np.zeros_like(img)
redimg=np.zeros_like(img)

withoutred=np.zeros_like(img)

for i in range(height):
    for j in range(width):
        withoutred[i,j,0]=0
        withoutred[i,j,1]=img[i,j,1]
        withoutred[i,j,2]=img[i,j,2]
        
  
  
withoutgreen=np.zeros_like(img)        
for i in range(height):
    for j in range(width):
        withoutgreen[i,j,0]=img[i,j,0]
        withoutgreen[i,j,1]=0
        withoutgreen[i,j,2]=img[i,j,2]
        
withoutblue=np.zeros_like(img)
for i in range(height):
    for j in range(width):
        withoutblue[i,j,0]=img[i,j,0]
        withoutblue[i,j,1]=img[i,j,1]
        withoutblue[i,j,2]=0
        
        
cv2.imshow("Without blue",withoutblue)
cv2.imshow("Without green",withoutgreen)
        
        
        
        
cv2.imshow("Original",img)
cv2.imshow("With out red",withoutred)        
        
def seperator(img):
    height,width=img.shape[:2]
    blueimg=np.zeros_like(img)
    greenimg=np.zeros_like(img)
    redimg=np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            blueimg[i,j,0]=img[i,j,0]
            greenimg[i,j,1]=img[i,j,1]
            redimg[i,j,2]=img[i,j,2]
    return blueimg,greenimg,redimg


blueimg,greenimg,redimg=seperator(img)

cv2.imshow("Original",img)
cv2.imshow("Red",redimg)
cv2.imshow("Green",greenimg)
cv2.imshow("Blue",blueimg)

cv2.waitKey(0)
cv2.destroyAllWindows()


# fig,axes=plt.subplots(1,4,figsize=(8,8))

# images=[img,redimg,greenimg,blueimg]
# titles=["Main","Red","Green","Blue"]
# fig,axes=plt.subplots(1,4,figsize=(16,4))

# for i in range(len(images)):
#   axes[i].imshow(images[i])
#   axes[i].set_title(titles[i])
#   axes[i].axis("off")
  
# plt.show()  