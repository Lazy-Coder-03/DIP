import cv2
import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("images/luc72.jpg")

height,width,ch=img.shape

def gethist(img):
    hist=np.zeros(256,dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            hist[img[i,j]]+=1
    return hist


hist=gethist(img)

fig,ax=plt.subplots(1,2,figsize=(15,5))

ax[0].imshow(img,cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")


ax[1].bar(np.arange(len(hist)), hist, color="black")
ax[1].set_title("Histogram")

plt.tight_layout()
plt.show()


