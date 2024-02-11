import cv2
import numpy as np


def intensitySlicingBg(img,low,high):
    height,width=img.shape
    result=np.zeros(shape=(height,width),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i,j]>low and img[i,j]<high:
                result[i,j]=255
            else:
                result[i,j]=img[i,j]
    return result

def instensitySlicingwithoutbg(img,low,high):
    height,width=img.shape
    result=np.zeros(shape=(height,width),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i,j]>low and img[i,j]<high:
                result[i,j]=img[i,j]
            else:
                result[i,j]=0
    return result

img=cv2.imread("images/cat.jpg",0)

low=100
high=200


def imgneg(img):
    height,width=img.shape
    result=np.zeros(shape=(height,width),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i,j]=255-img[i,j]
    return result

bg=intensitySlicingBg(img,low,high)
nobg=instensitySlicingwithoutbg(img,low,high)
neg=imgneg(img)

cv2.imshow("Original",img)

cv2.imshow("Background",bg)

cv2.imshow("Foreground",nobg)

cv2.imshow("Negative",neg)

cv2.waitKey(0)

cv2.destroyAllWindows()

