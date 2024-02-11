import cv2
import numpy as np

img=cv2.imread("images/say1.jpg",0)
height,width=img.shape



kernel=1

def mean_filter(img, kernel):
    height, width = img.shape
    meanfilter = np.zeros_like(img)
    for i in range(kernel, height - kernel):
        for j in range(kernel, width - kernel):
            meanfilter[i, j] = np.mean(img[i - kernel:i + kernel + 1, j - kernel:j + kernel + 1])

    return meanfilter


def max_filter(img, kernel):
    maxfilter=np.zeros_like(img)
    for i in range(kernel,height-kernel):
        for j in range(kernel,width-kernel):
            maxfilter[i,j]=np.max(img[i - kernel:i + kernel + 1, j - kernel:j + kernel + 1])
            
    return maxfilter

def min_filter(img, kernel):
    minfilter=np.zeros_like(img)
    for i in range(kernel,height-kernel):
        for j in range(kernel,width-kernel):
            minfilter[i,j]=np.min(img[i - kernel:i + kernel + 1, j - kernel:j + kernel + 1])
            
    return minfilter

def median_filter(img, kernel):
    medianfilter=np.zeros_like(img)
    for i in range(kernel,height-kernel):
        for j in range(kernel,width-kernel):
            medianfilter[i,j]=np.median(img[i - kernel:i + kernel + 1, j - kernel:j + kernel + 1])
            
    return medianfilter




mean=mean_filter(img,kernel)
maxf=max_filter(img,kernel)
minf=min_filter(img,kernel)
medianf=median_filter(img,kernel)
        
cv2.imshow("Original",img)
cv2.imshow("Mean Filter",mean)
cv2.imshow("Max Filter",maxf)
cv2.imshow("Min Filter",minf)
cv2.imshow("Median Filter",medianf)



cv2.waitKey(0)

cv2.destroyAllWindows()
