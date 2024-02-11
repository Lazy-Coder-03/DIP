import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a 400x400 matrix filled with zeros
matrix_size = 1000
matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint8)


def getNormalImg(height, width, mean, std):
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j] = np.random.normal(mean, std)
    return sortImg(img)

def getRangeImg(height, width, minval, maxval):
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j] = np.random.randint(minval, maxval)
    return sortImg(img)



def sortImg(img):
    sortedImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    sortedList = np.sort(img.flatten())
    k = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sortedImg[i, j] = sortedList[k]
            k += 1
    return sortedImg

        

def getHist(img):
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1
    return hist


def histStrecht(img):
    minval = np.min(img)
    maxval = np.max(img)
    strechedImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            strechedImg[i, j] = mapval(img[i, j], minval, maxval, 0, 255)
    return strechedImg


def mapval(x, minval, maxval, a, b):
    return (x - minval) * (b - a) / (maxval - minval) + a


def histEquilise(img):
    hist = getHist(img)
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdfmin = np.min(cdf)
    equilisedImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equilisedImg[i, j] = mapval(cdf[img[i, j]], cdfmin, cdf[255], 0, 255)
    return equilisedImg


matrix = getRangeImg(matrix_size, matrix_size, 0, 155)

plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.imshow(matrix, cmap="gray")
plt.title("Normal Image")

plt.subplot(3, 2, 2)
plt.bar(np.arange(len(getHist(matrix))), getHist(matrix), color="black", width=1.0)
plt.title("Histogram")

matrix = histStrecht(matrix)
plt.subplot(3, 2, 3)
plt.imshow(matrix, cmap="gray")
plt.title("streched Image")

plt.subplot(3, 2, 4)
plt.bar(np.arange(len(getHist(matrix))), getHist(matrix), color="black", width=1.0)
plt.title("Histogram")

matrix = histEquilise(matrix)
plt.subplot(3, 2, 5)
plt.imshow(matrix, cmap="gray")
plt.title("Equilised Image")

plt.subplot(3, 2, 6)
plt.bar(np.arange(len(getHist(matrix))), getHist(matrix), color="black", width=1.0)
plt.title("Histogram")




plt.show()




