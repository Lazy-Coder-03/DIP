import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

img = cv2.imread('input.png', 0)
factor = 20
resize_image = resize_image(img, 16 * factor, 9 * factor)
cv2.imwrite('output.png', resize_image)
print(np.max(resize_image), np.min(resize_image))

def getHistogram(image):
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[int(image[i, j])] += 1
    return hist

def histogramStretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i, j] = int((image[i, j] - min_val) * 255 / (max_val - min_val))
    return new_image

def histogramEqualization(image):
    hist = getHistogram(image)
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    cdf_min = cdf[np.nonzero(cdf)][0]
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i, j] = int((cdf[image[i, j]] - cdf_min) * 255 / (image.shape[0] * image.shape[1] - cdf_min))
    return new_image
# use plt to show the image and bar histogram with bars for each value using subplots and use np to get the values of the histogram

fig, axs = plt.subplots(3, 2, figsize=(10, 5))
axs[0, 0].imshow(resize_image, cmap='gray', aspect='auto')  # Corrected here
axs[0, 0].set_title('Image')
hist = getHistogram(resize_image)
axs[0, 1].bar(np.arange(256), hist)
axs[0, 1].set_title('Histogram before stretching')

newimage = histogramStretching(resize_image)
axs[1, 0].imshow(newimage, cmap='gray', aspect='auto')  # Corrected here
axs[1, 0].set_title('Image after stretching')
hist = getHistogram(newimage)
axs[1, 1].bar(np.arange(256), hist)
axs[1, 1].set_title('Histogram after stretching')

nwerimg=histogramEqualization(resize_image)
axs[2, 0].imshow(nwerimg, cmap='gray', aspect='auto')  # Corrected here
axs[2, 0].set_title('Image after equalization')
hist = getHistogram(nwerimg)
axs[2, 1].bar(np.arange(256), hist)
axs[2, 1].set_title('Histogram after equalization')

plt.show()
