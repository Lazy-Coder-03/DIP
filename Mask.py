import numpy as np
import cv2

mask = np.array([[0,  0, 0],
                 [0,   1, 0],
                 [0,  0, 0]])

def applyMask(image, mask):
    new_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2), dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            pixel_value = np.sum(image[i - 1:i + 2, j - 1:j + 2] * mask)
            new_image[i - 1, j - 1] = np.clip(pixel_value, 0, 255)
    return new_image

def contrastStretch(image):
    min_val = np.min(image)
    max_val = np.max(image)
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i, j] = int((image[i, j] - min_val) * 255 / (max_val - min_val))
    return new_image

def getHistogram(image):
    hist = np.zeros(256, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[int(image[i, j])] += 1
    return hist

def histogramEqualization(image):
    hist = getHistogram(image)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)][0]
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i, j] = int((cdf[image[i, j]] - cdf_min) * 255 / (image.shape[0] * image.shape[1] - cdf_min))
    return new_image
def cannyEdgeDetection(image):
    blur= cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Canny(blur, 20, 120)

img = cv2.imread('mehu.png', 0)

# Apply the mask, contrast stretch, and histogram equalization
masked = applyMask(img, mask)
contrast_stretched = contrastStretch(img)
equalized = histogramEqualization(img)
edgeDetection = cannyEdgeDetection(img)

# Display the images
cv2.imshow('Original', img)
cv2.imshow('Mask', masked)
cv2.imshow('Contrast stretched', contrast_stretched)
cv2.imshow('Equalized', equalized)
cv2.imshow('Edge detection', edgeDetection)

# Save the processed images
cv2.imwrite('mehu_masked.png', masked)
cv2.imwrite('mehu_contrast_stretched.png', contrast_stretched)
cv2.imwrite('mehu_equalized.png', equalized)
cv2.imwrite('mehu_canny.png', edgeDetection)


print(img.shape, masked.shape)

cv2.waitKey(0)
