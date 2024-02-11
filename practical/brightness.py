import cv2
import numpy as np

img = cv2.imread("images/cat.jpg", 0)

height, width = img.shape

def enhanceBrightness(img, alpha):
    new_img = np.clip(alpha * img, 0, 255).astype(np.uint8)
    return new_img

def enhanceContrast(img, beta):
    new_img = np.clip(img + beta, 0, 255).astype(np.uint8)
    return new_img



def enhance(img, alpha, beta):
    new_img = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            new_img[i, j] = np.clip(alpha * img[i, j] + beta, 0, 255)
    return new_img


cv2.imshow("Original", img)

cv2.imshow("Enhanced", enhance(img, 1.2, -50))

cv2.imshow("Enhanced Brightness", enhanceBrightness(img, 1.2))

cv2.imshow("Enhanced Contrast", enhanceContrast(img, -50))



cv2.waitKey(0)
cv2.destroyAllWindows()