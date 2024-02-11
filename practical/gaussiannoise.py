import cv2
import numpy as np

img = cv2.imread("images/cat.jpg", 0)

height, width = img.shape

mean = 0
std = 50

gaussian_noise = np.zeros(shape=(height, width), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        gaussian_noise[i, j] = img[i, j] + np.random.normal(mean, std)
        gaussian_noise[i, j] = np.clip(gaussian_noise[i, j], 0, 255)
        
        
cv2.imshow("Original", img)

cv2.imshow("Gaussian Noise", gaussian_noise)

cv2.waitKey(0)

cv2.destroyAllWindows()