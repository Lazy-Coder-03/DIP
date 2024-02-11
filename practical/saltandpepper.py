import cv2
import numpy as np

img = cv2.imread("images/cat.jpg", 0)

schance = 0.1
pchance = 0.1

height, width = img.shape

noisy = np.copy(img)  # Initialize the noisy image with the original image

for i in range(height):
    for j in range(width):
        sr = np.random.random()
        pr = np.random.random()
        if sr < schance:
            noisy[i, j] = 255
        elif pr < pchance:
            noisy[i, j] = 0

cv2.imshow("Original", img)
cv2.imshow("Noisy", noisy)

cv2.waitKey(0)
cv2.destroyAllWindows()
