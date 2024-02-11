import numpy as np
import matplotlib.pyplot as plt
import cv2


def applyMeanBlur(image, kernel_size=3):
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2

    # Apply the kernel to the image
    return cv2.filter2D(image, -1, kernel)

def convolve_even(image, kernel):
    h, w = image.shape
    ksize_y, ksize_x = kernel.shape
    padding_y, padding_x = ksize_y // 2, ksize_x // 2

    result = np.zeros_like(image, dtype=float)

    for i in range(padding_y, h - padding_y):
        for j in range(padding_x, w - padding_x):
            result[i, j] = np.sum(image[i - padding_y:i + padding_y + 1, j - padding_x:j + padding_x + 1] * kernel)

    return result






def convolve_odd(image, kernel):
    # Get the dimensions of the image and kernel
    img_height, img_width = image.shape
    kernel_size = len(kernel)
    padding = kernel_size // 2

    # Initialize the result image
    result = np.zeros_like(image, dtype=float)

    # Convolution operation
    for i in range(padding, img_height - padding):
        for j in range(padding, img_width - padding):
            result[i, j] = np.sum(image[i - padding:i + padding + 1, j - padding:j + padding + 1] * kernel)

    return result

def sobel_filter(image):
    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -2, -1], 
                         [ 0,  0,  0], 
                         [ 1,  2,  1]])

    # Apply Sobel filtering
    gradient_x = convolve_odd(image, kernel_x)
    gradient_y = convolve_odd(image, kernel_y)

    # Compute the magnitude of the gradient
    magnitude = np.clip(np.sqrt(gradient_x**2 + gradient_y**2),0,255).astype(np.uint8)

    return magnitude

def prewitt_filter(image):


    # Define Sobel kernels
    kernel_y = np.array([[-1, 0, 1], 
                         [-1, 0, 1], 
                         [-1, 0, 1]])
    
    kernel_x = np.array([[-1, -1, -1], 
                         [ 0,  0,  0], 
                         [ 1,  1,  1]])

    # Apply Sobel filtering
    gradient_x = convolve_odd(image, kernel_x)
    gradient_y = convolve_odd(image, kernel_y)

    # Compute the magnitude of the gradient
    magnitude = np.clip(np.sqrt(gradient_x**2 + gradient_y**2),0,255).astype(np.uint8)

    return magnitude


def robert_filter(img):
    kernel_x = np.array([[1, 0], 
                         [0, -1]])
    
    kernel_y = np.array([[0, 1], 
                         [-1, 0]])

    gradient_x = convolve_even(img, kernel_x)
    gradient_y = convolve_even(img, kernel_y)

    magnitude = np.clip(np.sqrt(gradient_x**2 + gradient_y**2), 0, 255).astype(np.uint8)

    return magnitude


image = cv2.imread('images/cat.jpg', 0)

# blurred = applyMeanBlur(image, 9)


prewitt = prewitt_filter(image)
sobel = sobel_filter(image)
robert=robert_filter(image)

cv2.imshow('Original', image)
cv2.imshow('Prewitt', prewitt)
cv2.imshow('Robert', robert)
cv2.imshow('Sobel', sobel)


cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.figure(figsize=(10, 5))

# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')

# plt.subplot(2, 2, 2)
# plt.imshow(gradient_x, cmap='gray')
# plt.title('Sobel X')

# plt.subplot(2, 2, 3)
# plt.imshow(gradient_y, cmap='gray')
# plt.title('Sobel Y')

# plt.subplot(2, 2, 4)
# plt.imshow(magnitude, cmap='gray')
# plt.title('Magnitude')

# # plt.tight_layout()
# plt.show()
