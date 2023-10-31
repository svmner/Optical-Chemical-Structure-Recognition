import cv2
import numpy as np

def resize_image(image, size=(128, 128)):
    """Resizes the image to a fixed size."""
    return cv2.resize(image, size)

def binarize_image(image, threshold=127):
    """Converts the image to a binary (black and white) image using a threshold."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def remove_noise(image, kernel_size=(3, 3)):
    """Applies morphological operations to remove small noise in the image."""
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def preprocess_image(image):
    """Applies a sequence of preprocessing steps to the image."""
    image = resize_image(image)
    image = binarize_image(image)
    image = remove_noise(image)
    return image
