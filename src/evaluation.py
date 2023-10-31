from skimage.metrics import structural_similarity as ssim
import cv2


def compare_images(imageA, imageB):
    """
    Compute the Structural Similarity Index (SSIM) and Mean Squared Error (MSE) between two images.

    Args:
    - imageA: First image (numpy array).
    - imageB: Second image (numpy array).

    Returns:
    - ssim_value: SSIM between imageA and imageB.
    - mse_value: MSE between imageA and imageB.
    """
    # Compute SSIM between two images
    ssim_value = ssim(imageA, imageB, multichannel=True)

    # Compute MSE between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    mse_value = err

    return ssim_value, mse_value
