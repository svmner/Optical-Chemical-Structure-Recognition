# Scale-Invariant Template Matching
import cv2
import numpy as np

def gaussian_blur(image, kernel_size):
    """Apply Gaussian filter to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sliding_window(image, stepSize, windowSize):
    """Slide a window across the image."""
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def non_max_suppression(boxes, threshold=0.3):
    """Remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick].astype("int")

# Note: This is just the foundational code for scale-invariant template matching.
# Additional steps and fine-tuning might be needed based on the actual dataset and results.

# Let's start by writing a function to load the templates from the train subfolders
def load_templates(train_path):
    """
    Load template images from the train directory.

    Args:
    - train_path (str): Path to the train directory containing subdirectories for each symbol.

    Returns:
    - templates (dict): A dictionary where keys are the symbol names and values are the corresponding template images.
    """
    templates = {}

    # List all symbol directories in the train path
    symbol_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    for symbol_dir in symbol_dirs:
        symbol_path = os.path.join(train_path, symbol_dir)
        for image_file in os.listdir(symbol_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(symbol_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # For simplicity, we'll use only the first image in each directory as the template
                templates[symbol_dir] = image
                break

    return templates


def recognize_symbols(image, templates, method=cv2.TM_CCOEFF_NORMED, threshold=0.7):
    """
    Recognize symbols in an image based on template matching.

    Args:
    - image (np.array): The input image in which symbols are to be recognized.
    - templates (dict): Dictionary containing symbol templates.
    - method (int): Template matching method. Default is cv2.TM_CCOEFF_NORMED.
    - threshold (float): Threshold for considering a match.

    Returns:
    - recognized_symbols (list): List of tuples containing recognized symbol, bounding box, and match score.
    """
    recognized_symbols = []

    for symbol, template in templates.items():
        # Match the template with the image
        result = cv2.matchTemplate(image, template, method)

        # Find locations where the match score is above the threshold
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            recognized_symbols.append(
                (symbol, (pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]), result[pt[1], pt[0]]))

    # Apply non-max suppression to remove overlapping bounding boxes
    if recognized_symbols:
        boxes = np.array([item[1] for item in recognized_symbols])
        scores = np.array([item[2] for item in recognized_symbols])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=threshold, nms_threshold=0.4)

        recognized_symbols = [recognized_symbols[i[0]] for i in indices]

    return recognized_symbols


def remove_text_using_inpainting(image, mask, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA):
    """
    Remove text from an image using inpainting.

    Args:
    - image (np.array): The input image from which text needs to be removed.
    - mask (np.array): Binary mask where text regions are white (255) and other areas are black (0).
    - inpaint_radius (int): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    - inpaint_method (int): Inpainting method, either cv2.INPAINT_TELEA or cv2.INPAINT_NS.

    Returns:
    - inpainted_image (np.array): Image with text regions inpainted.
    """
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=inpaint_method)
    return inpainted_image
