import cv2
import numpy as np
from sklearn.svm import LinearSVC,SVC
import joblib
import networkx

# Load the trained LinearSVC model
svc_classifier = joblib.load("atom_classifier.pkl")
def extract_hog_features_opencv(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Extract HOG features from an image using OpenCV.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    hog = cv2.HOGDescriptor(_winSize=(gray_image.shape[1] // cell_size[1] * cell_size[1],
                                      gray_image.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    features = hog.compute(gray_image)

    return features.flatten()
def detect_atoms(image, dp=1, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0):
    """
    Detect atoms (represented as circles) in a molecular diagram.

    Args:
    - image (np.array): The input image containing molecular structures.
    - dp (float): Inverse ratio of the accumulator resolution to the image resolution.
    - min_dist (int): Minimum distance between the centers of the detected circles.
    - param1 (float): First method-specific parameter for the cv2.HoughCircles function.
    - param2 (float): Second method-specific parameter for the cv2.HoughCircles function.
    - min_radius (int): Minimum circle radius.
    - max_radius (int): Maximum circle radius.

    Returns:
    - circles (np.array): Array containing detected circles. Each circle is represented by (x, y, radius).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

    return circles


def detect_bonds(image, threshold=100, min_line_length=50, max_line_gap=10):
    """
    Detect bonds (represented as lines) in a molecular diagram.

    Args:
    - image (np.array): The input image containing molecular structures.
    - threshold (int): Accumulator threshold parameter for the cv2.HoughLinesP function.
                       Only lines that get enough votes (> threshold) are returned.
    - min_line_length (int): Minimum line length. Line segments shorter than this are rejected.
    - max_line_gap (int): Maximum allowed gap between line segments to treat them as a single line.

    Returns:
    - lines (np.array): Array containing detected line segments. Each line is represented by start and end points (x1, y1, x2, y2).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        lines = lines.reshape((-1, 4))

    return lines


def recognize_atoms(image, atom_detection_params={}, hog_params={}):
    """
    Recognize atom types in a molecular diagram using HOG features and a trained classifier.
    """
    # Detect potential atom locations
    circles = detect_atoms(image, **atom_detection_params)

    atom_labels = []
    for circle in circles:
        x, y, radius = circle

        # Extract a small region around the atom center
        roi = image[y - radius:y + radius, x - radius:x + radius]

        # Extract HOG features from the ROI
        features = extract_hog_features_opencv(roi, **hog_params)

        # Predict atom type using the trained classifier
        # Assuming you've loaded your trained LinearSVC model as "svc_classifier"
        predicted_label = svc_classifier.predict([features])

        # If a label is predicted, associate it with the atom
        if predicted_label:
            atom_labels.append(((x, y), predicted_label[0]))

    return atom_labels


def construct_molecular_graph(atoms, bonds):
    """
    Construct a molecular graph based on detected atoms and bonds.

    Args:
    - atoms (list): List of tuples containing atom center coordinates and atom type.
    - bonds (list): List of tuples containing start and end points of lines and bond type.

    Returns:
    - G (networkx.Graph): Molecular graph with nodes as atoms and edges as bonds.
    """

    # Initialize an empty graph
    G = nx.Graph()

    # Add nodes (atoms) to the graph
    for coord, atom_type in atoms:
        G.add_node(coord, atom_type=atom_type)

    # Add edges (bonds) to the graph
    for (start, end), bond_type in bonds:
        G.add_edge(start, end, bond_type=bond_type)

    return G
