import os
import cv2
from data_loader import load_molecular_images
from preprocessing import preprocess_image
from text_recognition import load_templates, recognize_symbols, remove_text_using_inpainting
from feature_extraction import detect_atoms, detect_bonds, recognize_atoms, construct_molecular_graph
from visualization import draw_molecular_structure
from evaluation import compare_images  # Import the evaluation function


def main():
    # Directory paths (adapt these paths as needed)
    ORIGINAL_DIR = r"C:\Users\91636\Documents\Sem5\CV\Project\data"
    RECONSTRUCTED_DIR = r"C:\Users\91636\Documents\Sem5\CV\Project\reconstructed_molecules"
    TRAIN_PATH = r"C:\Users\91636\Documents\Sem5\CV\Project\train"  # Path to the training images for templates

    # Load text recognition templates
    templates = load_templates(TRAIN_PATH)

    # Ensure the reconstructed directory exists
    if not os.path.exists(RECONSTRUCTED_DIR):
        os.makedirs(RECONSTRUCTED_DIR)

    # Load molecular images
    images = load_molecular_images(ORIGINAL_DIR)

    for image_name, image in images.items():
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Recognize symbols/text in the image
        recognized_texts = recognize_symbols(preprocessed_image, templates)

        # Create a mask for inpainting based on recognized text regions
        mask = cv2.zeros_like(preprocessed_image)
        for _, (x1, y1, x2, y2), _ in recognized_texts:
            mask[y1:y2, x1:x2] = 255

        # Remove recognized textual elements using inpainting
        image_without_text = remove_text_using_inpainting(preprocessed_image, mask)

        # Detect atoms and bonds
        atoms_coords = detect_atoms(image_without_text)
        bonds_coords = detect_bonds(image_without_text)

        # Recognize atom types
        recognized_atoms = recognize_atoms(image_without_text, atom_detection_params={}, hog_params={})

        # Construct molecular graph
        graph = construct_molecular_graph(recognized_atoms, bonds_coords)

        # Draw the molecular structure
        reconstructed_image = draw_molecular_structure(graph, image_name, output_dir=RECONSTRUCTED_DIR)

        # Evaluate the reconstructed image against the original using SSIM and MSE
        ssim_value, mse_value = compare_images(image, reconstructed_image)
        print(f"SSIM for {image_name}: {ssim_value}")
        print(f"MSE for {image_name}: {mse_value}")


if __name__ == "__main__":
    main()
