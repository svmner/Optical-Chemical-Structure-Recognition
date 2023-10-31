import cv2
import os
def load_molecular_images(dataset_path):
    data = []
    labels = []

    # List all structure directories
    struct_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for struct_dir in struct_dirs:
        struct_path = os.path.join(dataset_path, struct_dir)
        for image_file in os.listdir(struct_path):
            # Ensure only image files are processed
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(struct_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                data.append(image)
                labels.append(struct_dir)

    return data, labels


def load_bond_data(bond_train_path):
    bond_data = []
    bond_labels = []

    # List all bond type directories
    bond_dirs = [d for d in os.listdir(bond_train_path) if os.path.isdir(os.path.join(bond_train_path, d))]

    for bond_dir in bond_dirs:
        bond_path = os.path.join(bond_train_path, bond_dir)
        for image_file in os.listdir(bond_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(bond_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                bond_data.append(image)
                bond_labels.append(bond_dir)

    return bond_data, bond_labels
