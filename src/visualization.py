import numpy as np
import cv2
import os


def draw_molecular_structure(graph, input_image_name, output_dir="reconstructed_molecules"):
    """
    Draw the molecular structure based on the molecular graph.

    Args:
    - graph: The molecular graph constructed earlier.
    - output_path (str): Path to save the reconstructed image.
    """
    # Create a blank canvas
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Loop through the nodes in the graph and draw atoms
    for node in graph.nodes(data=True):
        coord = node[1]['coord']
        atom_type = node[1]['atom_type']
        cv2.circle(canvas, coord, 20, (0, 255, 0), -1)
        cv2.putText(canvas, atom_type, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Loop through the edges in the graph and draw bonds
    for edge in graph.edges(data=True):
        start_coord = graph.nodes[edge[0]]['coord']
        end_coord = graph.nodes[edge[1]]['coord']
        bond_type = edge[2]['bond_type']

        if bond_type == "single":
            cv2.line(canvas, start_coord, end_coord, (0, 0, 255), 2)
        # Add other bond types as needed with different drawing logic

    output_filename = "reconstructed_" + input_image_name
    output_path = os.path.join(output_dir, output_filename)

    # Save the reconstructed image
    cv2.imwrite(output_path, canvas)

    return output_path