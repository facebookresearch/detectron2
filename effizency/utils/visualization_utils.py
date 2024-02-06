import numpy as np
import cv2
from matplotlib import patches, pyplot as plt

from effizency.config import CONFIG


def build_final_mask_from_predictions(masks, bboxes, final_shape=(640, 640)):
    """
    Create a final mask with individual instance masks placed according to bounding boxes.

    :param masks: NumPy array of shape (n, 1, 28, 28) containing n masks
    :param bboxes: NumPy array of shape (n, 4) containing n bounding boxes in the format (x1, y1, x2, y2) with float coordinates
    :param final_shape: Tuple defining the shape of the final mask
    :return: Final mask with individual masks overlaid
    """
    # Initialize the final mask with zeros
    final_mask = np.zeros(final_shape, dtype=np.uint8)

    # Calculate the scale factors for x and y dimensions
    scale_x = final_shape[1] / CONFIG['image_size'][1]
    scale_y = final_shape[0] / CONFIG['image_size'][0]

    for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
        # Scale and convert bounding box coordinates from float to int
        x1, y1, x2, y2 = map(int, [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y])

        # Adjust bbox dimensions to ensure it's within the final mask bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, final_shape[1]), min(y2, final_shape[0])

        # Handle cases where the bbox may have zero or negative dimensions after conversion
        if x2 <= x1 or y2 <= y1:
            continue

        # Resize mask to match the bounding box size
        mask_resized = cv2.resize(mask[0], (x2 - x1, y2 - y1))
        mask_resized = (mask_resized > CONFIG['mask_threshold']).astype(np.uint8)

        # Assign an integer i to the mask for instance_i
        final_mask[y1:y2, x1:x2][mask_resized == 1] = i + 1  # i+1 to make instance 0 distinguishable

    return final_mask


def polygons_to_mask(polygon, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    # polygons is a list of polygons, where each polygon is a list of points (x, y)
    # Convert the polygon points into an integer array
    int_polygon = np.array(polygon, dtype=np.int32).reshape((1, -1, 2))
    # Fill the polygon with ones
    cv2.fillPoly(mask, int_polygon, int(1))
    return mask


def draw_image_with_boxes(image, bounding_boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        confidence, label = bbox.confidence, bbox.label

        # Convert relative coordinates to absolute coordinates
        abs_x = x * image.shape[1]
        abs_y = y * image.shape[0]
        abs_width = width * image.shape[1]
        abs_height = height * image.shape[0]

        # Create a rectangle patch
        rect = patches.Rectangle(
            (abs_x - abs_width / 2, abs_y - abs_height / 2),
            abs_width, abs_height,
            linewidth=2, edgecolor='r', facecolor='none'
        )

        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Display label and confidence
        ax.text(abs_x - abs_width / 2, abs_y - abs_height / 2 - 5,
                f"{label} {confidence:.2f}", color='r', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Show the image with bounding boxes
    plt.show()


def draw_image_with_masks(image, mask, labels):
    """
    Draw each mask on the image with a different color and display the image.

    :param image: (numpy.ndarray) The original image.
    :param mask: (numpy.ndarray) .
    :param labels: (list) List of labels corresponding to each mask.
    """
    # Create a color map for each label
    colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

    # Make a copy of the image to draw on
    overlayed_image = image.copy()

    # Iterate through the unique values in the mask (excluding 0)
    for i, label in enumerate(labels):
        # Extract binary mask for the current instance
        instance_mask = (mask == (i + 1))

        # Find contours of the instance mask
        contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the image
        cv2.drawContours(overlayed_image, contours, -1, colors[i][:3] * 255, thickness=2)

        # Optionally, you can fill the instance mask with a translucent color
        overlayed_image[instance_mask] = (
                    overlayed_image[instance_mask] * (1 - 0.5) + np.array(colors[i][:3] * 255, dtype=np.uint8) * 0.5)

    # Display the result using matplotlib
    fig, ax = plt.subplots(1)
    ax.imshow(overlayed_image)
    plt.axis('off')  # Hide the axis
    plt.show()
