'''
Usage: python prepare_data/verify_txt.py --image_dir raw/CrowdHuman_train/images/ --label_dir raw/CrowdHuman_train/labels/
or
Usage: python prepare_data/verify_txt.py --image_dir raw/CrowdHuman_val/images/ --label_dir raw/CrowdHuman_val/labels/


'''

import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

def draw_boxes(image_path, label_path, output_dir):
    """
    Draw bounding boxes on the image based on the label file and save to output_dir.
    """
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Open the label file and draw each bounding box
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # YOLO format: class x_center y_center width height
            x_center, y_center, box_w, box_h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert to pixel values
            x1, y1, x2, y2 = int((x_center - box_w / 2) * w), int((y_center - box_h / 2) * h), int((x_center + box_w / 2) * w), int((y_center + box_h / 2) * h)
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with drawn boxes to output_dir
    output_path = output_dir / Path(image_path).name
    cv2.imwrite(str(output_path), image)

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images based on YOLO format labels and save them.")
    parser.add_argument("--image_dir", required=True, help="Directory containing the images.")
    parser.add_argument("--label_dir", required=True, help="Directory containing the label files in YOLO format.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = image_dir.parent / "masked_images"
    output_dir.mkdir(exist_ok=True)

    for label_path in tqdm(list(label_dir.glob("*.txt")), desc="Processing images"):
        # Assuming image and label files have the same basename
        image_path = image_dir / (label_path.stem + ".jpg")
        if image_path.exists():
            draw_boxes(str(image_path), str(label_path), output_dir)
        else:
            print(f"Image for {label_path.name} not found.")

if __name__ == "__main__":
    main()
