import json
from pathlib import Path
from argparse import ArgumentParser
import cv2
from tqdm import tqdm

def find_image_path(ID, img_dir):
    """Find the path of the image in the specified directory."""
    jpg_path = img_dir / f'{ID}.jpg'
    if jpg_path.exists():
        return jpg_path
    raise FileNotFoundError(f'Image {ID}.jpg not found in {img_dir}.')

def image_shape(jpg_path):
    """Get the shape of the image."""
    img = cv2.imread(jpg_path.as_posix())
    return img.shape

def txt_line(cls, bbox, img_w, img_h):
    """Generate 1 line in the txt file, normalized by the image size."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f'{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n'

def process(annotation_filename, img_dir):
    """Process annotations and save txt files in a corresponding labels directory."""
    img_dir = Path(img_dir)
    labels_dir = img_dir.parent / "labels"
    labels_dir.mkdir(exist_ok=True)

    with open(annotation_filename, 'r') as fanno:
        for line in tqdm(fanno, desc=f"Processing {annotation_filename}"):
            anno = json.loads(line)
            ID = anno['ID']
            jpg_path = find_image_path(ID, img_dir)
            img_h, img_w, _ = image_shape(jpg_path)
            txt_path = labels_dir / f'{ID}.txt'

            with open(txt_path, 'w') as ftxt:
                for obj in anno['gtboxes']:
                    if obj['tag'] == 'person' and 'vbox' in obj:  # Focus on visible body only
                        ftxt.write(txt_line(0, obj['vbox'], img_w, img_h))

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_annotation', default='raw/annotation_train.odgt', help='Path to training annotation file.')
    parser.add_argument('--val_annotation', default='raw/annotation_val.odgt', help='Path to validation annotation file.')
    parser.add_argument('--train_image_dir', default='raw/CrowdHuman_train/Images', help='Directory where train images are stored.')
    parser.add_argument('--val_image_dir', default='raw/CrowdHuman_val/Images', help='Directory where val images are stored.')
    args = parser.parse_args()

    process(args.train_annotation, args.train_image_dir)
    process(args.val_annotation, args.val_image_dir)

if __name__ == '__main__':
    main()
