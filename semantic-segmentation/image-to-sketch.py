import cv2
import glob
import os
from tqdm import tqdm

def convert_to_sketch(image_path):
    # Reading the image
    image = cv2.imread(image_path)

    # Converting it into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    gray_image = cv2.equalizeHist(gray_image)

    # Inverting the image
    inverted_image = 255 - gray_image

    # The pencil sketch
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    # Overwrite the original image with the sketched version
    cv2.imwrite(image_path, pencil_sketch)

def process_directory(directory_path):
    # Get all jpg files in the directory
    image_files = glob.glob(os.path.join(directory_path, '*.jpg'))

    for image_file in tqdm(image_files):
        convert_to_sketch(image_file)

# Path to the directory containing jpg images
directory_path = "data/ade20k_rooms_only/images/training"
process_directory(directory_path)

directory_path = "data/ade20k_rooms_only/images/validation"
process_directory(directory_path)