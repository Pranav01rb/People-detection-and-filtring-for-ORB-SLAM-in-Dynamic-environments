from yolov5 import *
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Paths
model_path = "/Users/pranavrbidare/Desktop/perception_proj/pranav/best.pt"
input_folder = "/Users/pranavrbidare/Desktop/perception_proj/pranav/rgbd_dataset_freiburg2_desk_with_person/rgb"
output_folder = "/Users/pranavrbidare/Desktop/perception_proj/pranav/masks"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the YOLO model
model = YOLO(model_path)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        save_path = os.path.join(
            output_folder, os.path.splitext(filename)[0] + '_masked.png')

        # Load the image
        img = cv2.imread(image_path)
        H, W, _ = img.shape

        # Perform inference
        print(filename)
        results = model(img)
        # Iterate over the results
        for result in results:
            if result.masks:
                for j, mask in enumerate(result.masks.data):
                    # Convert mask to a usable format
                    mask = mask.numpy() * 255
                    mask = cv2.resize(mask, (W, H))

                    # Convert mask to binary (0 or 255)
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                    # Invert the mask: white to black, and vice versa
                    inverted_mask = 255 - binary_mask

                    # Overlay the black parts from the inverted mask onto the original image
                    # Set the color of the blob as black
                    img[inverted_mask == 0] = [0, 0, 0]
            else:
                cv2.imwrite(save_path, img)
        # Save the result
        cv2.imwrite(save_path, img)