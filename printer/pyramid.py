import cv2
import numpy as np
import os

data_path = 't'
image_path = 't'
os.makedirs(data_path, exist_ok=True)
image_names = [f for f in os.listdir(image_path) if (f.endswith('.jpg') and  not f.endswith('mask.jpg') )]

with open(os.path.join(data_path, 'image_paths.txt'), 'w') as file:
    for image_name in image_names:
        absolute_path = os.path.abspath(os.path.join(image_path, image_name))
        absolute_path_no_ext = os.path.splitext(absolute_path)[0]
        print(f"{absolute_path} {absolute_path_no_ext}_mask.png\n")
        file.write(f"{absolute_path} {absolute_path_no_ext}_mask.png\n")