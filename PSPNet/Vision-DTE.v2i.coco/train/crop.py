import cv2
import numpy as np
import os

data_path = 'data'  

image_path = 'proccesed_mask'
os.makedirs(data_path, exist_ok=True)

# resolution 1024x768

image_names = [f for f in os.listdir(image_path) if f.endswith('.jpg')]

for image_name in image_names:
    mask_name = image_name[:-4] + '_mask.png'
    image = cv2.imread(os.path.join(image_path, image_name))
    mask = cv2.imread(os.path.join(image_path, mask_name), cv2.IMREAD_GRAYSCALE)
    # random crop with resolution 1024x768
    for i in range(10):
        x = np.random.randint(0, 1023)
        y = np.random.randint(0, 766)
        crop_image = image[y:y+768, x:x+1024]
        crop_mask = mask[y:y+768, x:x+1024]
        cv2.imwrite(os.path.join(data_path, image_name[:-4] + f'_{i}.jpg'), crop_image)
        cv2.imwrite(os.path.join(data_path, mask_name[:-9] + f'_{i}_mask.jpg'), crop_mask)

