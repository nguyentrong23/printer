import cv2
import numpy as np
import os

path_to_mask = 'mask'
os.makedirs('proccesed_mask', exist_ok=True)

image_list = [f for f in os.listdir(path_to_mask) if f.endswith('.jpg')]

for image_name in image_list:
    image =cv2.imread(os.path.join(path_to_mask, image_name))
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask1_name = os.path.join(path_to_mask, image_name[:-4] + '_1_mask.png')
    mask_1 = cv2.imread(mask1_name, cv2.IMREAD_GRAYSCALE)
    mask2_name = os.path.join(path_to_mask, image_name[:-4] + '_2_mask.png')
    mask_2 = cv2.imread(mask2_name, cv2.IMREAD_GRAYSCALE)
    mask3_name = os.path.join(path_to_mask, image_name[:-4] + '_3_mask.png')
    mask_3 = cv2.imread(mask3_name, cv2.IMREAD_GRAYSCALE)
    mask = np.clip(mask_1 - mask_2 + mask_3, 0, 255)
    # save image and mask
    mask_name = os.path.join('proccesed_mask', image_name[:-4] + '_mask.png')
    cv2.imwrite(mask_name, mask)
    cv2.imwrite(os.path.join('proccesed_mask', image_name), image)
    