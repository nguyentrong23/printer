import cv2
import numpy as np
import json
import os
# gen mask from anotation file
os.makedirs('mask', exist_ok=True)

anotation_path = '_annotations.coco.json'

# read file with annotations
data = None
with open(anotation_path) as f:
    data = json.load(f)
anotations = data['annotations']
for i in range(1, 3):
    for imageinfo in data['images']:
        image_name = imageinfo['file_name']
        image_id = imageinfo['id']
        image = cv2.imread(image_name) 
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for anotation in anotations:
            if anotation['image_id'] == image_id and anotation['category_id'] == i: #1, 2
                mask = cv2.fillPoly(mask, [np.array(anotation['segmentation']).reshape(-1, 1, 2).astype(np.int32)], 255)
        mask_name = os.path.join('mask', image_name[:-4] + f'_{i}_mask.png')
        cv2.imwrite(os.path.join('mask', image_name), image)
        cv2.imwrite(mask_name, mask)
