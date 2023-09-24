import os

import json

import numpy as np
from PIL import Image

# root = 'data/filter_novel_image_multi_root'
# new_root = 'data/filter_novel_image'
root = 'data/filter_novel_base_image_multi_root'
new_root = 'data/filter_novel_base_image'
categories = json.load(open("datasets/360/test.json"))['categories']
images = []
img_num = 0
for id in os.listdir(root):
    class_root = os.path.join(root, id)
    for img in os.listdir(class_root):
        name = img.split('.')[0]
        img_path = os.path.join(class_root, img)
        image = Image.open(open(img_path, 'rb'))
        try:
            image.save(f"{new_root}/{id}_{name}.jpg", "JPEG")
        except:
            print(f'wrong {id}/{name}.jpg')
            image = image.convert("RGB")
            image.save(f"{new_root}/{id}_{name}.jpg", 'JPEG')
        image = np.asarray(image.convert('RGB'))
        h, w = image.shape[:2]
        img_num += 1
        image_info = {
            'id': img_num,
            # 'file_name': '{}/{}_{}.jpg'.format(new_root, id, name),
            'file_name': '{}_{}.jpg'.format(id, name),  # 0723
            'height': h,
            'width': w,
            # 'captions': [cap],
            'pos_category_ids': [int(id)],
        }
        images.append(image_info)

out = {'images': images, 'categories': categories,
       'annotations': []}
# json.dump(out, open('datasets/360/9_filter_novel_class_0723.json', 'w'))
json.dump(out, open('datasets/360/9_filter_novel_base_class_0724.json', 'w'))

