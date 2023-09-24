import os

import json

import numpy as np
from PIL import Image

# 将下载的各类别图片放到各自目录中 并生成json类别标注文件
ori_root = 'data/ori_novel_image_multi_root'
new_root = 'data/ori_novel_image'
categories = json.load(open("datasets/360/test.json"))['categories']
images = []
img_num = 0
for id in os.listdir(ori_root):
    class_root = os.path.join(ori_root, id)
    for img in os.listdir(class_root):
        name = img.split('.')[0]
        img_path = os.path.join(class_root, img)
        image = Image.open(open(img_path, 'rb'))
        try:
            image.save(f"{new_root}/{id}_{name}.jpg", "JPEG")
        except:
            image = image.convert("RGB")
            image.save(f"{new_root}/{id}_{name}.jpg", "JPEG")
            # print(f'wrong {id}/{name}.jpg')
            # break
        image = np.asarray(image.convert('RGB'))
        h, w = image.shape[:2]
        img_num += 1
        image_info = {
            'id': img_num,
            'file_name': '{}/{}_{}.jpg'.format(new_root, id, name),
            'height': h,
            'width': w,
            # 'captions': [cap],
            'pos_category_ids': [int(id)],
        }
        images.append(image_info)

out = {'images': images, 'categories': categories,
       'annotations': []}
json.dump(out, open('datasets/360/8_novel_class.json', 'w'))

