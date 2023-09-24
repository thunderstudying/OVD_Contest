import os
from PIL import Image
import json

train = json.load(open('datasets/360/train.json'))
cid2imgid = {x+1: [] for x in range(len(train['categories']))}
for x in train['annotations']:
    cid2imgid[x['category_id']].append(x['image_id'])

imgid2imgpath = {}
for x in train['images']:
    # if imgid2imgpath[x['id']] in imgid2imgpath.keys():
    #     imgid2imgpath[x['id']].append(x['file_name'])
    # else:
    #     imgid2imgpath[x['id']] = []
    #     imgid2imgpath[x['id']].append(x['file_name'])
    imgid2imgpath[x['id']] = x['file_name']

for ind, imgid in cid2imgid.items():
    root = f'data/train_sort_by_id/{ind}'
    if not os.path.exists(root):
        os.makedirs(root)

    for i in imgid:
        img = imgid2imgpath[i]
        ori = os.path.join('data/train', img)
        image = Image.open(open(ori, 'rb'))
        try:
            image.save(os.path.join(root, img), 'JPEG')
        except:
            image = image.convert("RGB")
            image.save(os.path.join(root, img), 'JPEG')
