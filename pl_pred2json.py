import json

test_final = json.load(open('datasets/360/test_final.json'))
categories = test_final['categories']
images = test_final['images']

file = "360_upload_result/TTA_0829_1008/I_F_E_A_s_1008_30-3_TTA++.json"
ori_pred = json.load(open(file))
filtered_pred = [x for x in ori_pred if x['score'] >= 0.5]

annotations = [{'bbox': pred['bbox'],
                'image_id': pred['image_id'],
                'category_id': pred['category_id'],
                'id': annoid,
                'pred_score': pred['score']
                } for annoid, pred in enumerate(filtered_pred, 1)]
has_anno = set([x['image_id'] for x in annotations])
filter_images = [x for x in images if x['id'] in has_anno]
pl_json = {'images': filter_images, 'categories': categories, 'annotations': annotations}
json.dump(pl_json, open('datasets/360/pl_final.json', 'w'))
