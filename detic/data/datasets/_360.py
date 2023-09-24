# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from lvis import LVIS

logger = logging.getLogger(__name__)

_CUSTOM_SPLITS_360 = {
    "360_final": ("data_final_contest/train/", "datasets/360/train_final.json"),
    "360_novel": ("data/filter_novel_base_image/", "datasets/360/9_filter_novel_base_class_0724.json"),  # I_F_A
    '360_pl': ("data_final_contest/test/", "datasets/360/pl_final.json")
}


def custom_load_360_json(json_file, image_root):
    """changed from lvis_v1.py"""
    json_file = PathManager.get_local_path(json_file)
    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    catid2contid = {x['id']: i for i, x in enumerate(
        sorted(lvis_api.dataset['categories'], key=lambda x: x['id']))}
    for x in lvis_api.dataset['categories']:
        assert catid2contid[x['id']] == x['id'] - 1

    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), \
        "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in the LVIS v1 format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                file_name = file_name[-16:]
            record["file_name"] = os.path.join(image_root, file_name)
        elif 'coco_url' in img_dict:
            # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
            file_name = img_dict["coco_url"][30:]
            record["file_name"] = os.path.join(image_root, file_name)
        elif 'tar_index' in img_dict:
            record['tar_index'] = img_dict['tar_index']

        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        # NOTE: modified by Xingyi: convert to 0-based
        record["neg_category_ids"] = [
            catid2contid[x] for x in record["neg_category_ids"]]
        if 'pos_category_ids' in img_dict:
            record['pos_category_ids'] = [
                catid2contid[x] for x in img_dict.get("pos_category_ids", [])]
        if 'captions' in img_dict:
            record['captions'] = img_dict['captions']
        if 'caption_features' in img_dict:
            record['caption_features'] = img_dict['caption_features']
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = catid2contid[anno['category_id']]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def custom_register_360_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_360_json(
    # DatasetCatalog.register(name, lambda: load_360_json(
        json_file, image_root))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="coco", **metadata
    )


def get_360_instances_meta():
    cat_info = json.load(open('datasets/360/3_train_cat_info.json'))
    thing_classes = [x['en_name'] for x in cat_info]
    category_image_count = [{x['id']: x['image_count'] for x in cat_info}]
    meta = {"thing_classes": thing_classes, "class_image_count": category_image_count}
    return meta


for key, (image_root, json_file) in _CUSTOM_SPLITS_360.items():
    custom_register_360_instances(
        key,
        get_360_instances_meta(),
        # os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        # os.path.join("datasets", image_root),
        json_file,
        image_root
    )
