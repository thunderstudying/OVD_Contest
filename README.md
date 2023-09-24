# [Open Vocabulary Detection Contest - 开放世界目标检测竞赛 2023 第六名方案](https://360cvgroup.github.io/OVD_Contest/) 

## 本方案基于[Detic](https://github.com/facebookresearch/Detic)

## 数据
使用的数据为，竞赛提供的检测数据集，以及只有类别标注的弱监督数据集

弱监督数据集的搜集过程为，使用爬虫程序，在京东商城上用商品中文搜索，下载第一页搜索结果中的图像并过滤

弱监督数据集总共含有13,428张图片

## 方案
使用GIoU Loss（g）、SIoU Loss（s），指数滑动平均EMA（E），多尺度测试（T），伪标注微调（PLFT）等技术，[详细方案介绍，提取码ra23](https://www.aliyundrive.com/s/XdHviQYybqB)

## 性能
|       | Novel  | Base   | All    |
| --------- | ------ | ------ | ------ |
| IF_E_g    | 39.894 | 41.564 | 40.729 |
| IF_E_s    | 42.022 | 41.939 | 41.980 |
| IF_E_s_T  | 42.842 | 43.721 | 43.282 |
| IF_E_s_T_PLFT | 43.317 | 45.593 | 44.455 |

## 训练及测试
IF_+_E_s_T的训练和推理命令分别为：
```python
# 训练
python train_net.py --num-gpus 1 --config-file configs/360.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth OUTPUT_DIR final/ IF_+_E_s_T

# 推理
python demo.py --config-file configs/360.yaml --input data_final_contest/test/* --output output/test/ --vocabulary custom --custom_vocabulary headphone --confidence-threshold 0.0001 --pred_all_class --opts MODEL.WEIGHTS final/IF_+_E_s_T/model_0136444.pth
```

IF_+_E_s_T_PLFT的训练和推理命令分别为：
```python
# 训练
python train_net.py --num-gpus 1 --config-file configs/360_plft.yaml MODEL.WEIGHTS final/IF_+_E_s_T/model_0136444.pth OUTPUT_DIR final/IF_+_E_s_T_PLFT

# 推理
python demo.py --config-file configs/360_plft.yaml --input data_final_contest/test/* --output output/test/ --vocabulary custom --custom_vocabulary headphone --confidence-threshold 0.0001 --pred_all_class --opts MODEL.WEIGHTS final/IF_+_E_s_T_PLFT/model_0008492.pth
```