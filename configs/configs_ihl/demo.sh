CUDA_VISIBLE_DEVICES=0 python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster-rcnn_x101-64x4d_fpn_1x_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth \
    configs/configs_dcj/COCO_IHL.py \
    work_dirs/COCO_IHL/epoch_210.pth \
    --input tests/data/coco/000000123480.jpg \
    --draw-heatmap \
    --output-root demo_dirs/COCO_IHL/