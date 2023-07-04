CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/configs_ihl/COCO_IHL.py \
    work_dirs/COCO_IHL/epoch_210.pth \
    --out test_dirs/COCO_IHL.json

