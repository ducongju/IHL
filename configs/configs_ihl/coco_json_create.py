# Copyright (c) DuCongju. All rights reserved.

import json

def filter_annotations(input_file, output_file, detection_file, detection_output_file):
    # 处理person_keypoints_val2017.json
    with open(input_file, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']

    # 获取图像中所有注释的个数
    image_annotation_count = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id in image_annotation_count:
            image_annotation_count[image_id] += 1
        else:
            image_annotation_count[image_id] = 1

    # 筛选包含17个可见关键点的人体注释，并检查对应图像中的其他注释
    filtered_annotations = []
    filtered_image_ids = set()

    for annotation in annotations:
        image_id = annotation['image_id']
        keypoints = annotation['keypoints']
        num_keypoints = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0)

        if num_keypoints >= 13 and num_keypoints <= 17:  # 此处修改可见关键点个数
            if image_annotation_count[image_id] == 1:
                filtered_annotations.append(annotation)
                filtered_image_ids.add(image_id)
            else:
                # 检查对应图像中的其他注释是否都具有17个可见关键点
                image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

                # 判断图像中的其他注释是否都具有17个可见关键点
                all_visible = True
                for ann in image_annotations:
                    ann_keypoints = ann['keypoints']
                    ann_num_keypoints = sum(1 for i in range(0, len(ann_keypoints), 3) if ann_keypoints[i+2] > 0)
                    if ann_num_keypoints < 13 or ann_num_keypoints > 17:  # 此处修改可见关键点个数，与上面互补
                        all_visible = False
                        break

                if all_visible:
                    filtered_annotations.append(annotation)
                    filtered_image_ids.add(image_id)

    # 更新注释列表
    data['annotations'] = filtered_annotations

    # 删除对应的图像信息
    filtered_images = [image for image in images if image['id'] in filtered_image_ids]
    data['images'] = filtered_images

    # 修改info信息
    data['info']['description'] = 'Filtered COCO annotations'

    with open(output_file, 'w') as f:
        json.dump(data, f)

    # 处理COCO_val2017_detections_AP_H_56_person.json
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)

    detections = detection_data

    # 删除包含不可见关键点的人体注释
    filtered_detections = [detection for detection in detections if detection['image_id'] in filtered_image_ids]
    detection_data = filtered_detections

    with open(detection_output_file, 'w') as f:
        json.dump(detection_data, f)

# 使用示例
input_file = 'data/coco/annotations/person_keypoints_val2017.json'  # 输入的注释文件路径
output_file = 'data/coco/annotations/visible_13to17.json'  # 输出的筛选后的注释文件路径
detection_file = 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'  # 输入的检测注释文件路径
detection_output_file = 'data/coco/person_detection_results/detection_visible_13to17.json'  # 输出的筛选后的检测注释文件路径
filter_annotations(input_file, output_file, detection_file, detection_output_file)
