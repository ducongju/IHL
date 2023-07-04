# Copyright (c) DuCongju. All rights reserved.

from pycocotools.coco import COCO
import cv2
import json
import os
import math
import numpy as np
import time

COCO_PERSON_SKELETON = [
    (0, 1), (1, 3), (3, 5),         # left head
    (0, 2), (2, 4), (4, 6),         # right head
    (0, 5), (5, 7), (7, 9),         # left arm
    (0, 6), (6, 8), (8, 10),        # right arm
    (5, 6),                         # l shoulder to r shoulder
    (12, 11),                       # r hip to l hip
    (5, 11), (11, 13), (13, 15),    # left side
    (6, 12), (12, 14), (14, 16),    # rught side
]

def plot_pose(img, person_list, bool_fast_plot=True):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
              [255, 0, 85], [255, 0, 170], [255, 0, 255],
              [0, 255, 0], [85, 255, 0], [170, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255],
              [0, 0, 255], [0, 85, 255], [0, 170, 255],
              [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 255, 255]]

    image = img.copy()
    limb_thickness = 2

    for limb_type in range(len(COCO_PERSON_SKELETON)):
        # hide the limb from nose to arm
        if limb_type == 6 or limb_type == 9:
            continue
        for person_joint_info in person_list:
            joint1 = person_joint_info[COCO_PERSON_SKELETON[limb_type][0]].astype(int)
            joint2 = person_joint_info[COCO_PERSON_SKELETON[limb_type][1]].astype(int)

            # if limb_type == 0:
            #     print(person_joint_info, joint1, joint2)

            if joint1[-1] == -1 or joint2[-1] == -1:
                continue

            if joint1[0] == 0 or joint2[0] == 0 or joint1[1] == 0 or joint2[0] == 0:
                continue
            
            joint_coords = [joint1[:2], joint2[:2]]
            for joint in joint_coords:
                cv2.circle(image, tuple(joint.astype(
                    int)), 4, (255, 255, 255), thickness=-1)

            # mean along the axis=0 computes mean Y coord and mean X coord
            coords_center = tuple(
                np.round(np.mean(joint_coords, 0)).astype(int))

            limb_dir = joint_coords[0] - joint_coords[1]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            cur_image = image if bool_fast_plot else image.copy()
            polygon = cv2.ellipse2Poly(
                coords_center, (int(limb_length / 2), limb_thickness),
                int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_image, polygon, colors[limb_type])

            if not bool_fast_plot:
                image = cv2.addWeighted(image, 0.4, cur_image, 0.6, 0)

    # to_plot is the location of all joints found overlaid of image
    if bool_fast_plot:
        to_plot = image.copy()
    else:
        to_plot = cv2.addWeighted(img, 0.3, image, 0.7, 0)
    return to_plot, image

def find_annotation_by_image_id(image_id, annotations):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            return annotation
    return None

def annotate_keypoints(image_path, json_path, save_dir):

    image = cv2.imread(image_path)
    image_plot = image.copy()

    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = int(image_path.split('.')[-2].split('/')[-1])
    img = coco.loadImgs(imgIds)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for i in range(len(anns)):
        pose_gt = np.array(anns[i]['keypoints']).reshape(1, 17, 3)
        image_plot, _ = plot_pose(image_plot, pose_gt)
    img_file = os.path.join(save_dir, 'groundtruth_{}.jpg'.format(str(image_path.split('.')[-2].split('/')[-1])))
    cv2.imwrite(img_file, image_plot)

json_path = "data/coco/annotations/person_keypoints_val2017.json"
save_dir = "demo_dirs/GT/"

annotate_keypoints("tests/data/coco/000000382009.jpg", json_path, save_dir)