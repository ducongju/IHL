# Boosting Integral-based Human Pose Estimation Through Implicit Heatmap Learning

## Introduction
Human pose estimation typically encompasses three categories: heatmap-, regression-, and integral-based methods. While integral-based methods possess advantages such as end-to-end learning, full-convolution learning, and being free from quantization errors, they have garnered comparatively less attention due to inferior performance. In this paper, we revisit integral-based approaches for human pose estimation and propose a novel implicit heatmap learning paradigm, aiming to bridge the performance gap between heatmap- and integral-based methods. Specifically, Simple Implicit Heatmap Normalization (SIHN) is first introduced to calculate implicit heatmaps as an efficient representation for human pose estimation. As implicit heatmaps may introduce potential challenges related to variance and shape ambiguity arising from the inherent nature of implicit heatmaps, we thus propose a Differentiable Spatial-to-Distributive Transform (DSDT) method to aptly map those implicit heatmaps onto the transformation coefficients of a deformed distribution. The deformed distribution is predicted by a likelihood-based generative model to unravel the shape ambiguity quandary effectively, and the transformation coefficients are learned by a regression model to resolve the variance ambiguity issue. Additionally, to expedite the acquisition of precise shape representations throughout the training process, we introduce a Wasserstein distance-based constraint to ensure stable and reasonable supervision during the initial generation of implicit heatmaps. Experimental results on both the MSCOCO and MPII datasets demonstrate the effectiveness of our proposed method, achieving competitive performance against heatmap-based approaches while maintaining the advantages of integral-based approaches.

## Main Results
### Results on COCO val2017
| Backbone | Input size | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR |
|--------------------|--------------|-------|-------|--------|--------|--------|-------|
| **pose_resnet_101** | 256x192 | 70.4 | 87.4 |  77.0 |  66.3 |  78.7 |  78.8 |
| **pose_hrnet_w32** | 256x192 | 71.3 |  88.1 |  77.7 |  67.7 |  79.3 |  79.6 |

### Results on COCO test-dev2017
| Backbone | Input size | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR |
|--------------------|--------------|-------|-------|--------|--------|--------|-------|
| **pose_resnet_101** | 256x192 | 70.4 | 90.3 |  77.9 |  66.7 |  77.5 |  78.2 |
| **pose_hrnet_w32** | 256x192 | 71.5 |  90.4 |  79.0 |  68.1 |  78.4 |  79.2 |

## Environment

The code is developed using python 3.7, torch 1.10, torchvision 0.11 on cuda 11.1. NVIDIA GPUs are needed. The code is developed and tested using 2 NVIDIA A100 GPU cards for HRet-W32. Other platforms are not fully tested.

## Quick start

### Prepare the directory

1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.

2. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir work_dirs
   mkdir test_dirs
   mkdir test_results
   mkdir vis_input
   mkdir vis_output
   mkdir data
   ```

### Data preparation

**For [COCO](http://cocodataset.org/) data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing). Optionally, to evaluate on COCO'2017 test-dev, please download the [image-info](https://download.openmmlab.com/mmpose/datasets/person_keypoints_test-dev-2017.json). Download and extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- │-- annotations
                │   │-- person_keypoints_train2017.json
                │   |-- person_keypoints_val2017.json
                │   |-- person_keypoints_test-dev-2017.json
                |-- person_detection_results
                |   |-- COCO_val2017_detections_AP_H_56_person.json
                |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
                │-- train2017
                │   │-- 000000000009.jpg
                │   │-- 000000000025.jpg
                │   │-- 000000000030.jpg
                │   │-- ...
                `-- val2017
                    │-- 000000000139.jpg
                    │-- 000000000285.jpg
                    │-- 000000000632.jpg
                    │-- ...

**For [MPII](http://human-pose.mpi-inf.mpg.de/) data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). We have converted the original annotation files into json format, please download them from [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar). Extract them under {POSE_ROOT}/data, and make them look like this:

    ${POSE_ROOT}
    |-- data
    `-- │-- mpii
        `-- |-- annotations
            |   |-- mpii_gt_val.mat
            |   |-- mpii_test.json
            |   |-- mpii_train.json
            |   |-- mpii_trainval.json
            |   `-- mpii_val.json
            `-- images
                |-- 000001163.jpg
                |-- 000003072.jpg
                │-- ...


### Download the pretrained models

Download pretrained models and our well-trained models from [Google Drive](https://drive.google.com/drive/folders/1T5VQLccmQ82d9_KvCV5uATmCWjIyrGbr?usp=sharing) and make models directory look like this:

    ${POSE_ROOT}
    |-- work_dir       
    `-- |-- |-- IHL_COCO_HRNet32.pth
            |-- IHL_COCO_ResNet101.pth
            |-- IHL_COCO_ResNet50.pth
            |-- IHL_MPII_HRNet32.pth
            `-- IHL_MPII_ResNet50.pth

### Prepare the environment

If you are using SLURM (Simple Linux Utility for Resource Management), then execute:

```
sbatch prepare.sh
```

If you like, you can prepare the environment [**step by step**](https://github.com/open-mmlab/mmpose).

### Training and Testing

#### Training on COCO train2017 dataset

```
bash configs/configs_ihl/train.sh
```

#### Testing on COCO val2017 dataset using well-trained pose model

```
bash configs/configs_ihl/test.sh
```

#### Using heatmap demo

```
bash configs/configs_ihl/demo.sh
```

#### Using groundtruth demo

```
python configs/configs_ihl/demo_groundtruth.py
```

#### Using partitioned dataset

```
python configs/configs_ihl/coco_json_create.py
```

### Acknowledge
Thanks for the open-source [MMPose](https://github.com/open-mmlab/mmpose), it is a part of the [OpenMMLab](https://github.com/open-mmlab/) project.
### Other implementations
* [HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). It is a part of the [HRNet](https://github.com/HRNet) project.
