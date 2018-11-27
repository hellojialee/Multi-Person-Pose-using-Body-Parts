# Multi-Person Pose Estimation using Body Parts
Code and pre-trained models for our paper.

## Introduction
A bottom-up approach for the problem of multi-person pose estimation.
The complete source code will be released as soon as the paper is accepted.

### Contents
1. Training 
2. Evaluation 
3. Demo

### Task Lists
- [] Rewrite and speed up the code of keypoint assignment in C++ 

### Results on COCO Validation Dataset 


## Code Features
- Implement the models using Keras with TensorFlow backend
- Supprot training on multiple GPUs and slice training samples among GPUs
- Fast data preparing and augmentation during training
- Different learning rate at different layers

## Prepare
1. Download the COCO dataset 
2. [Download the pre-trained models](https://www.dropbox.com/s/bsr03ahhnaxppnf/model%26demo.rar?dl=0) 
3. Change the paths in the code according to your environment

## Run a Demo
`python demo_image.py`


## Evaluation Steps
The corresponding code is in pure python without multiprocess for now.

`python testing/evaluation.py`

## Training Steps
- [] The training code is available soon

    
## Referred Repositories
-  [Realtime Multi-Person Pose Estimation verson 1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
-  [Realtime Multi-Person Pose Estimation verson 2](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)
-  [Realtime Multi-Person Pose Estimation version 3](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
-  [Associative Embedding](https://github.com/princeton-vl/pose-ae-train)
- [Maxing Multiple GPUs of Different Sizes with Keras and TensorFlow](https://github.com/jinkos/multi-gpus)
