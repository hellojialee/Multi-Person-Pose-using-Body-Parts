# Multi-Person Pose Estimation using Body Parts

Code and pre-trained models for our paper "*Multi-Person Pose Estimation Based on Body Parts*"

This repo is the **Part A** of our paper project.

**Pat B** is in the repo on GitHub: [**Improved-Body-Parts**](https://github.com/jialee93/Improved-Body-Parts)



## Introduction

A bottom-up approach for the problem of multi-person pose estimation. This **Part** is based on the network backbones in [CMU-Pose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (namely OpenPose). 

### Contents

1. Training 
2. Evaluation 
3. Demo

### Task Lists

- [ ] Rewrite and speed up the code of keypoint assignment in C++ 

## Project Features

- Implement the models using **Keras with TensorFlow backend**
- VGG as the backbone
- No batch normalization layer 
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

### Results on MSCOCO2018  Dataset

Results on MSCOCO2018 validation subset (model trained without val data, default size 368, 4 scales)

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.581
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.796
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.632
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.557
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.811
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.684
```

Results on MSCOCO 2018 validation subset (model trained with val data, default size 368, 4 scales)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.607
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.817
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.661
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.581
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.837
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.717
```

## Update 

Results on MSCOCO 2018 validation subset (model trained with val data, default size 368, 4 scales)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.622
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.828
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.674
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.844
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.730
```

Results on MSCOCO 2018 test-dev subset 

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.599
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.825
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.647
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.580
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.848
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.705
```

## News!

Recently, we are lucky to have time and machine to utilize. Thus, we revisit our previous work.
More accurate results had been achieved after we adopted more powerful Network and 
use higher resolution of heatmap (stride=4). Enhanced modeles with body part representation
, variant loss function and training parameters have been tried these days. 

Please refer to  [**Improved-Body-Parts**](https://github.com/jialee93/Improved-Body-Parts)

Results on MSCOCO 2018 test-dev subset 
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.681
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.864
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.668
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.882
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.683
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.773
```

## Training Steps

- [x] The training code is available

    

## Referred Repositories (mainly)

- [Realtime Multi-Person Pose Estimation verson 1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation verson 2](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation version 3](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
- [Associative Embedding](https://github.com/princeton-vl/pose-ae-train)
- [Maxing Multiple GPUs of Different Sizes with Keras and TensorFlow](https://github.com/jinkos/multi-gpus)
