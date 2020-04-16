# Multi-Person Pose Estimation using Body Parts

Code and pre-trained models for our paper.

This repo is the **Part A** of our paper project.

**Pat B** is in the repo on GitHub, [**Improved-Body-Parts**](https://github.com/jialee93/Improved-Body-Parts).

## Introduction

A bottom-up approach for the problem of multi-person pose estimation. This **Part** is based on the network backbones in [CMU-Pose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (namely OpenPose). A modified network is also trained and evalutated. 

![method](https://github.com/jialee93/Improved-Body-Parts/raw/master/visulizatoin/2987.Figure2.png)

### Contents

1. Training 
2. Evaluation 
3. Demo

### Task Lists

- [ ] Add more descriptions and complete the project

## Project Features

- Implement the models using **Keras with TensorFlow backend**
- VGG as the backbone 
- No batch normalization layer 
- Supprot training on multiple GPUs and slice training samples among GPUs
- Fast data preparing and augmentation during training
- Different learning rate at different layers 

## Prepare

1. Install packages:

   Python=3.6, Keras=2.1.2, TensorFlow-gpu=1.3.0rc2 and other packages needed (please refer to requirements.txt). We haven't tested other platforms and packages of different versions. 

2. Download the MSCOCO dataset.

3. Download the pre-trained models from [Dropbox](https://www.dropbox.com/s/bsr03ahhnaxppnf/model%26demo.rar?dl=0).

4. Change the paths in the code according to your environment.

## Run a Demo

`python demo_image.py`

## Evaluation Steps

The corresponding code is in pure python without multiprocess for now.

`python testing/evaluation.py` 

### Results on MSCOCO2017  Dataset

Results on MSCOCO 2017 validation subset (model trained without val data, + focal L2 loss, default size 368, 4 scales + flip)

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

We have tried new network structures.

<img src="https://github.com/jialee93/Multi-Person-Pose-using-Body-Parts/blob/master/posenet/model3.png" width="75%">

Results of posenet/model3 on MSCOCO 2017 validation subset (model trained with val data, + focal L2 loss, default size 368, 4 scales + flip). 

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

Results on MSCOCO 2017 test-dev subset (model trained with val data, + focal L2 loss, default size 368, 8 scales + flip)

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

According to our results, the performance of posenet/model3 in this repo is similar to CMU-Net (the cascaed CNN used in [CMU-Pose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) ), which means only merging different feature maps with diferent receptive fields at low resolution (stride=8) could not help much (without offset regression). And the limitted capacity of the network is also a bottleneck of the estimation accuracy. 

## News!

Recently, we are lucky to have time and machine to utilize. Thus, we revisit our previous work. More accurate results had been achieved after we adopted more powerful Network and use higher resolution of heatmaps (stride=4). Enhanced models with body part representation, variant loss functions and training parameters have been tried. 

<font color="#0000dd">Please also refer to our new repo</font>:  [**Improved-Body-Parts**](https://github.com/jialee93/Improved-Body-Parts) (highly recommended)

![improved](https://github.com/jialee93/Improved-Body-Parts/raw/master/visulizatoin/2987.Figure3.png)

Results on MSCOCO 2017 test-dev subset (focal L2 loss, default size 512, 5 scales + flip）

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.685
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.867
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.749
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.664
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.688
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.784
```

## Training Steps

Before training, prepare the training data using ''training/coco_masks_hdf5.py''.

- [x] The training code is available

`python training/train_pose.py`

**Notice**: change the sample slicing ratios between different GPUs at [this line in the code](https://github.com/jialee93/Multi-Person-Pose-using-Body-Parts/blob/7dd783f8dca4a8292232fcc724ecbd911a69f428/training/train_common.py#L82 ) as you want. 

## Referred Repositories (mainly)

- [Realtime Multi-Person Pose Estimation verson 1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation verson 2](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation version 3](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
- [Associative Embedding](https://github.com/princeton-vl/pose-ae-train)
- [Maxing Multiple GPUs of Different Sizes with Keras and TensorFlow](https://github.com/jinkos/multi-gpus)

