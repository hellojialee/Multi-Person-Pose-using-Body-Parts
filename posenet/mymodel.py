from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
import re

def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None  # 后面给的值是(weight_decay, 0)，所以没有对bias做规则化
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,  # 如果没有这一个参数，那么传递的就是None
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01), #Initializer that generates tensors with a normal distribution
               bias_initializer=constant(0.0))(x)
    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")  # todo: 去掉一个pooling，提高了分辨率

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch, np_branch1, np_branch2):
    w_name = "weight_stage%d_L%d" % (stage, branch)

    # TODO: we have branch number here why we made so strange check
    assert np_branch1 != np_branch2 # we selecting branches by number of pafs, if they accidentally became the same it will be disaster

    if num_p == np_branch1:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight
    elif num_p == np_branch2:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    else:
        assert False, "wrong number of layers num_p=%d " % num_p
    return w


def get_training_model(weight_decay, np_branch1, np_branch2, stages=6, gpus = None):
    # training_model一共有三个输入！　即原始image，以及在训练模型过程中评估时，去除没有标记区域的confidence和paf的mask1, mask2
    img_input_shape = (None, None, 3)   # 输入的是3通道的图像！　如果是灰度图，读入的rgb每个通道值相同
    vec_input_shape = (None, None, np_branch1)
    heat_input_shape = (None, None, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    if np_branch1 > 0:
        inputs.append(vec_weight_input)

    if np_branch2 > 0:
        inputs.append(heat_weight_input)  # 网络在训练过程中的输入其实是有三个部分组成的input

    #img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
    img_normalized = img_input  # will be done on augmentation stage, in py_rmpe_data_iterator.py

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    new_x = []
    if np_branch1 > 0:
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
        w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1, np_branch1, np_branch2)
        outputs.append(w1)
        new_x.append(stage1_branch1_out)

    # stage 1 - branch 2 (confidence maps)

    if np_branch2 > 0:
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
        w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2, np_branch1, np_branch2)
        outputs.append(w2)
        new_x.append(stage1_branch2_out)

    new_x.append(stage0_out)  # 每一个stage都会有S, L, F三个特征图作为输入，其中F是VGG的输出

    x = Concatenate()(new_x)

    # stage sn >= 2
    for sn in range(2, stages + 1):

        new_x = []
        # stage SN - branch 1 (PAF)
        if np_branch1 > 0:
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
            w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1, np_branch1, np_branch2)
            outputs.append(w1)
            new_x.append(stageT_branch1_out)

        # stage SN - branch 2 (confidence maps)
        if np_branch2 > 0:
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2, np_branch1, np_branch2)
            outputs.append(w2)
            new_x.append(stageT_branch2_out)

        new_x.append(stage0_out)  # 每一个stage都会有S, L, F三个特征图作为输入，其中F是VGG的输出

        if sn < stages:
            x = Concatenate()(new_x)

    model = Model(inputs=inputs, outputs=outputs)  # 模型的输出是各个stage的PAF和confidence maps

    # --------------------------------------------------------------------------------------------------------#

    # 注意！　outputs与new_x是不同的，在训练网络阶段，new_x是每一个stage的网络真实的特征图输出，将作为下一个stage的输入，它包含了
    # 对没有标注区域的预测（在这个问题中预测出的结果就是heatmap和paf特征图）。　而在训练中因为训练用的label加入了mask,
    # 所以，outputs其实是去掉mask之后用来与label进行比较计算误差时使用的特征图。二者需要注意区别，二者都不能少！

    # ---------------------------------------------------------------------------------------------------------#
    return model


def get_lrmult(model):   # TODO: Set learning rate multipliers for different layers 设置不同阶段或者layer为不同的学习率　

    # setup lr multipliers for conv layers
    lr_mult = dict()

    for layer in model.layers:

        if isinstance(layer, Conv2D):  # 载入卷积网络kernel的权值

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):  # 最外层是对每一层layer进行循环，看是否是需要载入权值的层
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1   #
                lr_mult[bias_name] = 2
            # 模型每一层的命名格式如：
            # Mconv1_stage2_L1
            # Mconv1_stage2_L2
            # Mconv1_stage3_L1
            # Mconv1_stage3_L2
            # Mconv1_stage4_L1

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
               print("matched as vgg layer", layer.name)
               kernel_name = layer.weights[0].name
               bias_name = layer.weights[1].name
               lr_mult[kernel_name] = 1
               lr_mult[bias_name] = 2

    return lr_mult


def get_testing_model(np_branch1=36, np_branch2=19, stages=3):
    # todo: change the number of stages of test model, at the begining

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model

