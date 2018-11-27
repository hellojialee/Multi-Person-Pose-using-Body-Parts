from keras.utils import plot_model
from keras import Model, Sequential
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Conv2D, MaxPooling2D, \
    AveragePooling2D, UpSampling2D, Lambda, Dropout
from keras.layers.merge import Concatenate, Multiply, Add
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras.applications.vgg19 import preprocess_input
import keras.backend as K
import tensorflow as tf
import re
import numpy as np
from config import COCOSourceConfig, GetConfig

config = GetConfig("Canonical")


def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, name, weight_decay, stride=1, use_bias=False, use_bn=False, use_relu=True):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None  # 后面给的值是(weight_decay, 0)，所以没有对bias做规则化
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), strides=stride, padding='same', name=name,
               kernel_regularizer=kernel_reg,  # 如果没有这一个参数，那么传递的就是None
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),  #Initializer that generates tensors with a normal distribution
               bias_initializer=constant(0.0),
               use_bias=use_bias)(x)
    if use_bn:
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, name=name + '_bn', trainable=True)(x)

    if use_relu:
        x = relu(x)
    return x


def dilated_conv(x, nf, ks, name, weight_decay, stride=1, use_bias=False, dialated_rate=2, use_bn=False, use_relu=True):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None  # 后面给的值是(weight_decay, 0)，所以没有对bias做规则化
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), strides=stride, padding='same',
               dilation_rate=dialated_rate,
               name=name,
               kernel_regularizer=kernel_reg,  # 如果没有这一个参数，那么传递的就是None
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01), #Initializer that generates tensors with a normal distribution
               bias_initializer=constant(0.0),
               use_bias=use_bias)(x)

    if use_bn:
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name=name + '_bn', trainable=True)(x)

    if use_relu:
        x = relu(x)
    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def slice(x, h1, h2, w1, w2):
    """ Define a tensor slice function
    """
    return x[:, h1:h2, w1:w2, :]


def resize_like(input_tensor, ref_tensor):  # resizes input tensor wrt. ref_tensor
    H, W = tf.shape(ref_tensor)[1], tf.shape(ref_tensor)[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H, W])


def slice_layer(name):
    return Lambda(slice, arguments={'h1': 2, 'h2': -2, 'w1': 2, 'w2': -2}, name=name)


def ones_like():
    return Lambda(lambda x: K.ones_like(x))


def apply_mask(x, mask1, mask2, num_p, stage, branch, np_branch1, np_branch2):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    # s_name = "weight_stage%d_L%d" % (stage, branch)

    # TODO: we have branch number here why we made so strange check
    assert np_branch1 != np_branch2  # we selecting branches by number of pafs, if they accidentally became the same it will be disaster

    if num_p == np_branch1:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight
    elif num_p == np_branch2:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    else:
        assert False, "wrong number of layers num_p=%d " % num_p
    # w = slice_layer(name=s_name)(w)
    return w


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=1):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), padding='same', dilation_rate=dilation, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, dilation_rate=dilation,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), padding='same', dilation_rate=dilation, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), padding='same', strides=strides, dilation_rate=dilation,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation,
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), padding='same', dilation_rate=dilation, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), padding='same', strides=strides, dilation_rate=dilation,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def vgg_block(input_tensor, weight_decay):
    # Block 1
    x = conv(input_tensor, 64, 3, "conv1_1", (weight_decay, 0), use_bias=True)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0), use_bias=True)
    x_1 = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x_1, 128, 3, "conv2_1", (weight_decay, 0), use_bias=True)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0), use_bias=True)
    x_2 = pooling(x, 2, 2, "pool2_1")

    shortcut1 = conv(x_2, 64, 1, "conv2_shortcut1", (weight_decay, 0), stride=2,  use_bias=True)  # todo:增加一处shortcut, 提高分辨率之前stride=2,
    # 通过1*1卷及对low-level的特征通道进行压缩，这样后面的高层特征数量占主体，是一种程度上的偏重

    # Block 3
    x = conv(x_2, 256, 3, "conv3_1", (weight_decay, 0), use_bias=True)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0), use_bias=True)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0), use_bias=True)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0), use_bias=True)
    x_3 = pooling(x, 2, 2, "pool3_1")  # todo: 去掉一个pooling，提高了分辨率

    shortcut2 = conv(x_3, 64, 1, "conv2_shortcut2", (weight_decay, 0), use_bias=True)  # todo:增加一处shortcut

    # Block 4
    x = conv(x_3, 512, 3, "conv4_1", (weight_decay, 0), use_bias=True)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0), use_bias=True)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0), use_bias=True)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0), use_bias=True)
    #
    # # merge the lower level features and higher level features
    # resized_x_1 = Lambda(resize_like, arguments={'ref_tensor': x})(x_1)
    # resized_x_2 = Lambda(resize_like, arguments={'ref_tensor': x})(x_2)
    #
    x = Concatenate()([shortcut1, shortcut2,  x])
    # x = Dropout(rate=0.5)(x)

    return x


def stage1_block(x, num_p, stack_number, branch, weight_decay):

    # main branch

    x_1 = conv(x, 32, 1, 'iposenet_stack_%d_branch_%d_main_conv11' % (stack_number, branch), weight_decay, use_bias=True)
    x_1 = conv(x_1, 32, 3, 'iposenet_stack_%d_branch_%d_main_conv12' % (stack_number, branch), weight_decay, use_bias=True)

    x_2 = conv(x, 32, 1, 'iposenet_stack_%d_branch_%d_main_conv21' % (stack_number, branch), weight_decay, use_bias=True)
    x_2 = conv(x_2, 32, 3, 'iposenet_stack_%d_branch_%d_main_conv22' % (stack_number, branch), weight_decay, use_bias=True)
    x_2 = conv(x_2, 32, 3, 'iposenet_stack_%d_branch_%d_main_conv23' % (stack_number, branch), weight_decay, use_bias=True)

    x_3 = conv(x, 64, 1, 'iposenet_stack_%d_branch_%d_main_conv31' % (stack_number, branch), weight_decay, use_bias=True)
    x_3 = dilated_conv(x_3, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv32' % (stack_number, branch),
                     weight_decay, dialated_rate=2, use_bias=True)
    x_3 = dilated_conv(x_3, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv33' % (stack_number, branch),
                       weight_decay, dialated_rate=5, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大

    # dilation 3,4,5,5的感受野是35
    x_4 = conv(x, 64, 1, 'iposenet_stack_%d_branch_%d_main_conv41' % (stack_number, branch), weight_decay, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv42' % (stack_number, branch),
                     weight_decay, dialated_rate=3, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv43' % (stack_number, branch),
                       weight_decay, dialated_rate=4, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv44' % (stack_number, branch),
                       weight_decay, dialated_rate=5, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv45' % (stack_number, branch),
                       weight_decay, dialated_rate=5, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大

    #
    concate = Concatenate()([x_1, x_2, x_3, x_4])
    concate = conv(concate, 256, 1, 'iposenet_stack_%d_branch_%d_main_conv_cate' % (stack_number, branch), weight_decay, use_bias=True, use_relu=False)

    add_feature = Add()([x, concate])
    add_feature = relu(add_feature)

    attention_feature_next = conv(add_feature, 256, 1, 'iposenet_stack_%d_branch_%d_merge_conv1' % (stack_number, branch), weight_decay, use_bias=True)

    # regression branch
    # add_feature = Dropout(rate=0.5)(add_feature)
    regression_x = conv(add_feature, 256, 3, 'iposenet_stack_%d_branch_%d_regress_conv1' % (stack_number, branch), weight_decay, use_bias=True)
    regression_x = conv(regression_x, 128, 1, 'iposenet_stack_%d_branch_%d_regress_conv2' % (stack_number, branch), weight_decay, use_bias=True)
    regression_x = conv(regression_x, num_p, 1, 'iposenet_stack_%d_branch_%d_regress_conv3' % (stack_number, branch), weight_decay, use_bias=True, use_relu=False)
    # 最后一个回归层保险起见不添加activation !!!

    return regression_x, attention_feature_next


def stage_block(x, num_p, stack_number, branch, weight_decay):
    # x = ZeroPadding2D(2)(x)  # Fixme: the shape input into hourglass

    x_1 = conv(x, 32, 1, 'iposenet_stack_%d_branch_%d_main_conv11' % (stack_number, branch), weight_decay, use_bias=True)
    x_1 = conv(x_1, 64, 3, 'iposenet_stack_%d_branch_%d_main_conv12' % (stack_number, branch), weight_decay, use_bias=True)
    # It should be 32 here which is claimed in the paper. But it dose not matter much
    x_2 = conv(x, 32, 1, 'iposenet_stack_%d_branch_%d_main_conv21' % (stack_number, branch), weight_decay, use_bias=True)
    x_2 = conv(x_2, 32, 3, 'iposenet_stack_%d_branch_%d_main_conv22' % (stack_number, branch), weight_decay, use_bias=True)
    x_2 = conv(x_2, 32, 3, 'iposenet_stack_%d_branch_%d_main_conv23' % (stack_number, branch), weight_decay, use_bias=True)

    x_3 = conv(x, 64, 1, 'iposenet_stack_%d_branch_%d_main_conv31' % (stack_number, branch), weight_decay, use_bias=True)
    x_3 = dilated_conv(x_3, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv32' % (stack_number, branch),
                     weight_decay, dialated_rate=3, use_bias=True)
    x_3 = dilated_conv(x_3, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv33' % (stack_number, branch),
                       weight_decay, dialated_rate=4, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大

    # dilation=3,3,4,4,5,5的组合的感受野是49,而输入feature map尺寸为46*46
    x_4 = conv(x, 64, 1, 'iposenet_stack_%d_branch_%d_main_conv41' % (stack_number, branch), weight_decay, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv42' % (stack_number, branch),
                     weight_decay, dialated_rate=3, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv43' % (stack_number, branch),
                     weight_decay, dialated_rate=3, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv44' % (stack_number, branch),
                     weight_decay, dialated_rate=4, use_bias=True)
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv45' % (stack_number, branch),
                       weight_decay, dialated_rate=4, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv46' % (stack_number, branch),
                       weight_decay, dialated_rate=5, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大
    x_4 = dilated_conv(x_4, 64, 3, 'iposenet_stack_%d_branch_%d_main_diated_conv47' % (stack_number, branch),
                       weight_decay, dialated_rate=5, use_bias=True)  # todo: 计算一下感受野，看看是否需要再更大

    #
    concate = Concatenate()([x_1, x_2, x_3, x_4])
    concate = conv(concate, 256, 1, 'iposenet_stack_%d_branch_%d_main_conv_cate' % (stack_number, branch), weight_decay, use_bias=True, use_relu=False)

    add_feature = Add()([x, concate])

    attention_feature_next = conv(add_feature, 256, 1, 'iposenet_stack_%d_branch_%d_merge_conv1' % (stack_number, branch), weight_decay, use_bias=True)

    # regression branch
    # add_feature = Dropout(rate=0.5)(add_feature)
    regression_x = conv(add_feature, 256, 3, 'iposenet_stack_%d_branch_%d_regress_conv1' % (stack_number, branch), weight_decay, use_bias=True)
    regression_x = conv(regression_x, 128, 1, 'iposenet_stack_%d_branch_%d_regress_conv2' % (stack_number, branch), weight_decay, use_bias=True)
    regression_x = conv(regression_x, num_p, 1, 'iposenet_stack_%d_branch_%d_regress_conv3' % (stack_number, branch), weight_decay, use_bias=True, use_relu=False)
    # 最后一个回归层保险起见不添加activation !!!

    return regression_x, attention_feature_next


def get_testing_model(np_branch1=36, np_branch2=19, stages=3, weight_decay=None):
    # todo: change the number of stages of test model, at the begining

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)
    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  #  [-0.5, 0.5] todo: change it!
    # 或许在这个范围是有意义的，因为让网络的输入有正有负

    # VGG
    # stage0_out = DenseNet([6, 12, 24, 16], img_normalized, 0)
    stage0_out = vgg_block(img_normalized, weight_decay)
    # resnet = ResNet50(img_input, weight_decay)
    # stage0_out = resnet(img_normalized)

    # stage 1 - branch 1 (PAF)
    new_x = []
    after_out = []
    stage1_branch1_out, featuremap_next1 = stage1_block(stage0_out, np_branch1, 1, 1, (None, 0))
    new_x.append(featuremap_next1)
    after_out.append(stage1_branch1_out)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out, featuremap_next2 = stage1_block(stage0_out, np_branch2, 1, 2, (None, 0))
    new_x.append(featuremap_next2)
    after_out.append(stage1_branch2_out)

    new_x.append(stage0_out)
    out_cov = Concatenate()(after_out)
    out_cov = conv(out_cov, 256, 1, 'iposenet_stack_%d_afterpred_conv' % 1, weight_decay, use_bias=True)
    new_x.append(out_cov)
    x = Add()(new_x)  # Concatenate要变成加，要保持hourglass输入的ch

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        new_x = []
        after_out = []

        stageT_branch1_out, featuremap_next_t1 = stage_block(x, np_branch1, sn, 1, (0, 0))
        new_x.append(featuremap_next_t1)
        after_out.append(stageT_branch1_out)

        stageT_branch2_out, featuremap_next_t2 = stage_block(x, np_branch2, sn, 2, (0, 0))
        new_x.append(featuremap_next_t2)
        after_out.append(stageT_branch2_out)

        if (sn < stages):
            new_x.append(x)  # 上一个stage的输入直接传到下一个stage的输入
            out_cov = Concatenate()(after_out)
            out_cov = conv(out_cov, 256, 1, 'iposenet_stack_%d_afterpred_conv' % sn, weight_decay, use_bias=True)
            new_x.append(out_cov)
            x = Add()(new_x)
    if stages == 1:

        model = Model(inputs=[img_input], outputs=[stage1_branch1_out, stage1_branch2_out])  # todo: outputs=[stageT_branch1_out, stageT_branch2_out]

    else:
        model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])  # todo: outputs=[stageT_branch1_out, stageT_branch2_out]

    return model
