# coding:utf-8
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K
from keras.layers import BatchNormalization
from keras.utils import plot_model
from keras import Model, Sequential
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Conv2D, MaxPooling2D, \
    AveragePooling2D, UpSampling2D, Lambda
from keras.layers.merge import Concatenate, Multiply, Add
from keras.regularizers import l2
from keras.initializers import random_normal, constant

from keras.engine import Layer, InputSpec

from keras import constraints


from keras.utils.generic_utils import get_custom_objects


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        reshape_group_shape = list(input_shape)
        reshape_group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape = [-1, self.groups]
        group_shape.extend(reshape_group_shape[1:])
        group_reduction_axes = list(range(len(group_shape)))

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        inputs = K.reshape(inputs, group_shape)
        mean = K.mean(inputs, axis=group_reduction_axes[2:], keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes[2:], keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        original_shape = [-1] + list(input_shape[1:])
        inputs = K.reshape(inputs, original_shape)

        if needs_broadcasting:
            outputs = inputs

            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta
        else:
            outputs = inputs

            if self.scale:
                outputs = outputs * self.gamma

            if self.center:
                outputs = outputs + self.beta

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


class FixedBatchNormalization(Layer):
    '''
        # 此处使用的是参数固定的BN层，参数在ImageNet是进行了训练，在微调网络的时候固定不动，当成了线性单元
        copy form
        https://github.com/yhenon/keras-frcnn/issues/33
        For the usage of BN layers, after pre-training, we com-
    pute the BN statistics (means and variances) for each layer
    on the ImageNet training set. Then the BN layers are fixed
    during fine-tuning for object detection. As such, the BN
    layers become linear activations with constant offsets and
    scales, and BN statistics are not updated by fine-tuning. We
    fix the BN layers mainly for reducing memory consumption
    in Faster R-CNN training.
    '''

    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(x)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Hourglass_Network():   # FIXME: add ReLu before usage the code here！
    """
    Instantiate an n order Hourglass Network using recursive trick.

    For instance:
    input_tensor = Input(shape=(368,368,256))
    hourglass = Hourglass_Network(4, 256, increase=128)
    output_hourglass = hourglass(input_tensor)
    my_model = Model(input_tensor, out_put_hourglass)

    Return an hourglass network
    """
    def __init__(self, n, f, bn=None, increase=128, stack_number=1, branch_number=1):
        """
        Define an N order Hourglass network
        :param n: the order of houglass network
        :param f: the channel of input tensor and output tensor,
        i.e. hourglass will remain the feature map size in the end as the input feature map size.
        :param bn: use BN layer or not
        :param increase: increase of feature map channel in the lower branch
        """
        super(Hourglass_Network, self).__init__()
        f = f
        nf = f + increase
        self.up1 = Conv2D(f, 3, padding='same',
                          name='hourglass_stack_%d_branch_%d_order_%d_up1' % (stack_number, branch_number, n))
        # Lower branch
        self.pool1 = MaxPooling2D(2, 2,  # fixme: 模型输入的尺寸需要改，要能够４倍下采样之后可以被hourglass整除
                                  name='hourglass_stack_%d_branch_%d_order_%d_pool1' % (stack_number, branch_number, n))
        self.low1 = Conv2D(nf, 3, padding='same',
                           name='hourglass_stack_%d_branch_%d_order_%d_low1' % (stack_number, branch_number, n))
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass_Network(n-1, nf,  bn=bn, stack_number=stack_number, branch_number=branch_number)
            # notice: 循环调用时初始化的参数要传入，不然会用class定义的默认缺省值
        else:
            self.low2 = Conv2D(nf, 3, padding='same',
                               name='hourglass_stack_%d_branch_%d_order_%d_low2' % (stack_number, branch_number, n))
        self.low3 = Conv2D(f, 3, padding='same',
                           name='hourglass_stack_%d_branch_%d_order_%d_low3' % (stack_number, branch_number, n))
        self.up2 = UpSampling2D(2,
                                name='hourglass_stack_%d_branch_%d_order_%d_up2' % (stack_number, branch_number, n))

        self.add = Add(name='hourglass_stack_%d_branch_%d_order_%d_res_add' % (stack_number, branch_number, n))

    def __call__(self, x):
        """
        building the hourglass network automaticlly
        :param x: input_tensor
        :return: hourglass output tensor
        """
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.add([up1, up2])


