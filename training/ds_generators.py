#coding:utf-8
import numpy as np
import zmq
from ast import literal_eval as make_tuple
from py_rmpe_server.py_rmpe_data_iterator import RawDataIterator
from time import time


import six
if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa


class DataIteratorBase(object):   # DataIterator(DataIteratorBase)

    def __init__(self, global_config, batch_size=10):

        self.global_config = global_config
        self.batch_size = batch_size

        self.split_point = global_config.paf_layers
        self.vec_num = global_config.paf_layers    # FIXME if you add PAF foreground, you should +1
        self.heat_num = global_config.heat_layers + 1  # FIXME if you add keypoint foreground, you should +1

        self.image_shape = (self.batch_size, self.global_config.width, self.global_config.height, 3)  # global_config.width = 368
        self.mask1_shape = (self.batch_size, self.global_config.width//self.global_config.stride, self.global_config.height//self.global_config.stride, self.vec_num)
        self.mask2_shape = (self.batch_size, self.global_config.width//self.global_config.stride, self.global_config.height//self.global_config.stride, self.heat_num)
        self.ypafs1_shape = (self.batch_size, self.global_config.width//self.global_config.stride, self.global_config.height//self.global_config.stride, self.vec_num)
        self.yheat2_shape = (self.batch_size, self.global_config.width//self.global_config.stride, self.global_config.height//self.global_config.stride, self.heat_num)

        #self.keypoints = [None]*self.batch_size # this is never passed to NN, will be accessed by accuracy calculation

    def restart(self):  # 在这个基类中没有定义，但是会在后面的继承类中定义，这里是用覆盖父类方法的方式重写的

        assert False, "Not implemented"  # should restart connection, server should start new cycle on connection.

    def _recv_arrays(self):
        pass   # 放在子类中实现

    def gen_raw(self):  # this function used for test purposes in py_rmpe_server

        self.restart()  # 先将restart中的部分生效

        while True:
            yield tuple(self._recv_arrays())  # 基类中没有定义_recv_arrays()方法，不过在下面的子类中定义了


    def gen(self):

        sample_idx = 0
        batches_x = np.empty(self.image_shape)
        batches_x1 = np.empty(self.mask1_shape)
        batches_x2 = np.empty(self.mask2_shape)
        batches_y1 = np.empty(self.ypafs1_shape)
        batches_y2 = np.empty(self.yheat2_shape)

        for foo in self.gen_raw():  # 真正的生成数据的函数是gen_raw()  !!!!!!!!!!

            if len(foo) == 4:
                data_img, mask_img, label, kpts = foo
            else:
                data_img, mask_img, label = foo
                kpts = None

            batches_x[sample_idx] = data_img[np.newaxis, ...]
            # 在返回数组之前，前面加了一个轴 np.newaxis 是为了满足网络的输入Tensor的shape要求

            batches_x1[sample_idx, :, :, :] = mask_img[np.newaxis, :, :, :self.split_point]
            # 在返回数组之前，前面加了一个轴 np.newaxis 是为了满足网络的输入Tensor的shape要求
            batches_x2[sample_idx, :, :, :] = mask_img[np.newaxis, :, :, self.split_point:]

            batches_y1[sample_idx] = label[np.newaxis, :, :, :self.split_point ]
            batches_y2[sample_idx] = label[np.newaxis, :, :, self.split_point: ]

            #self.keypoints[sample_idx] = kpts

            sample_idx += 1

            if sample_idx == self.batch_size:
                # --------------------------------------------------  #
                # 在这里控制生成的数据个数，达到设定的batch时，yield返回batch个数据
                # --------------------------------------------------- #
                sample_idx = 0

                if self.vec_num>0 and self.heat_num>0:
                    # print('*************************************')
                    # print(batches_x.shape, batches_y1.shape)  # (10, 368, 368, 3) (10, 46, 46, 38) 当batch=10
                    yield [batches_x, batches_x1,  batches_x2], \
                          [batches_y1, batches_y2,
                            batches_y1, batches_y2,
                            # batches_y1, batches_y2,  # todo: change the number of output according to the number of output
                            # batches_y1, batches_y2,
                            batches_y1, batches_y2,
                            # batches_y1, batches_y2
                    ]
                    # 返回的是包含两个list元素的   tuple  ！!！! 这是keras中fit_generator需要满足的形式,分别对应网络的输入和输出
                    # batches_x1, batches_x2分别是paf和keypoint heatmap的mask,为了训练模型时去除没有标注的人群的影响

                elif self.vec_num == 0 and self.heat_num > 0:  # 当只有confidence map没有paf　map的时候

                    yield [batches_x, batches_x2], \
                          [batches_y2, batches_y2, batches_y2, batches_y2, batches_y2, batches_y2]

                else:
                    assert False, "Wtf or not implemented"

                # we should recreate this arrays because we in multiple threads, can't overwrite　# TODO：don't know　why
                # I use such things to prevent strange behaviour in multithreaded environment, actually keras iterator
                # could be used in multithreaded mode, and there will be race condition if one thread will be copying
                # data to gpu and other will be changing it same time
                # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/19
                # https://gist.github.com/dusenberrymw/b3268ac795ff9e48acebfc798a4045b7
                batches_x = np.empty(self.image_shape)
                batches_x1 = np.empty(self.mask1_shape)
                batches_x2 = np.empty(self.mask2_shape)
                batches_y1 = np.empty(self.ypafs1_shape)
                batches_y2 = np.empty(self.yheat2_shape)

                #self.keypoints = [None] * self.batch_size

    def keypoints(self):
        return self.keypoints

    def num_samples(self):
        assert False, "Not Implemented"


class DataGeneratorClient(DataIteratorBase):

    def __init__(self, global_config, host, port, hwm=20, batch_size=10, limit=None):

        super(DataGeneratorClient, self).__init__(global_config, batch_size)

        self.limit = limit
        self.records = 0

        """
        :param host:
        :param port:
        :param hwm:, optional
          The `ZeroMQ high-water mark (HWM)
          <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
          sending socket. Increasing this increases the buffer, which can be
          useful if your data preprocessing times are very random.  However,
          it will increase memory usage. There is no easy way to tell how
          many batches will actually be queued with a particular HWM.
          Defaults to 10. Be sure to set the corresponding HWM on the
          receiving end as well.
        :param batch_size:
        :param shuffle:
        :param seed:
        """
        self.host = host
        self.port = port
        self.hwm = hwm
        self.socket = None

        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(self.hwm)
        self.socket.connect("tcp://{}:{}".format(self.host, self.port))


    def _recv_arrays(self):
        """Receive a list of NumPy arrays.
        Parameters
        ----------
        socket : :class:`zmq.Socket`
        The socket to receive the arrays on.
        Returns
        -------
        list
        A list of :class:`numpy.ndarray` objects.
        Raises
        ------
        StopIteration
        If the first JSON object received contains the key `stop`,
        signifying that the server has finished a single epoch.
        """

        if self.limit is not None and self.records > self.limit:
            raise StopIteration

        headers = self.socket.recv_json()
        if 'stop' in headers:
            raise StopIteration
        arrays = []

        for header in headers:
            data = self.socket.recv()
            buf = buffer_(data)
            array = np.frombuffer(buf, dtype=np.dtype(header['descr']))
            array.shape = make_tuple(header['shape']) if isinstance(header['shape'], str) else header['shape']
            # this need for comparability with C++ code, for some reasons it is string here, not tuple

            if header['fortran_order']:
                array.shape = header['shape'][::-1]
                array = array.transpose()
            arrays.append(array)

        self.records += 1
        return arrays


class DataIterator(DataIteratorBase):
    # 继承了DataIteratorBase Class
    def __init__(self, global_config, config, shuffle=True, augment=True, batch_size=10, limit=None):

        super(DataIterator, self).__init__(global_config, batch_size)

        self.limit = limit
        self.records = 0
        self.global_config = global_config
        self.config = config
        self.shuffle = shuffle
        self.augment = augment

        self.raw_data_iterator = RawDataIterator(self.global_config, self.config, shuffle=self.shuffle, augment=self.augment)
        self.generator = None

    def restart(self):

        self.records = 0
        self.generator = self.raw_data_iterator.gen()

    def num_samples(self):
        return self.raw_data_iterator.num_keys()

    def _recv_arrays(self):

        while True:  # 生成器需要是无限循环生成数据

            if self.limit is not None and self.records > self.limit:
                raise StopIteration("Limit Reached")

            tpl = next(self.generator, None)  # 因为raw_data_iterator.gen()具有关键字yield，所以自动支持next()方法
            if tpl is not None:
                self.records += 1
                return tpl   # 返回了多个对象image, mask, labels, meta['joints']

            raise StopIteration("Limited and reached cycle")


