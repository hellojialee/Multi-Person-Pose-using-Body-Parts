# coding:utf-8
import sys
import os
import math
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
import random
from posenet.mymodel3 import get_training_model, get_lrmult
from training.optimizers import MultiSGD, MultiAdam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
from keras.utils import GeneratorEnqueuer, Sequence
import keras.backend as K
from multigpus import ratio_training_utils  # 使用可以按照ration拆分整个batch size进行多GPU训练的函数
from keras.utils import multi_gpu_model
from glob import glob
from config import GetConfig
import h5py
from testing.inhouse_metric import calc_batch_metrics
from keras.optimizers import adam


base_lr = 1e-4  # 1.5e-4  # 2e-5  # 初始学习率大时发生了梯度爆炸，可能需要梯度裁剪
momentum = 0.9
weight_decay = 5e-4   #  5e-4  # 注意区别weight decay和learning rate decay的区别，前者是正则项的系数，后者是学习率衰减的系数
lr_policy = "step"
gamma = 0.6   # 0.6
stepsize = 119860 * 10  #117519 * 10  #  150307 * 10 # 117519 * 5 #  117519 * 2  # *17
#  in original code each epoch is 121746 and step change is on 17th epoch
# 训练集图片134898个，测试集259个
max_iter = 200


def get_last_epoch_and_weights_file(WEIGHT_DIR, WEIGHTS_SAVE, epoch):  # 载入上一个训练周期数和权值
    # TODO: recover the training process last time. 应该要保存优化器的状态，否则无法继续上次训练效果

    os.makedirs(WEIGHT_DIR, exist_ok=True)  # only python3 support this
    # if not os.path.exists(WEIGHT_DIR):  # for python2
    #     os.makedirs(WEIGHT_DIR)
    print('******************', WEIGHT_DIR)

    if epoch is not None and epoch != '': #override
        return int(epoch),  WEIGHT_DIR + '/' + WEIGHTS_SAVE.format(epoch=epoch)  # WEIGHTS_SAVE = 'weights.{epoch:04d}.h5'
        # 返回值例如：　(222, '***/weights.0222.h5')

    files = [file for file in glob(WEIGHT_DIR + '/weights.*.h5')]
    files = [file.split('/')[-1] for file in files]
    epochs = [file.split('.')[1] for file in files if file]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit()]  # Return True if all characters in S are digits
    if len(epochs) == 0:
        if 'weights.best.h5' in files:
            return -1, WEIGHT_DIR + '/weights.best.h5'
    else:
        ep = max([int(epoch) for epoch in epochs])
        return ep, WEIGHT_DIR + '/' + WEIGHTS_SAVE.format(epoch=ep)
    return None, None  # 与上面的返回格式保持一致，一共返回两个对象


# save names will be looking like
# training/canonical/exp1
# training/canonical_exp1.csv
# training/canonical/exp2
# training/canonical_exp2.csv

def prepare(config, config_name, exp_id, train_samples, val_samples, batch_size, epoch=None):

    metrics_id = config_name + "_" + exp_id if exp_id is not None else config_name  # exp_id: experiment_name, 'Canonical'
    weights_id = config_name + "/" + exp_id if exp_id is not None else config_name

    WEIGHT_DIR = "./" + weights_id
    WEIGHTS_SAVE = 'weights.{epoch:04d}.h5'

    TRAINING_LOG = "./" + metrics_id + ".csv"
    LOGS_DIR = "./logs"

    with tf.device("/cpu:0"):
        model = get_training_model(weight_decay, np_branch1=config.paf_layers, np_branch2=config.heat_layers+1, stack_number=3)  # fixme: background heat_layers+1
        # +1是因为加上了背景类
    # todo: the key 'stages=' decide how many stages of the CNN model we will build
    multi_model = ratio_training_utils.multi_gpu_model(model, gpus=[0, 1], ratios=[3, 2])  # 1080ti的id是0,　1080的是1
    # multi_model = multi_gpu_model(model, gpus=2)  # 1080ti的id是0,　1080的是1


    ''' 根据官方文档
    To save the multi-gpu model, use .save(fname) or .save_weights(fname) with the template model (the argument you
    passed to multi_gpu_model), rather than the model returned by multi_gpu_model.
    或者:自己制定回调函数保存：　https://www.jianshu.com/p/db0ba022936f
    或者:modelGPU.__setattr__('callback_model',modelCPU)
　　　　　#now we can train as normal and the weights saving in our callbacks will be done by the CPU model
　　　　　modelGPU.fit_generator( . . .　　　https://github.com/keras-team/keras/issues/8123
    '''
    lr_mult = get_lrmult(model)

    # load previous weights or vgg19 if this is the first run
    last_epoch, wfile = get_last_epoch_and_weights_file(WEIGHT_DIR, WEIGHTS_SAVE, epoch)
    print("last_epoch:", last_epoch)

    if wfile is not None:
        print("Loading multi-GPU model weights of last epoch  %s ..." % wfile)

        multi_model.load_weights(wfile)
        # fixme:Notice!!!  check_point保存的是多gpu模型，所以应该载入权值给multi_model,否则继续训练就是从头开始，loss会突然变大
        # 之前没有by_name=True,需要保证定义的模型每一层的名称与载入的是一样的，可以通过观察h5文件的key验证
    else:
        print('No pre-trained weights available, loade the vgg bottom laysers and initialize the model weights...')
        # 原始代码片段已删除
        model.load_weights('../training/cpu_weights/my_cpu_model_epoch_finetune.h5', by_name=True)
        # # print(multi_model.weights)  # Notice!  此处使用单GPU模型载入权值。多GPU模型不能载入单GPU模型，单GPU模型不能载入多GPU模型！
        # # print(model.get_weights()[2])

        last_epoch = 0

    # euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    def eucl_loss(x, y):
        l = K.sum(K.square(x - y)) / batch_size / 2   # 除以2的好处是，平方项的微分会出现2，抵消，可以减少乘法操作
        return l

    def focal_loss(gamma=2, alpha=0.75):
        def focal_loss_fixed(y_true, y_pred):  # with tensorflow
            eps = 1e-12
            y_pred = K.clip(y_pred, eps,
                            1. - eps)  # improve the stability of the focal loss and see issues 1 for more information
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return focal_loss_fixed

    def focal_loss_eucl(y_true, y_pred):   # see the original focal loss again , 　权重系数是怎么设置的
        # alpha = 0.25  # 原始是2.0
        gamma = 2.0
        pt = tf.where(tf.greater(y_true, 0.01), y_pred, 1 - y_pred)  #　原始是0.3
        return K.sum(K.pow(1. - pt, gamma) * K.square(y_true - y_pred)) / batch_size / 2

    # learning rate schedule - equivalent of caffe lr_policy =  "step"
    iterations_per_epoch = train_samples // batch_size

    def step_decay(epoch):  # todo: change the learning rate schedule
        steps = epoch * iterations_per_epoch * batch_size
        lrate = base_lr * math.pow(gamma, math.floor(steps/stepsize))
        print("Epoch:", epoch, "Learning rate:", lrate)
        return lrate

    print("Weight decay policy...")
    for i in range(1, 100, 5): step_decay(i)  # 按照固定间隔打印出learning rate

    # configure callbacks
    lrate = LearningRateScheduler(step_decay)
    lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, verbose=1, min_lr=1e-9)
    checkpoint = ModelCheckpoint(WEIGHT_DIR + '/' + WEIGHTS_SAVE, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger(TRAINING_LOG, append=True)   # 用这个回调函数记录每个周期的训练结果
    tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=False, write_images=False)
    tnan = TerminateOnNaN()
    #coco_eval = CocoEval(train_client, val_client)

    callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]  # lrate, 如果不使用lr schedule, 则可在lr decay上等效

    # ---------------------------- Chose the lr multipliers optimizer -------------------------- #
    # sgd optimizer with lr multipliers
    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

    # adam optimizer with lr multipliers
    multi_adam = MultiAdam(lr=base_lr, decay=2e-5, lr_mult=lr_mult)  # , clipnorm=1, clipvalue=0.5 使用adam改进版，amsgrad
    # 在训练RNN, LSTM时更容易出现梯度爆炸，会设置clipnorm=1, clipvalue=0.5 fixme finetune clipnor
    # ------------------------------------------------------------------------------------------ #
    sigle_adam = adam(lr=base_lr)
    # start training


    # # Plot the keras model
    # from keras.utils import plot_model
    # plot_model(model, 'modeltr.png', show_shapes=True)
    # print('------------------------- plot the model!!  ----------------------')


    # losses = {"weight_stage1_L1": focal_loss_eucl1, "weight_stage1_L2": focal_loss_eucl2,
    #           "weight_stage2_L1": focal_loss_eucl1, "weight_stage2_L2": focal_loss_eucl2}

    multi_model.compile(loss=focal_loss_eucl, optimizer=multi_adam,
                        loss_weights=[0.3333*0.5, 0.5, 0.3333*0.8, 0.8, 0.3333, 1])  # 之前默认按照0.5, 0.8, 1.0比例进行叠加
    # 如果模型是多个输出，则可以使用不同的loss，最后会sum所有的loss
    # Plot the keras model
    # from keras.utils import plot_model
    # plot_model(multi_model, 'multi_modeltr.png', show_shapes=True)
    # print('plot the model!!')

    return model, multi_model, iterations_per_epoch, val_samples//batch_size, last_epoch, metrics_id, callbacks_list


def train(config, model, multi_model, train_client, val_client, iterations_per_epoch, validation_steps, metrics_id, last_epoch, use_client_gen, callbacks_list):

    for epoch in range(last_epoch, max_iter):
        train_di = train_client.gen()
        val_di = val_client.gen()

        # ----------------------------------------------------------------------------------------------------------
        # 注意！！ 生成器train_client.gen()生成的数据有两个部分，网络的输入image, mask1, mask2，
        # 还有各个stage的网络输出需要比较的label,即2*6=12个feature map
        # ----------------------------------------------------------------------------------------------------------
        # train for one iteration
        multi_model.fit_generator(train_di,
                            steps_per_epoch=iterations_per_epoch,  # 一个周期内有多少个iteration，一个batch就是一个
                            epochs=epoch+1,
                            callbacks=callbacks_list,
                            use_multiprocessing=False,  # TODO: if you set True touching generator from 2 threads will stuck the program
                            initial_epoch=epoch,
                            validation_data=val_di,
                            validation_steps=validation_steps,  # //5 只取1/5的数据
                            )
        # Successive calls to fit do not reset any of the parameters of the model, including the state
        # of the optimizers. Successive calls to fit with nb_epoch = 1 is effectively the same as a single call to fit.

        # 使用pickle保存优化器状态
        # import pickle
        # pickle.loads()
        # optimizer_state = multi_model.optimizer.get_weights()
        # pickle.dumps(optimizer_state)

        validate(config, model, multi_model, val_client, validation_steps, metrics_id, epoch+1)  #
        # 不在fit_generator中计算validation loss是为了节省时间
        # 使用multi_model更快，因为model只是cpu上的模型
        model.save_weights('cpu_weights/my_cpu_model_epoch_%04d.h5' % (epoch + 1))  # 保存最终的cpu模型，以便载入单卡运行

    # Can not save model using model.save following multi_gpu_model: https://github.com/keras-team/keras/issues/8446
    # 直接使用keras的check_point只能保存multi模型，所以可以另外保存cpu上的拷贝，即原始的single模型的全职

# 运行tensorboard观察loss曲线：　
# tensorboard --logdir=/home/jia/Desktop/keras_Realtime_Multi-Person_Pose_Estimation-new-generation/training/logs


def validate(config, model, multi_model, val_client, validation_steps, metrics_id, epoch):
    # 得到的X是包含image, confidence mask, paf mask的list,得到的Y是包含6个stage一共12个groundtruth的heapmap
    # 网络一共有三个输入（对于训练时的评估，指标为了反映出训练的效果在测试时网络模型就不用考虑对feature map的输出进行mask了，
    # 可以对所有区域预测）,即原始image，以及在训练模型过程中评估时，去除没有标记区域的confidence和paf的mask1, mask2

    val_di = val_client.gen()

    val_thre = GeneratorEnqueuer(val_di)  # The provided generator can be finite in which case the class will throw
    # a `StopIteration` exception. 但是这里实现的gen貌似不存在这种问题。不过这个函数提供了multiprocess的封装
    val_thre.start()

    model_metrics = []
    inhouse_metrics = []
    t0 = time()
    for i in range(validation_steps):  # 分成很多个batch进行预测估计的，为了减少validation耗时，在计算validation部分数据
        # validation_steps　＝　val_samples//batch_size 为了防止内存OOM，所以要分batch预测
        # if random.randint(0, 9) < 5:  # 只计算20%的数据
        #     continue
        X, GT = next(val_thre.get())
        Y = multi_model.predict(X)

        model_losses = [(np.sum((gt - y) ** 2) / gt.shape[0] / 2) for gt, y in zip(GT,Y)]
        # 与模型定义时的loss保持一致，除以2的好处是，平方项的微分会出现2，抵消，可以减少乘法操作
        mm = sum(model_losses)

        if config.paf_layers > 0 and config.heat_layers > 0:
            GTL6 = np.concatenate([GT[-2], GT[-1]], axis=3)
            YL6 = np.concatenate([Y[-2], Y[-1]], axis=3)
            mm6l1 = model_losses[-2]   # NOTICE! 计算的是模型最后一个阶段的预测和groundtruth的距离
            mm6l2 = model_losses[-1]
        elif config.paf_layers == 0 and config.heat_layers > 0:
            GTL6 = GT[-1]
            YL6 = Y[-1]
            mm6l1 = None
            mm6l2 = model_losses[-1]
        else:
            assert False, "Wtf or not implemented"

        m = calc_batch_metrics(i, GTL6, YL6, range(config.heat_start, config.bkg_start))
        inhouse_metrics += [m]

        model_metrics += [(i, mm, mm6l1, mm6l2, m["MAE"].sum()/GTL6.shape[0], m["RMSE"].sum()/GTL6.shape[0], m["DIST"].mean()) ]
        # 以epoch为key，group之后取平均值
        print("Validating[BATCH: %d] LOSS: %0.4f, S6L1: %0.4f, S6L2: %0.4f, MAE: %0.4f, RMSE: %0.4f, DIST: %0.2f" % model_metrics[-1] )

    t1 = time()
    print('The CNN prediction time during validation is : ', t1 - t0)
    # inhouse_metrics = pd.concat(inhouse_metrics)
    # inhouse_metrics['epoch'] = epoch
    # inhouse_metrics.to_csv("logs/val_scores.%s.%04d.csv" % (metrics_id, epoch))  # , sep="\t" 默认的不是\t，而是','
    # # 保存的是每个层的细节
    #
    # model_metrics = pd.DataFrame(model_metrics, columns=("batch","loss","stage6l1","stage6l2","mae","rmse","dist") )
    # model_metrics['epoch'] = epoch
    # del model_metrics['batch']
    # model_metrics = model_metrics.groupby('epoch').mean()
    # with open('%s.val.tsv' % metrics_id, 'a') as f:
    #     model_metrics.to_csv(f, header=(epoch==1), float_format='%.4f')  # sep="\t",
    #
    # print(inhouse_metrics[["layer", "epoch", "MAE", "RMSE", "DIST"]].groupby(["layer", "epoch"]).mean())

    val_thre.stop()


def save_network_input_output(model, val_client, validation_steps, metrics_id, batch_size, epoch=None):

    val_di = val_client.gen()

    if epoch is not None:
        filename = "nn_io.%s.%04d.h5" % (metrics_id, epoch)
    else:
        filename = "nn_gt.%s.h5" % metrics_id

    h5 = h5py.File(filename, 'w')

    for i in range(validation_steps):
        X, Y = next(val_di)

        grp = h5.create_group("%06d" % i)

        for n, v in enumerate(X):
            grp['x%02d' % n] = v

        for n, v in enumerate(Y):
            grp['gt%02d' % n] = v

        if model is not None:

            Yp = model.predict(X, batch_size=batch_size)

            for n, v in enumerate(Yp):
                grp['y%02d' % n] = v

        print(i)

    h5.close()


def test_augmentation_speed(train_client):
    # 本项目中的数据增强代码能够足够快地满足GPU处理数据量的需要
    # batches per second 5.7466899871438635
    # batches per second 5.768000631491158
    # batches per second 5.733300557500513
    # It is far more than we really need for training(10/per second per gpu)
    # I.e. it's  enough for approx 6 cards. This is with parallel model training, i.e. it is second running
    # augmentation on this server.  #　https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/11

    train_di = train_client.gen()

    start = time()
    batch = 0

    for X, Y in train_di:

        batch +=1
        print("batches per second ", batch/(time()-start))


