import sys
import os
import pandas
import re
import math
sys.path.append("..")

from config import COCOSourceConfig, GetConfig
from mymodel import get_training_model
from ds_generators import DataGeneratorClient, DataIterator
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19
import keras.backend as K

from glob import glob

from testing.py_rmpe_config import RmpeCocoConfig

batch_size = 10
base_lr = 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy = "step"
gamma = 0.333
stepsize = 121746*17 # in original code each epoch is 121746 and step change is on 17th epoch
max_iter = 200
use_multiple_gpus = None  #2 # set None for 1 gpu, not 1


WEIGHTS_SAVE = 'weights.{epoch:04d}.h5'
TRAINING_LOG = "training.csv"
LOGS_DIR = "./logs"
WEIGHT_DIR = "./weights"
config_name = "Canonical"
config = GetConfig(config_name)


def get_last_epoch_and_weights_file():
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    files = [file for file in glob(WEIGHT_DIR + '/weights.*.h5')]
    files = [file.split('/')[-1] for file in files]
    epochs = [file.split('.')[1] for file in files if file]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
    if len(epochs) == 0:
        if 'weights.best.h5' in files:
            return -1, WEIGHT_DIR + '/weights.best.h5'
    else:
        ep = max([int(epoch) for epoch in epochs])
        return ep, WEIGHT_DIR + '/' + WEIGHTS_SAVE.format(epoch=ep)
    return None, None

model = get_training_model(weight_decay, np_branch1=config.paf_layers, np_branch2=config.heat_layers + 1, stages=3)
# todo: the key 'stages=' decide how many stages of the CNN model we will build)

from_vgg = dict()
from_vgg['conv1_1'] = 'block1_conv1'
from_vgg['conv1_2'] = 'block1_conv2'
from_vgg['conv2_1'] = 'block2_conv1'
from_vgg['conv2_2'] = 'block2_conv2'
from_vgg['conv3_1'] = 'block3_conv1'
from_vgg['conv3_2'] = 'block3_conv2'
from_vgg['conv3_3'] = 'block3_conv3'
from_vgg['conv3_4'] = 'block3_conv4'
from_vgg['conv4_1'] = 'block4_conv1'
from_vgg['conv4_2'] = 'block4_conv2'

# load previous weights or vgg19 if this is the first run
last_epoch, wfile = get_last_epoch_and_weights_file()
if wfile is not None:
    print("Loading %s ..." % wfile)

    model.load_weights(wfile)
    last_epoch = last_epoch + 1

else:
    print("Loading vgg19 weights...")

    vgg_model = VGG19(include_top=False, weights='imagenet')

    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)

    last_epoch = 0

# setup lr multipliers for conv layers
lr_mult=dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):

        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # vgg
        else:
           kernel_name = layer.weights[0].name
           bias_name = layer.weights[1].name
           lr_mult[kernel_name] = 1
           lr_mult[bias_name] = 2

# configure loss functions

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    l = K.sum(K.square(x - y)) / batch_size / 2
    return l

#losses = {}
#losses["weight_stage1_L1"] = eucl_loss
#losses["weight_stage1_L2"] = eucl_loss
#losses["weight_stage2_L1"] = eucl_loss
#losses["weight_stage2_L2"] = eucl_loss
#losses["weight_stage3_L1"] = eucl_loss
#losses["weight_stage3_L2"] = eucl_loss
#losses["weight_stage4_L1"] = eucl_loss
#losses["weight_stage4_L2"] = eucl_loss
#losses["weight_stage5_L1"] = eucl_loss
#losses["weight_stage5_L2"] = eucl_loss
#losses["weight_stage6_L1"] = eucl_loss
#losses["weight_stage6_L2"] = eucl_loss


# prepare generators

# True = start zmq client, False local client
use_client_gen = False

if use_client_gen:
    train_client = DataGeneratorClient(config, port=5555, host="localhost", hwm=160, batch_size=batch_size)
    val_client = DataGeneratorClient(config, port=5556, host="localhost", hwm=160, batch_size=batch_size)
else:
    train_client = DataIterator(config, COCOSourceConfig("../dataset/coco/link2coco2017/coco_train_dataset.h5"),
                                shuffle=True,
                                augment=True, batch_size=batch_size)
    val_client = DataIterator(config, COCOSourceConfig("../dataset/coco/link2coco2017/coco_val_dataset.h5"),
                              shuffle=False, augment=False,
                              batch_size=batch_size)
train_di = train_client.gen()
train_samples = 117576
val_di = val_client.gen()
val_samples = 2475


# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_samples // batch_size

def step_decay(epoch):
    steps = epoch * iterations_per_epoch * batch_size
    lrate = base_lr * math.pow(gamma, math.floor(steps/stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate

print("Weight decay policy...")
for i in range(1,100,5): step_decay(i)

# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHT_DIR + '/' + WEIGHTS_SAVE, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)
tnan = TerminateOnNaN()
#coco_eval = CocoEval(train_client, val_client)

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

# start training

# if use_multiple_gpus is not None:
#     from keras.utils import multi_gpu_model
#     model = multi_gpu_model(model, gpus=use_multiple_gpus)

model.compile(loss=eucl_loss, optimizer=multisgd)


model.fit_generator(train_di,
                    steps_per_epoch=iterations_per_epoch,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=val_samples // batch_size,
                    use_multiprocessing=True,
                    initial_epoch=last_epoch
                    )

# loss = model.evaluate_generator(val_di,
#                    steps=val_samples // batch_size,
#                    use_multiprocessing=True,
#                    )
# print(loss)
