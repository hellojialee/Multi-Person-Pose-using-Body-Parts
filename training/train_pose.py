import sys

sys.path.append("..")

from training.train_common import prepare, train, validate, save_network_input_output, test_augmentation_speed
from training.ds_generators import DataGeneratorClient, DataIterator
from config import COCOSourceConfig, GetConfig
import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # notice: 默认ti序号为0,若加这句，那么1080的序号为0   !!!
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

use_client_gen = False
batch_size = 25

task = sys.argv[1] if len(sys.argv) > 1 else "test_augmentation_speed"  # "train"  # "validate"
config_name = sys.argv[2] if len(sys.argv) > 2 else "Canonical"
experiment_name = sys.argv[3] if len(sys.argv) > 3 else None
if experiment_name == '': experiment_name = None
epoch = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '' else None

config = GetConfig(config_name)

train_client = DataIterator(config, COCOSourceConfig("../dataset/coco/link2coco2017/coco_train_dataset.h5"),
                            shuffle=True,
                            augment=True, batch_size=batch_size)
val_client = DataIterator(config, COCOSourceConfig("../dataset/coco/link2coco2017/coco_val_dataset.h5"), shuffle=False,
                          augment=False,
                          batch_size=batch_size)
train_samples = train_client.num_samples()  # train sample: 117519
val_samples = val_client.num_samples()  # val sample: 2533

model, multi_model, iterations_per_epoch, validation_steps, epoch, metrics_id, callbacks_list = \
    prepare(config=config, config_name=config_name, exp_id=experiment_name, train_samples=train_samples,
            val_samples=val_samples, batch_size=batch_size, epoch=epoch)

if task == "train":
    train(config, model, multi_model, train_client, val_client, iterations_per_epoch, validation_steps, metrics_id,
          epoch, use_client_gen, callbacks_list)

elif task == "validate":
    validate(config, model, multi_model, val_client, validation_steps, metrics_id, epoch)

elif task == "save_network_input_output":
    save_network_input_output(model, val_client, validation_steps, metrics_id, batch_size, epoch)

elif task == "save_network_input":
    save_network_input_output(None, val_client, validation_steps, metrics_id, batch_size)

elif task == "test_augmentation_speed":
    test_augmentation_speed(train_client)
