# TRAINING
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import os
from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.utils import extract_bboxes
from Mask_RCNN.mrcnn.visualize import display_instances
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib import pyplot

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN

from our_dataset_onlyped_half import OurDataset

# define a configuration for the model
class TrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "pedestrian_only_traincfg"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 2
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
    
    LEARNING_RATE = 1e-3
    IMAGES_PER_GPU = 2

def main(ann_file_path,images_path,learning_rate,epochs,val_percentage):
    # train set
    train_set = OurDataset()
    train_set.load_dataset(images_path, ann_file_path, is_train=True, val_percentage = val_percentage)
    train_set.prepare()
    print('Train images: %d' % len(train_set.image_ids))

    # val set
    val_set = OurDataset()
    val_set.load_dataset(images_path, ann_file_path, is_train=False, val_percentage = val_percentage)
    val_set.prepare()
    print('Validation images: %d' % len(val_set.image_ids))

    # prepare config
    config = TrainConfig()

    # define the model
    model = MaskRCNN(mode='training', model_dir='/home/test/data/trained_models/', config=config)

    # load weights (mscoco)
    model.load_weights('/home/test/data/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

    print('START TRAINING')
    
    # train weights (output layers or 'heads')
    model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=2, layers='all')

    print('Training DONE')


ann_file_path = '/home/test/data/nightowls/nightowls_training.json'
images_path = '/home/test/data/nightowls/nightowls_training/nightowls_training'
learning_rate = 1e-3
epochs = 50
val_percentage = 0.005

main(ann_file_path,images_path,learning_rate,epochs, val_percentage)