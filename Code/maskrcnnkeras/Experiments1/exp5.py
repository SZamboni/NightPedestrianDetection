# TRAINING

from our_datasets import OurTestDataset, OurTrainDataset, OurALLTestDataset, OurUselessValidationDataset
from exp5dataset import OurExp5TrainDataset

import sys
sys.path.append('../')

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN

# define a configuration for the model
class TrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "exp5-1"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 2
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300
    
    LEARNING_RATE = 1e-3
    IMAGES_PER_GPU = 2

def main(ann_file_path,images_path,epochs,ped_percentage,val_images_path , val_ann_file):
    # train set
    train_set = OurExp5TrainDataset()
    train_set.load_dataset(images_path, ann_file_path,ped_percentage ) # ,ped_percentage
    train_set.prepare()
    print('Train images: %d' % len(train_set.image_ids))

    # val set
    val_set = OurALLTestDataset()
    val_set.load_dataset(val_images_path, val_ann_file)
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
    model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=epochs, layers='all')

    print('Training DONE')


train_ann_file_path = '/home/test/data/nightowls/nightowls_training.json'
train_images_path =  '/home/test/data/nightowls/training/nightowls_training'
val_ann_file = '/home/test/data/nightowls/nightowls_validation_1.json'
val_images_path =  '/home/test/data/nightowls/validation/nightowls_validation'
learning_rate = 1e-3
epochs = 132 # corresponding to three pass on all data
ped_percentage = 1

main(train_ann_file_path,train_images_path,epochs, ped_percentage,val_images_path,val_ann_file)
