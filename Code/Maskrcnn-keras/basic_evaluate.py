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

import json

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN

from our_dataset_onlyped import OurDataset

from our_evaluation import generateAnnotations, evaluation

# define the prediction configuration
class PredictionConfig(Config):
    
    IMAGES_PER_GPU = 1
    # define the name of the configuration
    NAME = "pedestrian_only_testcfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 2
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main(images_dir, ann_file, pred_file, net_weights_file):
    
    # test set
    test_set = OurDataset()
    test_set.load_dataset(images_dir, ann_file, is_train=True, val_percentage = 0.0)
    test_set.prepare()
    print('Test images: %d' % len(test_set.image_ids))
    
    # create config
    cfg = PredictionConfig()

    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

    # load model weights
    model.load_weights(net_weights_file, by_name=True)

    # generate the annotations for the test set using the model
    json_output = generateAnnotations(test_set,model,cfg)

    # save the file
    with open(pred_file, 'w') as outfile:
        json.dump(json_output, outfile)

    print('RESULTS FILE GENERATED')
    
    # evalutate the predictions and save them in results.txt
    evaluation(ann_file,pred_file)
    
    print('EVALUATION DONE')
    
    print('PROGRAM FINISHED')

model_weights_path = '/home/test/data/trained_models/pedestrian_only_traincfg20191030T1419/mask_rcnn_pedestrian_only_traincfg_0002.h5'
ann_file_path = '/home/test/data/nightowls/nightowls_validation_small.json'
images_path = '/home/test/data/nightowls/validation/nightowls_validation'
net_predictions_file = './out.json'

main(images_path,ann_file_path, net_predictions_file, model_weights_path)
