'''
Generate the prediction file for a dataset an evaluate it
using the COCO evaluation and the NightOwls evaluation

'''

from evalCOCO import evaluationCOCO
from evalNightOwls import evaluationNightOwls

from our_datasets import OurTestDataset, OurTrainDataset, OurALLTestDataset, OurUselessValidationDataset

import sys
sys.path.append('../')

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

import json


def fromOutputToAnn(image_id,out):
    
    recognized_objects = []
    
    for i in range(len(out['class_ids'])):
        class_id = out['class_ids'][i]
        bbox = out['rois'][i]
        score = out['scores'][i]
        # from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]
        bbox[2] = bbox[2]-bbox[0]
        bbox[3] = bbox[3]-bbox[1]
        
        bbox[0] = float(bbox[0])
        bbox[1] = float(bbox[1])
        bbox[2] = float(bbox[2])
        bbox[3] = float(bbox[3])
        bbox = bbox.tolist()
        
        new_box = [bbox[1],bbox[0],bbox[3],bbox[2]]
        
        
        entry = {
            "category_id": int(class_id),
            "bbox" : new_box,
            "score": float(score),
            "image_id" : int(image_id)
        }
        recognized_objects.append(entry)
        
    return recognized_objects
    
'''
Function that from a dataset and a model returns a file with all the predictions
'''
def generateAnnotations(dataset,model,cfg):
    i = 0
    all_outputs = []
    for image_id in dataset.image_ids:
        
        if (image_id % 100) == 0:
            print (image_id)
        # load image info
        info = dataset.image_info[image_id]
        
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        
        out = fromOutputToAnn(info['real_id'],yhat)
        all_outputs.extend(out)
        
        i = i+1
    return all_outputs


# define the prediction configuration
class PredictionConfig(Config):
    
    IMAGES_PER_GPU = 1
    # define the name of the configuration
    NAME = "night_testcfg"
    # number of classes (background + pedestrian)
    NUM_CLASSES = 2
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    
def main(images_dir, ann_file, pred_file, net_weights_file, nightowl_results_file):
    
    # test set
    test_set = OurTestDataset()
    test_set.load_dataset(images_dir, ann_file)
    test_set.prepare()
    print('Test images: %d' % len(test_set.image_ids))
    
    # create config
    cfg = PredictionConfig()

    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

    # load model weights
    model.load_weights(net_weights_file, by_name=True)

    print('START RESULT FILE GENERATION')
    
    # generate the annotations for the test set using the model
    json_output = generateAnnotations(test_set,model,cfg)

    # save the file
    with open(pred_file, 'w') as outfile:
        json.dump(json_output, outfile)

    print('RESULTS FILE GENERATED')
    
    print('EVALUATION COCO')
    evaluationCOCO(ann_file,pred_file)
    
    print('EVALUATION NIGHTOWLS')
    evaluationNightOwls(ann_file,pred_file,nightowl_results_file)
    
    print('PROGRAM FINISHED')

net_weights_file = '/home/test/data/trained_models/exp1-220191104T1724/mask_rcnn_exp1-2_0066.h5'
ann_file = '/home/test/data/nightowls/nightowls_validation.json'
images_dir = '/home/test/data/nightowls/validation/nightowls_validation'
net_predictions_file = './out.json'
nightowl_results_file = './nightwols_eval.txt'

main(images_dir,ann_file, net_predictions_file, net_weights_file,nightowl_results_file)


