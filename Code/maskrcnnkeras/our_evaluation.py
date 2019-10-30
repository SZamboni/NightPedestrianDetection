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

from our_dataset import OurDataset

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
        
        
        entry = {
            "category_id": int(class_id),
            "bbox" : bbox,
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


'''
Function that takes in input the ground truth file of the annotation 
and the output of a network in the json format and outputs the
miss rates in the output file
'''
def evaluation(annFile,resFile,outFile = "results.txt"):
    from coco import COCO # IMPORT THEIR COCO, not pycocotools
    from eval_MR_multisetup import COCOeval
    
    # running evaluation
    res_file = open("results.txt", "w")
    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup,res_file)

    res_file.close()