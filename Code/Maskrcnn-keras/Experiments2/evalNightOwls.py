'''
Evaluation script for the NIGHTWOLS evaluation with miss rate

'''

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

def evaluationNightOwls(annFile,resFile,outFile):
    from coco import COCO # IMPORT THEIR COCO, not pycocotools
    from eval_MR_multisetup import COCOeval
    
    # running evaluation
    res_file = open(outFile, "w")
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

def main(ann_file, pred_file,outFile):
    print('NIGHTOWLS evaluation for:')
    print(pred_file)
    print('Ground truth:')
    print(ann_file)
    print('saved in:')
    print(outFile)
    evaluationNightOwls(ann_file,pred_file,outFile)
      
if __name__ == '__main__':
    ann_file_path = '/home/test/data/nightowls/nightowls_validation_small.json'
    net_predictions_file = '../out.json'
    outFile = './nightwols_eval.txt'

    main(ann_file_path, net_predictions_file, outFile)