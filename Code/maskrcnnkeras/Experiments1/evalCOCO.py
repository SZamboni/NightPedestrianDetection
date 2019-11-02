'''
Evaluation script for the OFFICIAL COCO evaluation

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

'''
REAL COCO EVALUATION
'''
def evaluationCOCO(annFile,resFile):
    from pycocotools.coco import COCO # from the pycocotools library
    from pycocotools.cocoeval import COCOeval
    
    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    
def main(ann_file, pred_file):
    print('COCO evaluation for:')
    print(pred_file)
    print('Ground truth:')
    print(ann_file)
    evaluationCOCO(ann_file,pred_file)

if __name__ == '__main__':
    ann_file_path = '/home/test/data/nightowls/nightowls_validation_small.json'
    net_predictions_file = '../out.json'

    main(ann_file_path, net_predictions_file)