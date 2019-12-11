from coco import COCO
from eval_MR_multisetup import COCOeval

# Ground truth
annFile = '/nightowls/annotations/nightowls_validation.json'

# Detections
resFile = '../sample-Faster-RCNN-nightowls_validation.json'

## running evaluation
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