# split into train and test set
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


# class that defines and loads the nightowls dataset
class OurDataset(Dataset):
    
    # load the dataset definitions
    def load_dataset(self, images_dir, annFile, is_train=True):
        from pycocotools.coco import COCO
        self.cocovar = COCO(annFile)
        self.imgIds = list(sorted(self.cocovar.imgs.keys()))
        self.images_dir = images_dir
        
        self.add_class("dataset", 1, "pedestrian")
        self.add_class("dataset", 2, "bycicle")
        self.add_class("dataset", 4, "ignore")
        
        i=0
        for im_id in self.imgIds:
            cocoimg = self.cocovar.loadImgs(ids=im_id)[0]
            img_filename = cocoimg['file_name']
            path = os.path.join(self.images_dir,img_filename)
            
            if i >= 150 and is_train == False: # validation
                self.add_image('dataset', image_id=i, path=path, real_id = im_id)
            elif i < 150 and is_train == True: # training
                self.add_image('dataset', image_id=i, path=path, real_id = im_id)
            
            i=i+1

    # load the masks for an image
    def load_mask(self, image_id):
        
        # get the image id and its filename
        img_id = self.imgIds[image_id]
        cocoimg = self.cocovar.loadImgs(ids=img_id)[0]
        
        img_filename = cocoimg['file_name']
        img_height = cocoimg['height']
        img_width = cocoimg['width']

        #load the ids of the annotations for the images
        image_annotations = self.cocovar.getAnnIds(img_id)
        
        boxes = []
        categories = []
        
        # for every annotation
        for ann_id in image_annotations:
            #load all the annotation data
            ann = self.cocovar.loadAnns(ids=ann_id)[0]
            data = []
            for i in ann['bbox']:
                data.append(i)
            categories.append(ann['category_id'])
            data[2] = data[2]+data[0] # from x,y,width,height to xmin,ymin,xmax,ymax
            data[3] = data[3]+data[1]
            boxes.append(data)
        
        # create one array for all masks, each on a different channel
        masks = zeros([img_height, img_width, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            if categories[i] == 1:
                class_ids.append(self.class_names.index('pedestrian'))
            elif categories[i] == 2:
                class_ids.append(self.class_names.index('bycicle'))
            else:
                class_ids.append(self.class_names.index('ignore'))
            
        return masks, asarray(class_ids, dtype='int32')
               

    # load an image reference
    def image_reference(self, image_id):
        '''
        img_id = self.imgIds[image_id]
        cocoimg = self.cocovar.loadImgs(ids=img_id)[0]
        
        img_filename = cocoimg['file_name']
        return img_filename
        '''
        info = self.image_info[image_id]
        return info['path']