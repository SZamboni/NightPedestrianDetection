from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import os

import sys
sys.path.append('../')

from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.utils import extract_bboxes
from numpy import expand_dims
from numpy import mean
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.utils import compute_ap
from Mask_RCNN.mrcnn.model import load_image_gt
from Mask_RCNN.mrcnn.model import mold_image
from matplotlib import pyplot

import random

from pycocotools.coco import COCO

'''
NightOwls training has  images, of which: 130064
Background: 104394
With Pedestrians: 25670

therefore for every ped image we have 4 background images

shuffling of the images is done automatycally by mrcnn
'''

# class that defines and loads the nightowls dataset
class OurExp5TrainDataset(Dataset):
    
    # load the dataset definitions
    def load_dataset(self, images_dir, annFile, ped_ratio = 1):
        
        self.cocovar = COCO(annFile)
        self.imgIds = list(sorted(self.cocovar.imgs.keys()))
        self.images_dir = images_dir
        self.imgs_count = len(self.imgIds)
        
        self.add_class("dataset", 1, "pedestrian")
        
        i = 0 # count of the images id in the dataset
        total_ped_imgs = 0
        total_back_imgs = 0
        
        random.seed(1)
        
        #take all the pedestrians
        last_ped_taken = False
        for im_id in self.imgIds:
            cocoimg = self.cocovar.loadImgs(ids=im_id)[0]
            img_filename = cocoimg['file_name']
            path = os.path.join(self.images_dir,img_filename)
            
            image_annotations = self.cocovar.getAnnIds(im_id)
            
            pedestrian_present = False
            
            for ann_id in image_annotations:
                ann = self.cocovar.loadAnns(ids=ann_id)[0]
                
                if(ann['category_id'] == 1):
                    pedestrian_present = True
            
            if pedestrian_present == True:
                if last_ped_taken == False:
                    tmp = self.take_or_not_img('dataset', i, path, im_id, 1)
                    
                    if tmp == True:
                        total_ped_imgs = total_ped_imgs + 1
                        i=i+1
                        
                    last_ped_taken = True  
                else:
                    last_ped_taken = False
                    
        #take all the background images
        last_back_taken = False
        back_img_count = 0
        for im_id in self.imgIds:
            cocoimg = self.cocovar.loadImgs(ids=im_id)[0]
            img_filename = cocoimg['file_name']
            path = os.path.join(self.images_dir,img_filename)
            
            image_annotations = self.cocovar.getAnnIds(im_id)
            
            pedestrian_present = False
            
            for ann_id in image_annotations:
                ann = self.cocovar.loadAnns(ids=ann_id)[0]
                
                if(ann['category_id'] == 1):
                    pedestrian_present = True
            
            if pedestrian_present == False:
                if last_back_taken == False:
                    if back_img_count >= 3: # take one in every 4
                        back_img_count = 0
                        tmp = self.take_or_not_img('dataset', i, path, im_id, 1)
                        if tmp == True:
                            total_back_imgs = total_back_imgs + 1
                            i=i+1
                    else:
                        back_img_count = back_img_count + 1
                    last_back_taken = True
                else:
                    last_back_taken = False                
        
        print('Pedestrian Images: ' + str(total_ped_imgs))
        print('Background Images: ' + str(total_back_imgs))
        

    def take_or_not_img(self, text,image_id,path,real_id, probability):
        result = random.uniform(0, 1)
        
        if result < probability:
            self.add_image(text, image_id=image_id, path=path, real_id = real_id)
            return True
        else:
            return False
        
    # load the masks for an image
    def load_mask(self, image_id):
        
        # get the image id and its filename
        #img_id = self.imgIds[image_id]
        img_id = self.image_info[image_id]['real_id']
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
            
            # skip the non-pedestrinas
            if(ann['category_id'] != 1):
                continue
            
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