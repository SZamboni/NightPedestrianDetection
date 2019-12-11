import random
from os import path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from coco import COCO

# Ground truth
annFile = '/nightowls/annotations/nightowls_training.json'
image_directory = '/nightowls/images'


cocoGt = COCO(annFile)
imgIds = sorted(cocoGt.getImgIds())
print('There are %d images in the training set' % len(imgIds))

annotations = cocoGt.getAnnIds()
print('There are %d annotations in the training set' % len(annotations))


# Select random annotation
anno_id = annotations[random.randint(0, len(annotations))]
anno = cocoGt.loadAnns(ids=anno_id)[0]
print('Annotation (id=%d): %s' % (anno_id, anno))

cat = cocoGt.loadCats(ids=anno['category_id'])[0]
category_name = cat['name']
print('Object type %s' % category_name)


# Show the annotation in its image
image = cocoGt.loadImgs(ids=anno['image_id'])[0]
file_path = path.join(image_directory, image['file_name'])


fig,ax = plt.subplots(1)

img=mpimg.imread(file_path)
ax.imshow(img)

bbox = anno['bbox']
rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='g',facecolor='none')
ax.add_patch(rect)


plt.show()




