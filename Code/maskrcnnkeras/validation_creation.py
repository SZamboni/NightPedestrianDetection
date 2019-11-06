'''
The evaluation works only on all the dataset, and this is inconvenient
Here I try to circumvent it: create a json that represent a smaller part of the dataset,
with only the images processed by the model
'''

import json

annFile = '/home/test/data/nightowls/nightowls_validation.json'
resFile = './out.json'
output_file = './new_dataset.json'

with open(annFile, 'r') as f:
    annData = json.load(f)
    
with open(resFile, 'r') as f:
    resData = json.load(f)


for key, value in annData.items() :
    print (key)
    

newFile = {}
newFile['categories'] = annData['categories']
newFile['poses'] = annData['poses']
newFile['annotations']=[]
newFile['images']=[]

imgs_id_res = []

for i in range(len(resData)):
    imgs_id_res.append(resData[i]['image_id'])

imgs_id_res.sort()
print(len(imgs_id_res))

for i in range(len(annData['annotations'])):
    im_id = annData['annotations'][i]['image_id']
    if im_id in imgs_id_res:
        newFile['annotations'].append(annData['annotations'][i])
        
for i in range(len(annData['images'])):
    im_id = annData['images'][i]['id']
    if im_id in imgs_id_res:
        newFile['images'].append(annData['images'][i])
        
with open(output_file, 'w') as f:
        json.dump(newFile, f)    
    
print("FINISH")
