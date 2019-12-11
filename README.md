# Pedestrian detection at night with Deep Learning
Pedestrian detection at night with Deep Learning project done for the Research Methodology and Scientific Writing course of periods 1 and 2 (from September 2019 to January 2020) held by Mihhail Matskin at KTH Royal Institute of Technology. 
Our project aimed to improve the state-of-the-art techniques for detection using deep learning on dark images.
The project has been done by **Olivier Nicolini** and **Simone Zamboni**. <br/>
The code was done using Keras framework and it is based on the repository https://github.com/matterport/Mask_RCNN implementing MaskRCNN with Keras. <br/>
The dataset used is the NightOwls dataset (http://www.nightowls-dataset.org/ ) and if you want to run the code you have to download the training and validation images on your PC. <br/>
In this repository there is also the report for the course project, under the name "Report_Nicolini_Zamboni.pdf". <br/>

## Results

With signifincatly less training and resources we were able to achieve good performance by training a network on a subset of the whole dataset using half of the images with pedestrians inside and half of just background images. We also used Adaptive Gamma Correction as preprocessing to make images clearer and brighter before feeding them to the network.  <br/>
Using Adaptive Gamma Correction with Weighting Distribution to preprocess the images and 18k iterations of training, we achieved an mAP of 0.058 and a miss rate of 38%, while the Faster R-CNN trained for 100k iterations by the authors of the NightOwls dataset achieved an mAP of 0.060 and a miss rate of 29%.

## Repository structure
In the main folder it is possible to find our report for the course. <br/>
The folder "Code" contains a directory called "Preliminary_test" that contains some tests that we did using Pytorch framewok to implement the project, but in the end we abondoned this route since we used Keras, and therefore these file are only there for reference. <br/>
The folder containing the actual code is called "Maskrcnn-keras". In order to make the code work, the repostory https://github.com/matterport/Mask_RCNN needs to be copied inside the folder "Mask_RCNN". In the main folder "Maskrcnn-keras" there are some useful files, but the only necessary files for the experiments are in the folders "Experiment1" and "Experiment2". <br/>
In "Experiment1" there are the experiments regarding the different splits between background images and pedestrian images, while in the folder "Experiment2" there are the experiments concerning pre-processing of the images with algorithms such as Histogram Equalization, Adaptive Gamma Correction, CLAHE and MSRCR. <br/>

Our experiments were conducted on Google Cloud with a Nvidia K80 GPU with a total cost of more than 600$.
