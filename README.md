# Pedestrian detection at night with Deep Learning
Pedestrian detection at night with Deep Learning project done for the Research Methodology and Scientific Writing course of periods 1 and 2 (from September 2019 to January 2020) held by Mihhail Matskin at KTH Royal Institute of Technology. 
Our project aimed to improve the state-of-the-art techniques for detection using deep learning on dark images.
The project has been done by Olivier Nicolini and Simone Zamboni. 
The mian code was done in Keras and it is based on the repository https://github.com/matterport/Mask_RCNN implementing MaskRCNN with Keras. 
The dataset used is the NihtOwls dataset (http://www.nightowls-dataset.org/ ) and if you want to rund the code you have to download training and validation images on your PC.
In this repository tere is the report for the course project, under the name "Report_Nicolini_Zamboni.pdf".

## Repository structure
In the main folder it is possible to find our report for the course.
The folder "Code" contains a directory called "Preliminary_test" that contains some tests we did with Pytorch to implement the project, but in the end we abondoned this route and therefore these file are only here for reference.
The folder containing the actual code is called "Maskrcnn-keras". In order to make the code work the repostory https://github.com/matterport/Mask_RCNN needs to be copied inside the folder "Mask_RCNN". In the main folder "Maskrcnn-keras" there are some useful files but the only necessary files for the experiments are in the folders "Experiment1" and "Experiment2".
In "Experiment1" there are the experiments regarding the different splits between background images and pedestrian images, while in the folder "Experiment2" there are the experiments concerning pre-processing of the images with algorithms such as Histogram Equalization, Adaptive Gamma Correction, CLAHE and MSRCR.

Our experiments were conducted on Google Cloud with a Nvidia K80 GPU with a total cost of more than 600$.
