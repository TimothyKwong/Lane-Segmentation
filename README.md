# Ego-Lane Segmentation
## Description
This project aimed to train a state-of-the-art model for ego-lane segmentation using the [BDD100K](https://bdd-data.berkeley.edu/) dataset. The trained model would be used for ego-lane segmentation as a first step towards autonomous driving in Euro Truck Simulator 2.
## Note
The current code utilizes Mask-RCNN from the Pytorch library.However, due to the labels containing vertices for polygons of segmented lane regions, a different network will need to be utilized.
