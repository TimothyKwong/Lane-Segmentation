# Get dataset into Dataloader

import os
import torch
from PIL.Image import open
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import fdataset

class BDD100K_Dataset(Dataset):
    def __init__(self, path_tolabels, path_toImage, path_toMasks):
        self.Label = fdataset.format_Dataset(path_tolabels).getLabels()
        self.path_tolabels = path_tolabels
        self.path_toImage = path_toImage
        self.path_toMasks = path_toMasks
        self.t = Compose([ToTensor()])
        self.n_samples = len(self.Label)

    def __getitem__(self, index):
            labels = self.Label[index]

            try:
                length = len(labels.get('labels'))

                image_name = os.path.join(self.path_toImage, labels.get('name'))
                mask_name = os.path.join(self.path_toMasks, labels.get('name')[:-3]+'png')

                boxes = []
                for idx in range( length ):
                    boxes.append( labels.get('labels')[idx].get('poly2d')[0].get('vertices') )

                label = []
                for idx in range( length ):
                    cate = labels.get('labels')[idx].get('category')
                    if cate == 'direct':
                        label.append( 0 )
                    elif cate == 'alternative':
                        label.append( 1 )
                    else: 
                        label.append( 2 ) #background

                image = self.t(open(image_name))
                mask = self.t(open(mask_name))

                print(boxes) #<FIX HERE>
                print(label) #<FIX HERE>
                target = {'boxes':torch.FloatTensor(boxes), 'labels':torch.FloatTensor(label), 'mask':mask}
                return image, target

            except TypeError:
                pass
                """
                boxes = torch.tensor([0, 0, 0, 0])
                label = torch.tensor([2])
                mask = self.t(open(mask_name))
                image = self.t(open(image_name))
                target = {'boxes':boxes, 'labels':label, 'mask':mask}
                return image, target
                """

    def __len__(self):
        return self.n_samples