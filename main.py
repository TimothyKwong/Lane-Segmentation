import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from gdataset import BDD100K_Dataset

class Main():
    def __init__(self):
        self.path_toLabels_train = 'BDD100K/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/polygons/drivable_train.json'
        self.path_toImage_train = 'BDD100K/bdd100k_images_100k/bdd100k/images/100k/train'
        self.path_toMasks_train = 'BDD100K/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks/train'

        self.path_toLabels_val = 'BDD100K/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/polygons/drivable_val.json'
        self.path_toImage_val = 'BDD100K/bdd100k_images_100k/bdd100k/images/100k/train'
        self.path_toMasks_val = 'BDD100K/bdd100k_drivable_labels_trainval/bdd100k/labels/drivable/masks/val'

        self.batch_size = 16
        self.num_classes = 3
        self.epochs = 125
        self.lr = 0.003
        self.weight_decay = 0.1
        self.momentum = 0.9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        image, target = zip(*batch)
        return image, target

    def get_Dataset(self):
        training_dataset = BDD100K_Dataset(self.path_toLabels_train, self.path_toImage_train, self.path_toMasks_train)
        training_generator = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        validation_dataset = BDD100K_Dataset(self.path_toLabels_val, self.path_toImage_val, self.path_toMasks_val)
        validation_generator = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        return training_generator, validation_generator

    def get_model(self):
        model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=self.num_classes, pretrained_backbone=False)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True)
        model.roi_heads.box_predictor.bbox_pred = FastRCNNPredictor(in_features, self.num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)
                                                        
        #model.to(self.device)
        optimizer = SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        criterion = CrossEntropyLoss()
        return model, optimizer, criterion

    def train_model(self, model, training_generator, optimizer, criterion):
        running_losses = []
        running_losses_val = []
        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_loss_val = 0.0
            for idx, (input, target) in enumerate(training_generator, 0):
                input = input.to(self.device).to(torch.float)
                label = label.to(self.device).to(torch.long)
                # clear the parameter gradient
                optimizer.zero_grad()
                # forward propagation
                output = model(input, target)
                # calculate the loss
                loss = criterion(output, target)
                # back propagation
                loss.backward()
                # update model weights
                optimizer.step()
                # track loss
                running_loss += loss.item()
            running_losses.append(running_loss)

            print( 'Epoch: {}, Loss: {}'.format(epoch, np.round(running_loss, 2)) )
        return running_losses, running_losses_val
    
    def main(self):
        training_generator, validation_generator = self.get_Dataset()
        model, optimizer, criterion = self.get_model()
        running_losses, running_losses_val = self.train_model(model=model, training_generator=training_generator, optimizer=optimizer, criterion=criterion)

if __name__=='__main__':
    Main().main()