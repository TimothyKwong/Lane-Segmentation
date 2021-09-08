# Format dataset within json for Dataloader

import json

class format_Dataset():
    def __init__(self, path_toLabels):
        self.path_toLabels = path_toLabels

    def getLabels(self):
        with open(self.path_toLabels, 'r') as f:
            labels = json.load(f)
            return labels
