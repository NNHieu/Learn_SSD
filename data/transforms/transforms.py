from . import functional as myTF
import torchvision.transforms.functional as TF
from random import random

class SSDTrainDataTrans:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.expand_scale = 4

    def __call__(self, image, target):
        target = myTF.target_to_tensor(target)
        boxes, labels, difs = target
        image = myTF.photometric_distort(image)
        
        image = TF.to_tensor(image)
        if random() < 0.5: 
            image, boxes = myTF.expand(image, 
                                        boxes, 
                                        filler=self.mean, 
                                        max_scale=self.expand_scale)
        
        image, boxes, labels, difs = myTF.random_crop(image, boxes, labels, difs)
        
        image = TF.to_pil_image(image)
        if random() < 0.5:
            image, boxes = myTF.flip(image, boxes)
        image, boxes = myTF.resize(image, boxes, return_percent_coords=True)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        return image, (boxes, labels, difs)

class SSDValDataTrans:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, target):
        target = myTF.target_to_tensor(target)
        boxes, labels, difs = target

        image, boxes = myTF.resize(image, boxes, return_percent_coords=True)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        return image, (boxes, labels, difs)