from .voc import *
import torch

def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    difs = []
    for i in batch:
        images.append(i[0])
        boxes.append(torch.FloatTensor(i[1][0]))
        labels.append(torch.LongTensor(i[1][1]))
        difs.append(torch.BoolTensor(i[1][2]))
    
    images = torch.stack(images)
    return images, (boxes, labels, difs)