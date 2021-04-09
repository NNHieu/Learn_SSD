from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import VOCDetection as TorchVocDetection
from PIL import Image
import os
import xml.etree.ElementTree as ET

import warnings

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
VOC_LABEL_MAP = {k: v + 1 for v, k in enumerate(VOC_CLASSES)}
VOC_LABEL_MAP['background'] = 0
VOC_REV_LABEL_MAP = [None] * len(VOC_LABEL_MAP)  # Inverse mapping
for k, v in VOC_LABEL_MAP.items():
    VOC_REV_LABEL_MAP[v] = k

class VOCDetection(TorchVocDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        keep_difficult (bool): Keep difficult objects.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(
            self,
            root: str,
            keep_difficult: bool = False,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCDetection, self).__init__(root, year=year, image_set=image_set, download=download, transform=transform, target_transform=target_transform, transforms=transforms)
        self.keep_difficult = keep_difficult

    def parse_annotation(self, annotation):
        # Get detection data
        boxes, labels, difficulties = [], [], []
        for obj in annotation['object']:
            difficult = int(obj['difficult']) == 1
            if not self.keep_difficult and difficult: continue
            difficulties.append(difficult)

            name = obj['name'].lower().strip()
            labels.append(VOC_LABEL_MAP[name])

            bbox = obj['bndbox']
            boxes.append([int(c) for c in (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])])

        return [tuple(boxes), tuple(labels), tuple(difficulties)]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a tuple of boxes, labels and difficullties
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        target = self.parse_annotation(target['annotation'])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

def create_voc_dataset(root: str,splits = [(2007, 'trainval'), (2012, 'trainval')],
            keep_difficult: bool = False,
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None):
    """Helper function create Pascal VOC detection Dataset.

    Parameters
    ----------
    root (string): Root directory of the VOC Dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    keep_difficult (bool): Keep difficult objects.
    year (string, optional): The dataset year, supports years 2007 to 2012.
    image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
        (default: alphabetic indexing of VOC's 20 classes).
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, required): A function/transform that takes in the
        target and transforms it.
    transforms (callable, optional): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
    """
    ds = []
    for year, image_set in splits:
        ds.append(
            VOCDetection(root, keep_difficult=keep_difficult, 
                        year=str(year), image_set=image_set, 
                        download=download, 
                        transform=transform, target_transform=target_transform, transforms=transforms))
    if len(ds) > 1:
        ds = ConcatDataset(ds)
    else:
        ds = ds[0]
    return ds