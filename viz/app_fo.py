import fiftyone as fo
from PIL import Image

def create_fo_sample(image: Image, labels: str, boxes):
    """
    Args
    -----------
    image: PIL image
    labels: label name
    boxes: xyxy format
    """
    assert len(labels) == len(boxes)
    detections = []
    for i in range(len(labels)):
        detections.append(fo.Detection(label=labels[i], bounding_box=boxes[i]))
    sample = fo.Sample(filepath=image.filename)
    sample["ground_truth"] = fo.Detections(detections=detections)
    return sample