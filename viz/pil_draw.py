from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import numpy as np
import platform
from torch import Tensor
from torchvision.ops.boxes import box_convert

def grid_image(image, shape, thickness=1, color="#ffffff", ):
    im_width, im_height = image.size
    row, col = shape
    step_row, step_col = im_height / row, im_width / col
    draw = ImageDraw.Draw(image)
    for i in range(row):
        y = step_row * (i + 1)
        draw.line([(0, y), (im_width, y)], width=thickness, fill=color)
    for j in range(col):
        x = step_col * (j + 1)
        draw.line([(x, 0), (x, im_height)], width=thickness, fill=color)
    return image


def draw_boxes(boxes, image=None, draw=None, thickness=4, color="#00ff00", boxes_format='xyxy'):
    if draw is None:
        draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if boxes_format != 'xyxy':
        boxes = box_convert(boxes, boxes_format, 'xyxy')
    for box in boxes:
        (left, top, right, bottom) = (box[0] * im_width, box[2] * im_width, box[1] * im_height, box[3] * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                width=thickness,
                fill=color)
    return image


def draw_bounding_box_on_image(image, box, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    (left, top, right, bottom) = box
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_preds(image, boxes, labels, conf_scores, class_list, boxes_format='xyxy'):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    try:
        if platform.system() == 'Darwin':
            font = ImageFont.truetype("/System/Library/Fonts/NewYork.ttf", 17)
        else:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                   18)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    if boxes_format != 'xyxy':
        boxes = box_convert(boxes, boxes_format, 'xyxy')
    for i in range(len(boxes)):
        if labels[i] == 0: continue
    #   image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        class_name = class_list[labels[i] - 1]
        display_str = "{}: {:.2f}%".format(
            # class_names[i].decode("ascii"),
            class_name, conf_scores[i] * 100)
        color = colors[hash(class_name) % len(colors)]
        draw_bounding_box_on_image(
            image,
            tuple(boxes[i]),
            color,
            font,
            display_str_list=[display_str])
        #   np.copyto(image, np.array(image_pil))
    return image

def t2i(t: Tensor):
    return Image.fromarray(np.uint8(t)).convert("RGB")

class Vizualizer:
    def __init__(self, CLASSES, transforms=None) -> None:
        self.CLASSES = CLASSES
        self.transforms = transforms
        self.colors = list(ImageColor.colormap.values())
        try:
            if platform.system() == 'Darwin':
                font = ImageFont.truetype("/System/Library/Fonts/NewYork.ttf", 17)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                    18)
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()
        self.font = font

    def show(self, image: Tensor, bboxes, class_ids, conf=None):
        if bboxes.dim() == 1:
            assert class_ids.dim() == 0
            bboxes = bboxes.unsqueeze(0)
            class_ids = class_ids.unsqueeze(0)
            if conf is not None:
                conf = conf.unsqueeze(0)
        if self.transforms is not None:
            image, bboxes, class_ids = self.transforms(image, bboxes, class_ids)
        image = image.copy()
        self.draw_preds(image, bboxes, class_ids, conf)
        return image
    
    def draw_preds(self, image, boxes, labels, conf_scores=None, boxes_format='xyxy'):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        if boxes_format != 'xyxy':
            boxes = box_convert(boxes, boxes_format, 'xyxy')
        for i in range(len(boxes)):
            class_name = self.CLASSES[labels[i]]
            display_str = "{}".format(class_name)# class_names[i].decode("ascii"),
            if conf_scores is not None:
                display_str += ": {:.2f}%".format(conf_scores[i] * 100)
            color = self.colors[hash(class_name) % len(self.colors)]
            draw_bounding_box_on_image(
                image,
                tuple(boxes[i]),
                color,
                self.font,
                display_str_list=[display_str])
        return image