from torch import Tensor
from torchvision.ops import box_iou, box_convert
import torch
from math import sqrt
from torch import nn
from settings import fmap_dims, obj_scales, aspect_ratios

def matching_box(pbs: Tensor, gbs: Tensor, pb_format='cxcywh', gb_format='cxcywh', threshold=0.5) -> Tensor:
    """
    Matching the default boxes to ground truth boxes of category

    Args: 2 set of boxes in (x1, y1, x2, y2) format.
        pbs - Tensor[num_prior, 4]
        gbs - Tensor[num_obj, 4]
    Return:
        positive_map, positive_set
    """
    # print(pbs.device, gbs.device)
    xy_pbs = box_convert(pbs, pb_format, 'xyxy')
    xy_gbs = box_convert(gbs, gb_format, 'xyxy')
    # print(xy_pbs.device, xy_gbs.device)
    overlaps = box_iou(xy_pbs, xy_gbs)  # [N, M]

    # Các trường hợp dẫn đến tồn tại một obj không được gắn với bất kì prior box nào trong tập positive
    # 1. Nó không phải là best cho bất kì prior box nào
    # 2. Các overlab của nó nhỏ hơn threshold

    best_p4g_ind = torch.argmax(overlaps, dim=0)  # [M]
    assert best_p4g_ind.size(0) == gbs.size(0)
    best_g4p_overlap, best_g4p_ind = torch.max(overlaps, dim=1)

    best_g4p_ind[best_p4g_ind] = torch.LongTensor(
        range(best_p4g_ind.size(0))).to(best_g4p_ind.device)  # Giải quyết TH1
    # Đảm bảo vượt qua bước kiểm tra threshold, Giải quyết TH2
    best_g4p_overlap[best_p4g_ind] = 1.
    # then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).

    positive_map = best_g4p_overlap > threshold
    positive_set = best_g4p_ind[positive_map]
    return positive_map, positive_set  # [num_prior, 1]

def create_prior_box(interested_k=None):
    if interested_k is None:
        interested_k = fmap_dims.keys()
    configs = [(fmap_dims[k], obj_scales[k], aspect_ratios[k]) for k in interested_k]
    prior_boxes = []

    for k, (fmap_dim, scale, ratios) in enumerate(configs):
        # if fmap_dim != 5: continue
        for i in range(fmap_dim):
            for j in range(fmap_dim):
                cx = (i + 0.5) / fmap_dim
                cy = (j + 0.5) / fmap_dim
                for ratio in ratios:
                    prior_boxes.append(
                        [cx, cy, scale*sqrt(ratio), scale/sqrt(ratio)])
                    if ratio == 1:
                        try:
                            next_scale = configs[k + 1][1]
                        except IndexError:
                            next_scale = 1.
                        additional_scale = sqrt(scale*next_scale)
                        prior_boxes.append(
                            [cx, cy, additional_scale, additional_scale])
    prior_boxes = torch.FloatTensor(prior_boxes)
    prior_boxes.clamp_(0, 1)  # (8732, 4)

    return prior_boxes
