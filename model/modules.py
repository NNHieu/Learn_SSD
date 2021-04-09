from torch import Tensor
from torchvision.ops.boxes import box_convert, box_iou
from utils import decimate
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG
from math import sqrt


class DetectionConv2d(nn.Module):
    def __init__(self, in_channel, n_boxes, n_classes):
        super(DetectionConv2d, self).__init__()
        self.n_classes = n_classes
        self.conv_box = nn.Conv2d(
            in_channel, n_boxes * 4, kernel_size=3, padding=1)
        self.conv_class = nn.Conv2d(
            in_channel, n_boxes * n_classes, kernel_size=3, padding=1)
        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, fmaps: Tensor):
        batch_size = fmaps.size(0)
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        locs = self.conv_box(fmaps).permute(0, 2, 3, 1).contiguous()
        locs = locs.view(batch_size, -1, 4)

        class_scores = self.conv_class(fmaps).permute(0, 2, 3, 1).contiguous()
        class_scores = class_scores.view(batch_size, -1, self.n_classes)

        return locs, class_scores


class Stage1(nn.Module):
    def __init__(self, vgg16_features: nn.Module, num_classes=21, num_db=4):
        super(Stage1, self).__init__()
        vgg16_features = list(vgg16_features)
        vgg16_features[16] = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True)
        self.features = nn.Sequential(*vgg16_features[:23])  # 22: relu 4_3
        self.detector = DetectionConv2d(512, num_db, num_classes)
        self.pool = vgg16_features[23]

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

    def forward(self, X):
        # print("Stage1-input:", X.size())
        fmaps = self.features(X) # (N, 512, 38, 38)
        
        # Rescale conv4_3 after L2 norm
        norm = fmaps.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        rescaled_fmaps = fmaps / norm  # (N, 512, 38, 38)
        rescaled_fmaps = rescaled_fmaps * self.rescale_factors  # (N, 512, 38, 38)
        
        # print("Stage1:", fmaps.size())
        locs, class_scores = self.detector(rescaled_fmaps) 
        fmaps = self.pool(fmaps)

        return fmaps, locs, class_scores


class Stage2(nn.Module):
    def __init__(self, vgg16: VGG, num_classes=21, num_db=6):
        super(Stage2, self).__init__()
        vgg16_features = list(vgg16.features)
        vgg16_classifier = vgg16.classifier

        conv6 = nn.Conv2d(512, 1024, kernel_size=3,
                          padding=6, dilation=6)  # atrous convolution
        conv6.weight.data.copy_(decimate(vgg16_classifier[0].weight.data.view(
            4096, 512, 7, 7), m=(4, None, 3, 3)))
        conv6.bias.data.copy_(decimate(vgg16_classifier[0].bias.data, m=[4]))

        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        conv7.weight.data.copy_(decimate(vgg16_classifier[3].weight.data.view(
            4096, 4096, 1, 1), m=[4, 4, None, None]))
        conv7.bias.data.copy_(decimate(vgg16_classifier[3].bias.data, m=[4]))

        self.features = nn.Sequential(
            *vgg16_features[24:-1], # relu 5_3
            # retains size because stride is 1 (and padding)
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv6,
            nn.ReLU(True),
            conv7,
            nn.ReLU(True)
        ) 
        self.detector = DetectionConv2d(1024, num_db, num_classes)

    def forward(self, X):
        fmaps = self.features(X)  # (N, 1024, 19, 19)
        locs, class_scores = self.detector(fmaps)
        return fmaps, locs, class_scores


class AuxStage(nn.Module):
    def __init__(self, in_channel, mid, out, stride_2, padding_2, num_classes, num_db):
        super(AuxStage, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, mid, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(mid, out, kernel_size=3,
                      stride=stride_2, padding=padding_2),
            nn.ReLU(True),
        )
        self.detector = DetectionConv2d(out, num_db, num_classes)
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, X):
        fmaps = self.features(X)
        locs, class_scores = self.detector(fmaps)
        return fmaps, locs, class_scores


class GlueLayer(nn.Module):
    def __init__(self):
        super(GlueLayer, self).__init__()
        self.all_pred_locs = []
        self.all_pred_class_scores = []

    def forward(self, args):
        fmaps, locs, class_scores = args
        # print("Collect fmap :", fmaps.size())
        self.all_pred_locs.append(locs)
        self.all_pred_class_scores.append(class_scores)
        return fmaps

    def glue(self):
        # print("Glue")
        locs = torch.cat(self.all_pred_locs, dim=1)
        class_scores = torch.cat(self.all_pred_class_scores, dim=1)
        self.all_pred_locs = []
        self.all_pred_class_scores = []
        return locs, class_scores



class BoxEncoder(nn.Module):
    fmap_dims = {'stage1': 38,
             'stage2': 19,
             'stage3': 10,
             'stage4': 5,
             'stage5': 3,
             'stage6': 1}

    obj_scales = {'stage1': 0.1,
                'stage2': 0.2,
                'stage3': 0.375,
                'stage4': 0.55,
                'stage5': 0.725,
                'stage6': 0.9}

    aspect_ratios = {'stage1': [1., 2., 0.5],
                    'stage2': [1., 2., 3., 0.5, .333],
                    'stage3': [1., 2., 3., 0.5, .333],
                    'stage4': [1., 2., 3., 0.5, .333],
                    'stage5': [1., 2., 0.5],
                    'stage6': [1., 2., 0.5]}

    def __init__(self):
        super(BoxEncoder, self).__init__()
        interested_k = self.fmap_dims.keys()
        configs = [(self.fmap_dims[k], self.obj_scales[k], self.aspect_ratios[k]) for k in interested_k]
        prior_boxes = []

        for k, (fmap_dim, scale, ratios) in enumerate(configs):
            # if fmap_dim != 5: continue
            for i in range(fmap_dim):
                for j in range(fmap_dim):
                    cx = (j + 0.5) / fmap_dim
                    cy = (i + 0.5) / fmap_dim
                    for ratio in ratios:
                        prior_boxes.append(
                            [cx, cy, scale*sqrt(ratio), scale/sqrt(ratio)])
                        if ratio == 1:
                            try:
                                next_scale = configs[k + 1][1]
                                additional_scale = sqrt(scale*next_scale)
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append(
                                [cx, cy, additional_scale, additional_scale])
        prior_boxes = torch.FloatTensor(prior_boxes)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        self.cxcywh_dboxes = nn.Parameter(prior_boxes, requires_grad=False)
        self.xyxy_dboxes = nn.Parameter(box_convert(prior_boxes, 'cxcywh', 'xyxy'), requires_grad=False)
        self.nboxes = self.cxcywh_dboxes.size(0)

    def matching(self, gbs, threshold=.5):
        """
        Matching the default boxes to ground truth boxes of category

        Args: 2 set of boxes in (x1, y1, x2, y2) format.
            pbs - Tensor[num_prior, 4]
            gbs - Tensor[num_obj, 4]
        Return:
            positive_map, positive_set
        """
        overlaps = box_iou(self.xyxy_dboxes, gbs)  # [num dboxes, num obj]

        # Các trường hợp dẫn đến tồn tại một obj không được gắn với bất kì prior box nào trong tập positive
        # 1. Nó không phải là best cho bất kì prior box nào
        # 2. Các overlab của nó nhỏ hơn threshold

        best_p4g_ind = torch.argmax(overlaps, dim=0)  # [num obj]
        assert best_p4g_ind.size(0) == gbs.size(0)
        best_g4p_overlap, best_g4p_ind = torch.max(overlaps, dim=1) # [num dboxes]

        # Giải quyết TH1
        best_g4p_ind[best_p4g_ind] =  torch.arange(0, best_p4g_ind.size(0), dtype=torch.long, device=best_g4p_ind.device) 
        # Đảm bảo vượt qua bước kiểm tra threshold, Giải quyết TH2
        best_g4p_overlap[best_p4g_ind] = 1.
        # then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).

        positive_map = best_g4p_overlap > threshold
        positive_set = best_g4p_ind[positive_map]

        # matched_labels = torch.zeros(self.nboxes, dtype=torch.long)
        # matched_labels[positive_map] = glabels[positive_set]

        # matched_boxes = torch.zeros_like(self.xyxy_dboxes)
        # matched_boxes[positive_map] = gbs[positive_set]

        return positive_map, positive_set  # [num_prior, 1]

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        n_classes = predicted_scores.size(-1)
        device = predicted_locs.device

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        for i in range(batch_size):
            # print('detect ', i)
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = box_convert(self.gcxgcy_to_cxcy(predicted_locs[i]), 'cxcywh', 'xyxy')  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Check for each class
            for c in range(1, n_classes):
                # print('class', c)
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = box_iou(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool, device=device) # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box]:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress |= (overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = False

                # Store only unsuppressed boxes for this class
                non_suppress = ~suppress
                image_boxes.append(class_decoded_locs[non_suppress])
                image_labels.append(torch.LongTensor((non_suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[non_suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

    def cxcy_to_gcxgcy(self, cxcy):
        # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
        # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
        # See https://github.com/weiliu89/caffe/issues/155
        return torch.cat([(cxcy[:, :2] - self.cxcywh_dboxes[:, :2]) / (self.cxcywh_dboxes[:, 2:] / 10),  # g_c_x, g_c_y
                        torch.log(cxcy[:, 2:] / self.cxcywh_dboxes[:, 2:]) * 5], 1)  # g_w, g_h
    
    def gcxgcy_to_cxcy(self, gcxgcy):
        return torch.cat([gcxgcy[:, :2] * self.cxcywh_dboxes[:, 2:] / 10 + self.cxcywh_dboxes[:, :2],  # c_x, c_y
                        torch.exp(gcxgcy[:, 2:] / 5) * self.cxcywh_dboxes[:, 2:]], 1)  # w, h


class SSDLoss(nn.Module):
    def __init__(self, bencoder: BoxEncoder, alpha=1, neg_pos_ratio=3, threshold=0.5):
        super(SSDLoss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.bencoder = bencoder

        self.sl1 = nn.SmoothL1Loss()
        self.crossent = nn.CrossEntropyLoss(reduction='none')

    def forward(self, b_pred_prior_offsets: Tensor, b_pred_prior_classes: Tensor, b_anchors, b_labels):
        """

        """
        # print("SSDLoss - prior device:", self.prior_anchor.device)
        # self.prior_anchor = self.prior_anchor.to(b_gbs[0].device)
        device = b_pred_prior_classes.device
        batch_size = b_pred_prior_offsets.size(0)
        true_locs = torch.ones((batch_size, self.bencoder.nboxes, 4), dtype=torch.float, device=device) * self.bencoder.cxcywh_dboxes.unsqueeze(0)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size,  self.bencoder.nboxes), dtype=torch.long, device=device)  # (N, 8732)

        for i, (anchors, labels) in enumerate(zip(b_anchors, b_labels)):
            positive_map, positive_set = self.bencoder.matching(anchors, threshold=self.threshold)
            true_classes[i, positive_map] = labels[positive_set]
            true_locs[i, positive_map] = box_convert(anchors[positive_set], 'xyxy', 'cxcywh')
            true_locs[i] = self.bencoder.cxcy_to_gcxgcy(true_locs[i]) 
        positive_map = true_classes > 0 # (N, 8732)
        loc_loss = self.sl1(b_pred_prior_offsets[positive_map], true_locs[positive_map])

        nclasses = b_pred_prior_classes.size(-1)
        # Number of positive and hard-negative priors per image
        n_positives = positive_map.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        conf_loss_all = self.crossent(b_pred_prior_classes.view(-1, nclasses), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, -1)  # (N, 8732)
        
        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_map]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_map] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.arange(0, self.bencoder.nboxes, step=1, dtype=torch.int, device=device).unsqueeze(0).expand_as(conf_loss_neg) # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        
        return conf_loss, loc_loss