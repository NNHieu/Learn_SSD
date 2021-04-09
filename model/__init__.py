from torch import nn
from .modules import Stage1, Stage2, AuxStage, GlueLayer
from torch import Tensor
from torchvision.models import vgg16
from .modules import BoxEncoder
from .compare import *

class StageSSD300(nn.Module):

    aux_configs = {
        'stage3': (1024, 256, 512, 2, 1),
        'stage4': (512, 128, 256, 2, 1),
        'stage5': (256, 128, 256, 1, 0),
        'stage6': (256, 128, 256, 1, 0),
    }

    def __init__(self, num_classes, vgg16_pretrained=False):
        super(StageSSD300, self).__init__()
        aspect_ratios = BoxEncoder.aspect_ratios
        vgg16_model = vgg16(pretrained=vgg16_pretrained)
        aux_stages = [AuxStage(*self.aux_configs[k], num_classes, 1 +
                               len(aspect_ratios[k])) for k in self.aux_configs.keys()]
        self.glue_layer = GlueLayer()
        self.stages = [Stage1(vgg16_model.features, num_classes=num_classes,
                              num_db=1 + len(aspect_ratios['stage1'])),
                       self.glue_layer,
                       Stage2(vgg16_model, num_classes=num_classes,
                              num_db=1 + len(aspect_ratios['stage2'])),
                       self.glue_layer
                       ]
        for stage in aux_stages:
            self.stages += [stage, self.glue_layer]
        self.stages = nn.Sequential(*self.stages)


    def forward(self, images: Tensor):
        """
        

        Args:
            image : batch image - [batch, 3, 300, 300]

        Return: Depend on phase
            test phase:
                pred: class_label, confident, loc - [batch, topk, 6]

            train phase::
                localization - [batch, num_default_box, 4]
                classification - [batch, num_default_box, num_class]
        """
        self.stages(images)
        locs, class_scores = self.glue_layer.glue()
        return locs, class_scores

