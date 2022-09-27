import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .semantic_fpn_wrapper_new import SemanticFPNWrapper
from .TextVisualEncoder import MaskDecoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# attnetion map, decoder结尾过线性层
class MainModule(nn.Module):
    def __init__(self) -> None:
        super(MainModule, self).__init__()

        self.encoder = MaskDecoder(3, d_model=256)
        self.localization_fpn = SemanticFPNWrapper()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.activate = nn.ReLU()
        self.text_linear = nn.Linear(768, 256)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def forward(self, image, text, image_mask):
        
        b, n = text.shape[0], text.shape[1]

        image_layer = self.localization_fpn(image)

        h, w = image_layer.shape[2], image_layer.shape[3]

        text = self.activate(self.text_linear(text))

        mask = torch.nn.Sigmoid()(torch.einsum('bchw,bnc->bnhw', image_layer, text))
        # image_mask = torch.ones((b, n, h, w)).cuda()
        # text_mask = ann_types == 0

        # text_mask = text_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        # text_mask = text_mask.repeat([1, 8, 1, 1])
        
        out, masks = self.encoder(text, image_layer, None, mask)

        return image_layer, out, masks




