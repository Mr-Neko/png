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

        self.box_linear = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True),
            nn.ReLU()
        )

        self.mask_linear = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True),
            nn.ReLU()
        )

        self.box_ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.LayerNorm((4,), eps=1e-05, elementwise_affine=True),
            nn.Sigmoid()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def forward(self, image, text, image_mask):
        
        b, n = text.shape[0], text.shape[1]

        image_layer = self.localization_fpn(image)

        h, w = image_layer.shape[2], image_layer.shape[3]

        text = self.text_linear(text)

        mask_kernel = self.mask_linear(text)
        box_kernel = self.mask_linear(text)

        mask = torch.einsum('bchw,bnc->bnhw', image_layer, mask_kernel)
        box = self.box_ffn(box_kernel)
        # image_mask = torch.ones((b, n, h, w)).cuda()
        # text_mask = ann_types == 0

        # text_mask = text_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        # text_mask = text_mask.repeat([1, 8, 1, 1])
        
        out, masks, boxes = self.encoder(text, image_layer, None, torch.nn.Sigmoid()(mask), box)

        masks.insert(0, self.upsample2(mask))
        boxes.insert(0, box)
        return image_layer, out, masks, boxes




