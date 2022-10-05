from email.errors import NonPrintableDefect
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .decoder import DecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MaskDecoder(nn.Module):
    def __init__(self, N_dec, d_model=256, d_k=32, d_v=32, h=8, d_ff=2048, dropout=.0):
        super(MaskDecoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.mask_activate = nn.Sigmoid()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        '''
        self.dynamic_fc = nn.ModuleList(
            [nn.Linear(d_model, d_model+1) for _ in range(N_dec)])
        '''
        self.N = N_dec
        self.h = h

    def _map(self, feature, kernel, i):
        
        b = feature.shape[0]
        h, w = feature.shape[2], feature.shape[3]
        n = kernel.shape[1]

        # kernel = self.mask_fcs[i](kernel).view(b, n, 9, -1)
        # temp = self.dynamic_fc[i](kernel)
        # linear, bias = temp[..., :-1], temp[..., -1]
        mask = torch.einsum('bchw,bnc->bnhw', feature, kernel)
        # mask = mask + bias.unsqueeze(dim=-1).unsqueeze(dim=-1)

        return self.mask_activate(mask), mask

    def forward(self, inputs, encoder_output, gt_mask, attn_map, box):
        # input (b, n, c)
        # encoder_output  b, c, h, w
        # mask b, n, h, w

        out = inputs

        masks = []
        boxes = []
        b = inputs.shape[0]
        n = inputs.shape[1]
        c = encoder_output.shape[1]
        h = encoder_output.shape[2]
        w = encoder_output.shape[-1]
        
        '''
        i, j = torch.meshgrid(torch.arange(h), torch.arange(w))
        indexes = torch.stack([i.flatten(), j.flatten()]) * 1.0
        R = torch.sqrt(((indexes.transpose(0, 1).unsqueeze(-1) - indexes.unsqueeze(0)) ** 2).sum(1))
        R = ((4 - torch.clamp(R, 0.00001, 4)) * 0.25).cuda()
        '''

        for i, l in enumerate(self.layers):
            
            
            if attn_map is None:
                attn_map = None
            else:
                attn_map = (attn_map.view(b, n, -1).unsqueeze(dim=1).expand(b, self.h, n, h*w)) < 0.5
            

            temp_box, mask_kernel, out = l(out, encoder_output.view(b, -1, h*w).permute(0, 2, 1), [h, w], attn_map, None)
            attn_map, mask = self._map(encoder_output, mask_kernel, i)

            box = box + torch.exp(temp_box)
            masks.append(self.upsample2(mask))
            boxes.append(box)

        return out, masks, boxes
