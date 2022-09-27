import torch
import torch.nn as nn
from .attention import MultiHeadAttention
# from .DSA_attention import DSAMultiHeadAttention
from .attention import PositionWiseFeedForward

class DepthwiseConv(nn.Module):
    def __init__(self,in_ch,out_ch, kernel_size):
        super(DepthwiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=((kernel_size - 1 ) // 2),
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_k=32, d_v=32, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        # self.self_att = DSAMultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        # self.dropout1=nn.Dropout(dropout)
        # self.lnorm1=nn.LayerNorm(d_model)

        self.dropout2=nn.Dropout(dropout)
        self.lnorm2=nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.mask_fcs = nn.ModuleList()
        for _ in range(3):
            self.mask_fcs.append(
                nn.Linear(d_model, d_model, bias=False))
            self.mask_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.mask_fcs.append(nn.ReLU(inplace=True))

        self.fc_mask = nn.Linear(d_model, d_model)


    def forward(self, input, enc_output, input_map, enc_map, R):
        #MHA+AddNorm

        b = input.shape[0]
        h, w = input_map

        '''
        self_att = self.self_att(input, input, input, input_map)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        '''
        
        # MHA+AddNorm
        enc_att = self.enc_att(input, enc_output, enc_output, enc_map, [h, w], R)
        enc_att = self.lnorm2(input + self.dropout2(enc_att))

        # FFN+AddNorm
        ff = self.pwff(enc_att)

        for reg_layer in self.mask_fcs:
            ff = reg_layer(ff)
        ff = self.fc_mask(ff)
        return ff