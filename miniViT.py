import torch
import torch.nn as nn

from models.layers import PatchTransformerEncoder,PixelWiseDotProduct
import torch.nn.functional as F

import os

import torch
import torch.nn as nn
import torchvision.models
import collections
import math

# from network.backbone import resnet101


def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()




class SceneUnderstandingModule(nn.Module):
    def __init__(self,in_channels):
        super(SceneUnderstandingModule, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True))
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True))
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True))
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True))

        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True))

        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(128 * 4, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(128, 128, 1),  # KITTI 142 NYU 136 In paper, K = 80 is best, so use 160 is good!
        )

        weights_init(self.modules(), type='xavier')

    def forward(self, x):
        print('x.size', x.size())
        x5 = self.aspp4(x)

        x4 = self.aspp3(x5)
        x4 = self.aspp1(x4)

        x3 = self.aspp2(x)
        x3 = self.aspp1(x3)

        x2 = self.aspp1(x)
        # x2 = self.aspp1(x3)
        # print("x2",x2.shape)
        # x3 = self.aspp2(x4)
        # print("x3",x3.shape)
        # x4 = self.aspp3(x5)
        # print("x4",x4.shape)
        # x5 = self.aspp4(x)
        # print("x5",x5.shape)
        x1 = self.conv3x3(x2)
        print("shapes", x2.size(), x3.size(), x4.size(), x5.size())
        x6 = torch.cat((x1,x2,x3,x4), dim=1)
        print("x6",x6.shape)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        out = F.interpolate(out, size=(240, 320), mode="bilinear", align_corners=True)
        print("out",out.shape)
        return out

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1) #SceneUnderstandingModule() 
        # self.conv3x3 = nn.Sequential(
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout2d(p=0.5),
        #     # nn.Conv2d(128, 128, 2),  # KITTI 142 NYU 136 In paper, K = 80 is best, so use 160 is good!
        #     # nn.UpsamplingBilinear2d(scale_factor=8)
        #     nn.UpsamplingBilinear2d(size=(240, 320))
        # )
        self.convaspp = SceneUnderstandingModule(in_channels) 
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        # x1 = torch.flatten(x)
        # x1 = x1[:6082560]
        # x1 = x1.view((2,2048,33,45))
        x_in = x
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E
        print("x_in",x.shape)
        print("tgt",tgt.shape)
        # x_in = self.conv3x3(x_in)
        # x = x[:,:128,:240,:320]
        # print("x_in_3*3",x.shape)
        # x = self.conv3x3(x)
        # print("x_out_3*3",x.shape)
        print("x_in_app",x_in.shape)
        xapp = self.convaspp(x_in)
        x_in = self.conv3x3(x_in)
        x_in = self.conv3x3(x_in)
        # print("xapp",xapp.shape)
        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]
        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(xapp, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
