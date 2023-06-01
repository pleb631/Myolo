import torch
from torch import nn
from utils import *

class CAM(nn.Module):
    def __init__(self, inc, fusion='weight'):
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion
        
        self.conv1 = Conv(inc, inc, 3, 1, None, 1, 1)
        self.conv2 = Conv(inc, inc, 3, 1, None, 1, 3)
        self.conv3 = Conv(inc, inc, 3, 1, None, 1, 5)
        
        self.fusion_1 = Conv(inc, inc, 1)
        self.fusion_2 = Conv(inc, inc, 1)
        self.fusion_3 = Conv(inc, inc, 1)

        if self.fusion == 'adaptive':
            self.fusion_4 = Conv(inc * 3, 3, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2) + self.fusion_3(x3)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
            x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight
        else:
            return torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)


class PCRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(3, 512 * 3, kernel_size=1)
        self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.mcrc = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )
        self.acrc = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.C1(x)
        x2 = self.mcrc(x1)
        x3 = self.acrc(x1)
        return self.R1(x2) + self.R1(x3)

# FRM模块实现输入x为tensor列表形式
class FRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.R1 = nn.Upsample(None, 2, 'nearest')  #上采样扩充2倍采用邻近扩充
        self.R3 = nn.MaxPool2d(kernel_size=2, stride=2)   #下采样使用最大池化
        self.C1 = nn.Conv2d(1024 + 512 + 256, 3, kernel_size=1)

        self.C2 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        self.C3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.C4 = nn.Conv2d(1, 256, kernel_size=1, stride=1)
        self.C5 = nn.Conv2d(1, 512, kernel_size=1, stride=1)
        self.C6 = nn.Conv2d(1, 1024, kernel_size=1, stride=1)
        self.pcrc = PCRC()

    def forward(self, x):

        x0 = self.R1(x[0])
        x2 = self.R3(x[2])
        x1 = self.C1(torch.cat((x0, x[1], x2), 1))
        Conv_1_1 = torch.split(torch.softmax(x1, dim=0), 1, 1)       #第一维度1为步长进行分割
        Conv_1_2 = torch.split(self.pcrc(x1), 512, 1)                #第一维度512为步长进行分割
        y0 = (x0 * self.C6(Conv_1_1[0])) + (self.C2(Conv_1_2[0]) * x0)
        y1 = (x[1] * self.C5(Conv_1_1[1])) + (Conv_1_2[1] * x[1])
        y2 = (x2 * self.C4(Conv_1_1[2])) + (self.C3(Conv_1_2[2]) * x2)

        y0 = self.R3(y0)
        y2 = self.R1(y2)

        return [y0, y1, y2]


"""参考

backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0
   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   [-1, 1, Bottleneck, [64]],
   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
   [-1, 2, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 5-P3/8
   [-1, 8, Bottleneck, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 8, Bottleneck, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 9-P5/32
   [-1, 4, Bottleneck, [1024]],  # 10
  ]

# YOLOv3 head
head:
  [[-1, 1, CAM,  [1024]],      #CAM层为新增no.11,相关实现在common.py进行
   [-2, 1, Bottleneck, [1024, False]],
   [-1, 1, Conv, [512, [1, 1]]],
   [-1, 1, Conv, [1024, 3, 1]],
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [1024, 3, 1]],  # 16 (P5/32-large)
   [[-1, 11], 1, Conadd, [1]],   # 17 Conadd为新增层，相关实现在common.py进行,沿通道方向进行级联 11层与17层进行add操作后送入detect层

   [-3, 1, Conv, [256, 1, 1]],  #第15层与18层进行连接
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 8], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Bottleneck, [512, False]],
   [-1, 1, Bottleneck, [512, False]],
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],  # 22 (P4/16-medium)

   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Bottleneck, [256, False]],
   [-1, 2, Bottleneck, [256, False]],  # 27 (P3/8-small)
   [[17, 24, 29],1,FRM,[]] #30

   [[30], 1, Detect, [nc, anchors]],   # Detect(P3, P4, P5) 27 22 15   修改检测层15改为17  后面的递增2层
  ]
"""