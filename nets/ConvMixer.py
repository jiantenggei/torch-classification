import torch
import torch.nn as nn
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------------------------------------------------------------#
#    Conv-Mixer 网络
#    论文地址：https://openreview.net/pdf?id=TVHS5Y4dNvM
#    我的博客 ：https://blog.csdn.net/qq_38676487/article/details/120705254
#-------------------------------------------------------------------------#
class ConvMixerLayer(nn.Module):
    def __init__(self,dim,kernel_size = 9):
        super().__init__()
        #残差结构
        self.Resnet =  nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        #逐点卷积
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
    def forward(self,x):
        x = x +self.Resnet(x)
        x = self.Conv_1x1(x)
        return x

class ConvMixer(nn.Module):
    def __init__(self,dim,depth,kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(3,dim,kernel_size=patch_size,stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks =nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim,kernel_size=kernel_size))

        self.head =  nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim,n_classes)
        )
        
    def forward(self,x):
        x = self.conv2d1(x)

        for ConvMixer_block in  self.ConvMixer_blocks:
             x = ConvMixer_block(x)
        
        x = self.head(x)

        return x
#-----------------------------------------------------------------------#
#    论文中给出的配置：
#    ConvMixer_h_d   h：dim 隐藏层维度  d：depth 网络深度
#-------------------------------------------------------------------------#
def ConvMixer_1536_20(n_classes = 1000):
    return ConvMixer(dim=1536,depth=20,patch_size=7,kernel_size=9,n_classes=n_classes)

def ConvMixer_768_32(n_classes = 1000):
    return ConvMixer(dim=768,depth=32,patch_size=7,kernel_size=7,n_classes=n_classes)

# 自定义的 ConvMixer 不传参 为 ConvMixer_768_32
def custom_ConvMixer(dim=768,depth=32,patch_size=7,kernel_size=7,n_classes=1000):
     return ConvMixer(dim=dim,depth=depth,patch_size = patch_size,kernel_size=kernel_size,n_classes=n_classes)

if __name__ == '__main__':
    model =ConvMixer_1536_20().to(device)
    summary(model, (3, 224, 224))