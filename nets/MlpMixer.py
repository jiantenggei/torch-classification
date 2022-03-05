#定义多层感知机
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from torchsummary import summary
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------------------------------------------------------------#
#    MLP-Mixer 网络
#    论文地址：https://arxiv.org/abs/2105.01601v1
#    我的博客 ：https://blog.csdn.net/qq_38676487/article/details/118640096
#-------------------------------------------------------------------------#
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            #由此可以看出 FeedForward 的输入和输出维度是一致的
            nn.Linear(dim,hidden_dim),
            #激活函数
            nn.GELU(),
            #防止过拟合
            nn.Dropout(dropout),
            #重复上述过程
            nn.Linear(hidden_dim,dim),

            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self,dim,num_patch,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),   #这里是[batch_size, num_patch, dim] -> [batch_size, dim, num_patch]
            FeedForward(num_patch,token_dim,dropout),
            Rearrange('b d n -> b n d')    #[batch_size, dim, num_patch] -> [batch_size, num_patch, dim]

         )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
    def forward(self,x):

        x=x+self.token_mixer(x)

        x=x+self.channel_mixer(x)

        return x

class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,num_classes,patch_size,image_size,depth,token_dim,channel_dim,dropout=0.):
        super().__init__()
        assert image_size%patch_size==0
        self.num_patches=(image_size//patch_size)**2
        #embedding 操作，用卷积来分成一小块一小块的
        self.to_embedding=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=dim,kernel_size=patch_size,stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        #经过Mixer Layer 的次数
        self.mixer_blocks=nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim,self.num_patches,token_dim,channel_dim,dropout))
        self.layer_normal=nn.LayerNorm(dim)

        self.mlp_head=nn.Sequential(
            nn.Linear(dim,num_classes)
        )
    def forward(self,x):
        x=self.to_embedding(x)
        for mixer_block in self.mixer_blocks:
            x=mixer_block(x)
        x=self.layer_normal(x)
        x=x.mean(dim=1)

        x=self.mlp_head(x)

        return x

#测试Mlp-Mixer
if __name__ == '__main__':    
      model = MLPMixer(in_channels=3, dim=512, num_classes=1000, patch_size=16, image_size=224, depth=1, token_dim=256,
                     channel_dim=2048).to(device)
      summary(model,(3,224,224))
