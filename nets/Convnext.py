from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_,DropPath
from timm.models.registry import register_model
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------------------------------------------------------------#
#    ConvNeXt 网络
#    论文地址： https://arxiv.org/pdf/2201.03545.pdf
#    我的博客 ：https://blog.csdn.net/qq_38676487/article/details/123298605
#-------------------------------------------------------------------------#
class Block(nn.Module):
    #-----------------------------------------------------------------------#
    # ConvNeXt Block 块 两种实现方式
    #   (1) 深度可分离卷积 + 1x1 的卷积
    #       DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    #   (2)深度可分离卷积 + Linear 全连接来代替 1x1 卷积 ，发现在pytorch 更快
    #        DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    # 参数：
    #   dim：维度  drop_path：0~1  layer_scale_init_value：
    #-------------------------------------------------------------------------#
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # 深度课分离卷积
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # 用全连接代替1x1的卷积
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 一个可学习的参数
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    #-----------------------------------------------------------------------#
    #    自定义 LayerNorm 默认channels_last
    #    channels_last  [batch_size, height, width, channels]
    #    channels_first [batch_size, channels, height, width]
    #-------------------------------------------------------------------------#
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()


        #保存stem 和下采样
        self.downsample_layers = nn.ModuleList() # 
        #[batch_size,3,224,224] -> [batch_size,dim[0],56,56]
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem)


        #下采样 -> 用2x2 步长为2 的卷积来代替池化
        # 这里一次将所有stage的下采样放入 
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        #添加stage
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        #最后的分类输出
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


#-----------------------------------------------------------------------#
#     论文中的 model， 以及其预训练权重
#     model.head = torch.nn.Linear(768,num_classes) 
#     加载预训练权重后 仍然可以调整分类数
#     训练数据需要是三通道彩图
#-------------------------------------------------------------------------#
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, num_classes=1000,**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
        #更改model head 使其能够符合自己的分类数
        model.head = torch.nn.Linear(768,num_classes)
    return model


@register_model
def convnext_small(pretrained=False,num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = torch.nn.Linear(768,num_classes)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False,num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = torch.nn.Linear(1024,num_classes)
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False,num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = torch.nn.Linear(1536,num_classes)
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        url = model_urls['convnext_xlarge_22k'] if in_22k else model_urls['convnext_xlarge_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = torch.nn.Linear(2048,num_classes)
    return model

if __name__ == '__main__':
    # 测试自定义分类数
    model_tiny  =convnext_tiny(pretrained=False,num_classes=2).to(device)
    summary(model_tiny, (3, 48, 48))
    # model_xlarge =convnext_xlarge(pretrained=False,num_classes=2).to(device)
    # summary(model_xlarge, (3, 224, 224))
    # model_large =convnext_large(pretrained=False,num_classes=2).to(device)
    # summary(model_large, (3, 224, 224))
    # model_base  =convnext_base(pretrained=False,num_classes=2).to(device)
    # summary(model_base, (3, 224, 224))
    # model_small =convnext_small(pretrained=False,num_classes=2).to(device)
    # summary(model_small, (3, 224, 224))
    

        