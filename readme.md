# classification
Pytorch 用于训练自己的图像分类模型

# 环境要求
pytorch
opencv

# conda虚拟环境一键导入：
```bash
conda env create -f torch.yaml
```

# 理论地址：
csdn 博客地址：
 MLP-Mixer：https://no-coding.blog.csdn.net/article/details/121682740

 Conv-Mixer：https://blog.csdn.net/qq_38676487/article/details/120705254

 ConvNeXt：https://blog.csdn.net/qq_38676487/article/details/123298605


# How2Train
   ## 	1.数据集:

```bash
─dataset
    ├─train
    │	└─cats
    │		└─xxjj.jpg
    │	└─dogs
    │	 	└─xxx.jpg
    ├─test
    │	└─cats
    │	└─dogs
```

classes.txt:

```
cats
dogs
```

在txt_annotation.py 中 calsses 与上述文件classes.txt 分类顺序一致，运行txt_annotation.py  生成 cls_train.txt, cls_text.txt

```
classes = ["cats", "dogs"]
sets    = ["train", "test"]
```

## 	2.训练

​	在config 中配置训练参数：

```python
Cuda             = False  #是否使用GPU 没有为Flase

input_shape      = [224,224]  # 输入图片大小


batch_size      = 2 # 自己可以更改
lr              = 1e-3         

classes_path    = 'classes.txt'


num_workers     = 0  # 是否开启多进程


annotation_path     = 'cls_train.txt'  



val_split       = 0.1  #验证集比率


resume          =''  # 加载训练权重路径

log_dir         = 'logs' # 日志路径 tensorboard 保存
```

在 trian.py 中

```python
#---------------------------------------------------#
#  定义模型，可在nets 导入自己的模型去训练，
#  目前支持MLP-Mixer Conv-Mixer ConvNeXt系列模型
#  只有ConvNeXt 支持pretrain 官方提供的权重
#---------------------------------------------------#
 model = ConvMixer_768_32(n_classes=num_classes)
```

## 日志查看

由于每次启动训练时，会在logs 文件下按照时间创建一个日志文件。如：

```bash
tensorboard --logdir=logs\loss_2022_03_06_12_11_30
```

# How2Eval

实验室电脑被项目占用中。。。。。

还未调试
# How2Predict

实验室电脑被项目占用中。。。。。

还未调试


## 训练技巧和练丹

​		losses: Focl_loss(样本不均衡策略)

​		

存在bug及其他问题私信：1308659229@qq.com

**如果觉得有用清给我点star**
