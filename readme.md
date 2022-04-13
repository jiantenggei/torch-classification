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

在eval.py 中:

```python
if __name__ == "__main__":

    # 读取测试集路劲和标签
    with open("./cls_test.txt","r") as f: 
        lines = f.readlines()
    #---------------------------------------------------#
    #   权重和模型
    #   注意：训练时设置的模型需要和权重匹配，
    #   也就是训练的啥模型使用啥权重
    #---------------------------------------------------#
    model_path = '' #训练好的权重路径
    model = ConvMixer_768_32(n_classes=2) # 自己训练好的模型

    mode = load_dict(model_path,model) # 加载权重
    eval = eval_top(anno_lines=lines,model=model)
    #---------------------------------------------------#
    #   top1 预测概率最好高的值与真实标签一致 √
    #   top5 预测概率前五个值由一个与真实标签一致 √
    #---------------------------------------------------#
    print('start eval.....')
    top1 = eval.eval_top1()
    
    top5 = eval.eval_top5()
    print('top1:%.3f,top5:%3.f'%(top1,top5))
    print('Eval Finished')

```



# How2Predict

predict.py 中，设置好模型和权重，控制台输入图片路径。

```python
#加载模型
model_path = 'logs\ep050-loss0.414-val_loss0.376.pth'
model = ConvMixer_768_32(n_classes=2)
model = load_dict(model_path,model)
eval = eval_top(anno_lines=None,model=model)

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = eval.detect_img(image,mode='predict')
        print(class_name)

```

控制台：

```bash
Loading weights into state dict...
Input image filename:d:\Classification\torch\datasets\test\cats\cat.4006.jpg 
```


## 训练技巧和练丹

+ [ ] Focl_loss(样本不均衡策略)

+ [x] label_smoothing (训练样本偏少时，防止过拟合策略)
+ [x] 学习率衰减(使模型收敛更充分) 

存在bug及其他问题私信：1308659229@qq.com

# 其他

该仓库可能存在bug，希望大家在使用过程中能及时反馈，或者留下一些代码修改意见。我们一起让它变得更好。



**如果觉得有用清给我点star**
