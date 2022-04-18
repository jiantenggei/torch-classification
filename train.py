from cProfile import label
import numpy as np
from config import *
import torch.backends.cudnn as cudnn
from utils.train_one_epoch import fit_one_epoch
from utils.dataloader import DataGenerator,detection_collate
from utils.utils import get_classes,weights_init,create_tbWriter
from nets.resnet import ResNet18
from nets.ConvMixer import ConvMixer_768_32
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.training_utils import cross_entropy,smooth_one_hot
import config
import torch.utils.model_zoo as model_zoo

def train():
    # 获得分类数
    class_names, num_classes = get_classes(classes_path)
    #---------------------------------------------------#
    #  定义模型，可在nets 导入自己的模型去训练，
    #  目前支持MLP-Mixer Conv-Mixer ConvNeXt系列模型
    #  只有ConvNeXt 支持pretrain 官方提供的权重
    #---------------------------------------------------#
    model = ConvMixer_768_32(n_classes=num_classes)
    #初始化
    if resume == '':
        #---------------------------------------------------#
        # 初始模型的方式： str: normal xavier kaiming orthogonal
        #---------------------------------------------------#
        weights_init(model,init_type='normal')
    
    else:
        #载入训练过的权重
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #配置训练
    model_train = model.train()

    # 使用gpu
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    #这里我将创建一个文件下 将所有的配置参数都记录下来
    #hyperparameters = 
    tb_writer = create_tbWriter(log_dir=log_dir)
    # 设置loss 
    if label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing_value)
    else:
        criterion = nn.CrossEntropyLoss()
    #读取 train
    with open(train_annotation_path, "r") as f:
        lines = f.readlines()
    
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    with open(val_annotation_path,'r') as f:
        val_lines = f.readlines()
    # num_val     = int(len(lines) * val_split)
    num_val = len(val_lines)
    num_train   = len(lines)

    #配置训练参数
    lr              = config.lr
    Batch_size      = config.batch_size
    Init_Epoch      = 0
    End_Epoch    = 50

    epoch_step      = num_train // Batch_size
    epoch_step_val  = num_val // Batch_size

    if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    
    optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)

    if  config.scheduler== 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.95,verbose=True)

    if config.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=End_Epoch)
  
   
    train_dataset   = DataGenerator(lines, input_shape, True,is_grayscale=is_grayscale)
    val_dataset     = DataGenerator(val_lines, input_shape, False,is_grayscale=is_grayscale)
    gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
    gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

    for epoch in range(Init_Epoch,End_Epoch):
        train_loss,train_accuracy,val_loss,val_accuracy=fit_one_epoch(model_train, model, tb_writer, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, End_Epoch, Cuda,criterion)
               #学习率 衰减
        if config.scheduler == 'reduce':
            scheduler.step(val_accuracy)

        if config.scheduler == 'cos':
            scheduler.step()
        
            


if __name__ == "__main__":
    train()

   