from cProfile import label
import numpy as np
from config import *
import torch.backends.cudnn as cudnn
from utils.train_one_epoch import fit_one_epoch
from utils.dataloader import DataGenerator,detection_collate
from utils.utils import get_classes,weights_init,create_tbWriter
from nets.ConvMixer import ConvMixer, ConvMixer_768_32,custom_ConvMixer
from nets.MlpMixer import MLPMixer
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.training_utils import cross_entropy,smooth_one_hot

def train():
    # 获得分类数
    class_names, num_classes = get_classes(classes_path)
    #---------------------------------------------------#
    #  定义模型，可在nets 导入自己的模型去训练，
    #  目前支持MLP-Mixer Conv-Mixer ConvNeXt系列模型
    #  只有ConvNeXt 支持pretrain 官方提供的权重
    #---------------------------------------------------#
    model = custom_ConvMixer(dim=128,depth=12,patch_size=7,kernel_size=7,n_classes=num_classes)
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
    
    tb_writer = create_tbWriter(log_dir=log_dir)
    # 设置loss 
    if label_smoothing:
        criterion = cross_entropy
    else:
        criterion = nn.CrossEntropyLoss()
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val

    #配置训练参数
    lr              = 1e-3
    Batch_size      = 32
    Init_Epoch      = 0
    Freeze_Epoch    = 50

    epoch_step      = num_train // Batch_size
    epoch_step_val  = num_val // Batch_size

    if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
    optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

    train_dataset   = DataGenerator(lines[:num_train], input_shape, True)
    val_dataset     = DataGenerator(lines[num_train:], input_shape, False)
    gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
    gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

    for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model_train, model, tb_writer, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_Epoch, Cuda,criterion)
            lr_scheduler.step()







if __name__ == "__main__":
    train()

   