import imp
from msilib import type_binary
from statistics import mode
import numpy as np
import torch
from PIL import Image
import datetime
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
#---------------------------------------------------#
#   转化为RGB 格式
#---------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   初始化权重
#---------------------------------------------------#
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


#---------------------------------------------------#
#   创建一个文件下，以当前日期为文件名 记录logs
#---------------------------------------------------#
def create_tbWriter(log_dir:str):

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')

    save_path  = os.path.join(log_dir, "loss_" + str(time_str))

    os.makedirs(save_path)

    tb_writer = SummaryWriter(log_dir=save_path)

    return tb_writer

#---------------------------------------------------#
#   加灰条的resize 防止图片是真
#---------------------------------------------------#
def letterbox_image(image, size):
    iw, ih = image.size
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image
    
#---------------------------------------------------#
#   加载权重
#---------------------------------------------------#
def load_dict(model_path,model):
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

if __name__=='__main__':
    # 测试tb_writer
    t = create_tbWriter('logs')

    # image = Image.open(r'C:\Users\Jian\Pictures\Camera Roll\\1.jpg')
    # image = cvtColor(image)
    # image = letterbox_image(image,(2224,2225))
    # image = np.array(image)
    # cv2.imshow('c',image)
    # cv2.waitKey(0)


    