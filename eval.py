from unicodedata import name
import matplotlib.pyplot as plt
import torch
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import cvtColor,letterbox_image,get_classes,load_dict
from config import input_shape,Cuda,classes_path # 来源于config.py 中的Cuda
import numpy as np
from PIL import Image
from nets.ConvMixer import ConvMixer_768_32
from tqdm import tqdm
import cv2   
import sys
@torch.no_grad()
def evaluate(model, data_loader, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    model.to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images  = torch.from_numpy(images).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.FloatTensor).long()
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 读取测试集路劲和标签
    with open("./cls_val.txt","r") as f: 
        lines = f.readlines()
    #---------------------------------------------------#
    #   权重和模型
    #   注意：训练时设置的模型需要和权重匹配，
    #   也就是训练的啥模型使用啥权重
    #---------------------------------------------------#
    model_path = 'logs\ep300-loss0.990-val_loss1.105.pth' #训练好的权重路径
    from nets.resnet import ResNet18
    model = ResNet18()

    model = load_dict(model_path,model)
    
    
    val_dataset     = DataGenerator(lines, input_shape, False,is_grayscale=True)
    
    gen_val         = DataLoader(val_dataset, batch_size=128, num_workers=0, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

    print(evaluate(model=model,data_loader=gen_val,epoch=0))