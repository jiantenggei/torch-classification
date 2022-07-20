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

class eval_top:
    
    def __init__(self,anno_lines,model) -> None:

        self.anno_lines = anno_lines
        self.model = model
        self.class_name,_ = get_classes(classes_path)
    
    def detect_img(self,image,mode='predict'):

        #  image -> RGB
        image = cvtColor(image)

        image_data = letterbox_image(image,input_shape)

        # 归一化
        image_data = np.array(image_data, np.float32)
        image_data = image_data/127.5
        image_data -= 1.0

        
        # 添加bacth_size 维度
        image_data = np.expand_dims(image_data,0)
        #[batch_size,width,height,channel] -> [batch_size,channel,width,height]
        image_data = np.transpose(image_data,(0, 3, 1, 2))

        with torch.no_grad():
            img = torch.from_numpy(image_data)
            if Cuda:
                img = img.cuda()
                self.model.cuda()
            
            pred = torch.softmax(self.model(img)[0],dim=-1).cpu().numpy()
        
        name = self.class_name[np.argmax(pred)]

        # 预测
        if mode == 'predict':
            probability = np.max(pred)
            #---------------------------------------------------#
            #   绘图并写字
            #---------------------------------------------------#
            plt.subplot(1, 1, 1)
            plt.imshow(np.array(image))
            plt.title('Class:%s Probability:%.3f' %(name, probability))
            plt.show()
            return name
        #top1
        elif mode == 'top1':
            return np.argmax(pred)
        ##top5
        elif mode == 'top5' :
            arg_pred = np.argsort(pred)[::-1]
            arg_pred_top5 = arg_pred[:5]
            return arg_pred_top5

    #---------------------------------------------------#
    #   eval_top1
    #---------------------------------------------------#
    def eval_top1(self):
        print('Eval Top1....')
        correct = 0
        total = len(self.anno_lines)
        with tqdm(total=total,postfix=dict,mininterval=0.3) as pbar:
            for idx,line in enumerate(self.anno_lines):
                annotation_path = line.split(';')[1].split()[0]
                x = Image.open(annotation_path)
                y = int(line.split(';')[0])

                pred = self.detect_img(x,mode='top1')
                correct += pred == y
                pbar.update(1)
        return correct / total

    #---------------------------------------------------#
    #   eval_top5 更新进度条
    #---------------------------------------------------#
    def eval_top5(self):
        correct = 0
        total = len(self.anno_lines)
        print('Eval Top5....')
        with tqdm(total=total,postfix=dict,mininterval=0.3) as pbar:
            for idx,line in enumerate(self.anno_lines):
                annotation_path = line.split(';')[1].split()[0]
                x = Image.open(annotation_path)
                y = int(line.split(';')[0])

                pred = self.detect_img(x,'top5')
                correct += y in pred
                pbar.update(1)
        return correct / total    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 读取测试集路劲和标签
    with open("./cls_test.txt","r") as f: 
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
    
    
    dataset     = DataGenerator(lines, input_shape, False,is_grayscale=True)
    
    gen_val         = DataLoader(dataset, batch_size=128, num_workers=0, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

    print(evaluate(model=model,data_loader=gen_val,epoch=0))