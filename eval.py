from unicodedata import name
import matplotlib.pyplot as plt
import torch
from utils.utils import cvtColor,letterbox_image,get_classes,load_dict
from config import input_shape,Cuda,classes_path # 来源于config.py 中的Cuda
import numpy as np
from PIL import Image
from nets.ConvMixer import ConvMixer_768_32

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
        #
        elif mode == 'top1':
            return np.argmax(pred)

        elif mode == 'top5' :
            arg_pred = np.argsort(pred)[::-1]
            arg_pred_top5 = arg_pred[:5]
            return arg_pred_top5

    #---------------------------------------------------#
    #   eval_top1
    #---------------------------------------------------#
    def eval_top1(self):
        correct = 0
        total = len(self.anno_lines)
        for idx,line in enumerate(self.anno_lines):
            annotation_path = line.split(';')[1].split()[0]
            x = Image.open(annotation_path)
            y = int(line.split(';')[0])

            pred = self.detect_img(x,mode='top1')
            correct += pred == y
        return correct / total

    #---------------------------------------------------#
    #   eval_top5
    #---------------------------------------------------#
    def eval_top5(self):
        correct = 0
        total = len(self.anno_lines)
        for idx,line in enumerate(self.anno_lines):
            annotation_path = line.split(';')[1].split()[0]
            x = Image.open(annotation_path)
            y = int(line.split(';')[0])

            pred = self.detect_img(x,'top5')
            correct += y in pred
        return correct / total    
    

if __name__ == "__main__":

    # 读取测试集路劲和标签
    with open("./cls_test.txt","r") as f: 
        lines = f.readlines()
    #---------------------------------------------------#
    #   权重和模型
    #   注意：训练时设置的模型需要和权重匹配，
    #   也就是训练的啥模型使用啥权重
    #---------------------------------------------------#
    model_path = ''
    model = ConvMixer_768_32(n_classes=2)

    mode = load_dict(model_path,model)
    eval = eval_top(anno_lines=lines[:10],model=model)
    #---------------------------------------------------#
    #   top1 预测概率最好高的值与真实标签一致 √
    #   top5 预测概率前五个值由一个与真实标签一致 √
    #---------------------------------------------------#
    print('start eval.....')
    top1 = eval.eval_top1()
    
    top5 = eval.eval_top5()
    print('top1:%.3f,top5:%3.f'%(top1,top5))
    print('Eval Finished')





