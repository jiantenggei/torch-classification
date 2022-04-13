import imp
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from losses.focal_loss import FocalLoss
from config import label_smoothing
from utils.training_utils import smooth_one_hot
from utils.utils import get_classes
from config import classes_path,smoothing_value
import config
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
#nn.BCEWithLogitsLoss是对网络的输出进行Sigmoid(); 交叉熵则是采用的Softmax
def fit_one_epoch(model_train, model, tb_writer, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,criterion):
    
    # 记录日志啊

    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    train_loss      = 0
    train_accuracy  = 0
    val_accuracy     = 0
    val_loss        = 0
    _,classes = get_classes(classes_path)


    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            images, targets = batch
            with torch.no_grad():
                images      = torch.from_numpy(images).type(torch.FloatTensor)
                targets     = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()
            # label_smmothing
            if label_smoothing:
                targets = smooth_one_hot(targets,classes=classes,smoothing=smoothing_value)
            optimizer.zero_grad()
            outputs     = model_train(images)
            loss_value = criterion(outputs,targets)
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()
            with torch.no_grad(): # 训练集准确率
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                train_accuracy += accuracy.item()

            pbar.set_postfix(**{'train_loss': train_loss / (iteration + 1), 
                                'train_accuracy'  : train_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs     = model_train(images)
                loss_value = criterion(outputs,targets)
                    
                val_loss    += loss_value.item()
                # 验证集准确率
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                val_accuracy += accuracy.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'val_accuracy'  : val_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], train_accuracy, epoch)
    tb_writer.add_scalar(tags[2], val_loss, epoch)
    tb_writer.add_scalar(tags[3], val_accuracy, epoch)
    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model_train.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), train_loss / epoch_step, val_loss / epoch_step_val))

    return train_loss,train_accuracy,val_loss,val_accuracy
 



    
