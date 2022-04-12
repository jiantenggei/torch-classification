Cuda             = True  #是否使用GPU 没有为Flase

input_shape      = [112,112]  # 输入图片大小

batch_size      = 4 # 自己可以更改
lr              = 1e-3         

classes_path    = 'classes.txt'


num_workers     = 0  # 是否开启多进程


annotation_path     = 'cls_train.txt'  



val_split       = 0.1  #验证集比率


resume          =''  # 加载训练权重路径

log_dir         = 'logs' # 日志路径 tensorboard 保存

#------------------------------------------#
#   FocalLoss ：处理样本不均衡
#   alpha   
#   gamma >0 当 gamma=0 时就是交叉熵损失函数
#   论文中gamma = [0,0.5,1,2,5]
#   一般而言当γ增加的时候，a需要减小一点
#   reduction ： 就平均：'mean' 求和 'sum'
#   还未ti
#------------------------------------------#
#Focal_loss      = True  # True Focal loss 处理原本不均衡 False  使用 CrossEntropyLoss() # 还未使用成功

#label_smoothing 防止过拟合
label_smoothing =  False #

smoothing_value = 0.1  #[0，1] 之间



#学习率变化策略
scheduler   = '' #[None,reduce,cos] None保持不变 reduce  按epoch 来减少 cos 余弦下降算法