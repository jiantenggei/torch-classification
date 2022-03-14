Cuda             = True  #是否使用GPU 没有为Flase

input_shape      = [224,224]  # 输入图片大小


batch_size      = 2 # 自己可以更改
lr              = 1e-3         

classes_path    = 'classes.txt'


num_workers     = 0  # 是否开启多进程


annotation_path     = 'cls_train.txt'  



val_split       = 0.1  #验证集比率


resume          =''  # 加载训练权重路径

log_dir         = 'logs' # 日志路径 tensorboard 保存