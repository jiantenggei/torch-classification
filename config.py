Cuda             = False  #是否使用GPU 没有为Flase

input_shape      = [224,224]  # 输入图片大小


# backbone        = 'ConvMixer'  # 可选ConvMixer,MLP-Mixer

batch_size      = 1 # 自己可以更改
lr              = 1e-3         

classes_path    = 'classes.txt'


num_workers     = 0  # 是否开启多进程


annotation_path     = 'cls_train.txt'  



val_split       = 0.1


resume          =''

log_dir         = 'logs'


