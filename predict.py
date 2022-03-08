from PIL import Image
from eval import eval_top
from nets.Convnext import convnext_tiny
from utils.utils import load_dict
#加载模型
model_path = 'logs\ep050-loss0.414-val_loss0.376.pth'
model = convnext_tiny(pretrained=True,num_classes=2)
# mode = load_dict(model_path,model)
eval = eval_top(anno_lines=None,model=model)

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = eval.detect_img(image,mode='predict')
        print(class_name)
