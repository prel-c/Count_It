import torch
import torch.nn as nn
import torch.nn.functional as F
from model import  Resnet50FPN
from utils import MAPS, Scales, extract_features, \
extract_segmentation_map, show_segmentation_result, show_pca_features, \
show_top_channels, show_max_activation
from PIL import Image
from torchvision import transforms
import cv2

MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def prepare_sample(image, max_hw=1504):
    W, H = image.size
    scale_factor = 1.0
    
    if W > max_hw or H > max_hw:
        scale_factor = float(max_hw) / max(H, W)
    
    #ResNet важны размеры кратные 8
    new_H = 8 * int(H * scale_factor / 8)
    new_W = 8 * int(W * scale_factor / 8)
    
    resized_image = transforms.Resize((new_H, new_W))(image)
    image_tensor = Normalize(resized_image).unsqueeze(0) # [1, 3, H, W]
    return image_tensor

Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

resnet50_conv = Resnet50FPN()
resnet50_conv.to(device)
resnet50_conv.eval()

image = Image.open("./data/images_384_VarV2/200.jpg").convert('RGB')
image.load()

image = prepare_sample(image)
#print(image.shape, type(image))
    
Image_features = resnet50_conv(image)
feat_map = Image_features['map1'] # можно выбирать и map2, map3, map4

result = show_pca_features(feat_map)
#result = show_top_channels(feat_map, num_channels=3, upscale_factor=2)
#esult = show_max_activation(feat_map)

cv2.imshow("Результат",result)
cv2.waitKey(0)