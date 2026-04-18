import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import  Resnet50FPN, CountRegressor, weights_normal_init
from utils import MAPS, Scales, extract_features, \
extract_segmentation_map, show_segmentation_result, show_pca_features, \
show_top_channels, show_max_activation
from PIL import Image
from torchvision import transforms
from os.path import exists,join
import cv2
#from sobel import sobel

MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def prepare_sample(image, max_hw=1504):
    """
    Упрощенная версия resizeImage
    image: PIL Image
    """
    W, H = image.size
    scale_factor = 1.0
    
    # 1. Считаем коэффициент масштабирования
    if W > max_hw or H > max_hw:
        scale_factor = float(max_hw) / max(H, W)
    
    # 2. Делаем размеры кратными 8 (ОЧЕНЬ ВАЖНО для ResNet)
    new_H = 8 * int(H * scale_factor / 8)
    new_W = 8 * int(W * scale_factor / 8)
    
    # Реальный масштаб может чуть измениться из-за округления до кратного 8
    # Но для простоты обычно используют scale_factor
    resized_image = transforms.Resize((new_H, new_W))(image)
    
    # Превращаем в тензоры
    image_tensor = Normalize(resized_image).unsqueeze(0) # [1, 3, H, W]

    
    return image_tensor

Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])


"""def rank_channels_by_contrast(feat_map):
    if torch.is_tensor(feat_map):
        # Убираем батч, переводим на CPU и в Numpy
        feat_map = feat_map[0].detach().cpu().numpy()
    # feat_map: [C, H, W]
    scores = []
    for i in range(feat_map.shape[0]):
        ch = feat_map[i]
        # Чем выше вариативность (std), тем более "выражен" объект на фоне
        score = np.std(ch)
        scores.append(score)
    return np.argsort(scores)[::-1]"""

"""data_path = 'D:/mygit/opd/ResNet/data/'
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

resnet50_conv = Resnet50FPN()
resnet50_conv.to(device)
resnet50_conv.eval()

"""with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

im_ids = data_split['train']
random.shuffle(im_ids)
train_mae = 0
train_rmse = 0
train_loss = 0
pbar = tqdm(im_ids)
cnt = 0
for im_id in pbar:
    cnt += 1
    anno = annotations[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    rects = list()
    for bbox in bboxes:
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[2][0]
        y2 = bbox[2][1]
        rects.append([y1, x1, y2, x2])
    print('WEKLGHAKLHG', rects)

    if len(rects) == 0:
        print(f" Пропуск {im_id}: нет боксов-примеров в аннотации.")
        continue


    image = Image.open('{}/{}'.format(im_dir, im_id))
    #print(image.filename)
    image = image.convert('RGB')
    image.load()

    sample = {'image':image,'lines_boxes':rects}
    sample = TransformTrain(sample)
    image, boxes = sample['image'].to(device), sample['boxes'].to(device)

    if boxes.dim() == 3:
                boxes = boxes.unsqueeze(1) 
    elif boxes.dim() == 2:
        boxes = boxes.unsqueeze(1).unsqueeze(1)

    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes, MAPS, Scales)"""

"""# Извлекаем признаки (ваша новая функция)
seg_features = extract_segmentation_map(resnet50_conv, image.unsqueeze(0), boxes, target_map='map1')

# Показываем результат
show_segmentation_result(image, seg_features)"""

image = Image.open("./data/images_384_VarV2/200.jpg").convert('RGB')

image.load()

image = prepare_sample(image)
#print(image.shape, type(image))
    
Image_features = resnet50_conv(image)
feat_map = Image_features['map1']

#print(rank_channels_by_contrast(feat_map))


#result = show_pca_features(feat_map)
result = show_top_channels(feat_map, num_channels=3, upscale_factor=2)
#esult = show_max_activation(feat_map)

cv2.imshow("Результат",result)
cv2.waitKey(0)