import numpy as np
import torch.nn.functional as F
import math
from torchvision import transforms
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')
from sklearn.decomposition import PCA
from sobel import sobel, ad_cont


MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Pres 'n' and then use mouse drag to sdraw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI("image", image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    """if sumh != 0:
        h /= sumh"""
    return h

def PerturbationLoss(output,boxes,sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu and torch.cuda.is_available():
                GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu and torch.cuda.is_available():
                GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(),GaussKernel) 
    return Loss

def MincountLoss(output, boxes, use_gpu=True):
    ones = torch.ones(1)
    if torch.cuda.is_available() and use_gpu:
        ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)  
    return Loss

def pad_to_size(feat, desire_h, desire_w):
    """ zero-padding a four dim feature matrix: N*C*H*W so that the new Height and Width are the desired ones
        desire_h and desire_w should be largers than the current height and weight
    """

    cur_h = feat.shape[-2]
    cur_w = feat.shape[-1]

    left_pad = (desire_w - cur_w + 1) // 2
    right_pad = (desire_w - cur_w) - left_pad
    top_pad = (desire_h - cur_h + 1) // 2
    bottom_pad =(desire_h - cur_h) - top_pad

    return F.pad(feat, (left_pad, right_pad, top_pad, bottom_pad))

def extract_features(feature_model, image, boxes, feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
    N, M = image.shape[0], boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)

    feat = Image_features['map2'][0] # Форма станет [512, 48, 72]

    # Усредняем все каналы, чтобы увидеть "общую активацию"
    feat_mean = torch.mean(feat, dim=0).cpu().numpy() 

    # 3. Нормализация (чтобы значения были от 0 до 255)
    feat_min = feat_mean.min()
    feat_max = feat_mean.max()
    if feat_max > feat_min:
        feat_norm = (feat_mean - feat_min) / (feat_max - feat_min)
    else:
        feat_norm = feat_mean

    feat_viz = (feat_norm * 255).astype(np.uint8)

    feat_viz = cv2.resize(feat_viz, (48*8, 72*8), interpolation=cv2.INTER_LINEAR)


    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        #boxes = boxes.squeeze(0)
        curr_boxes = boxes[ix][0]
        cnter = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = curr_boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive 
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))            
            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])  
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4]) 
                #print(y1,y2,x1,x2,max_h,max_w)
                if j == 0:
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)
                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')                    
                else:
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                    examples_features = torch.cat((examples_features,feat),dim=0)
            """
            Convolving example features over image features
            """
            h, w = examples_features.shape[2], examples_features.shape[3]
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            combined = features.permute([1,0,2,3])
            # computing features for scales 0.9 and 1.1 
            for scale in exemplar_scales:
                    h1 = math.ceil(h * scale)
                    w1 = math.ceil(w * scale)
                    if h1 < 1: # use original size if scaled size is too small
                        h1 = h
                    if w1 < 1:
                        w1 = w
                    examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')  
                    features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    examples_features_scaled)
                    features_scaled = features_scaled.permute([1,0,2,3])
                    combined = torch.cat((combined,features_scaled),dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
    return All_feat


class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage( MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize_output_and_save(input_, output, boxes, save_path, figsize=(20, 12), dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    # get the total count
    pred_cnt = output.sum().item()
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(boxes[i, 3].item()), int(
            boxes[i, 4].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = 0.2989*img1[:,:,0] + 0.5870*img1[:,:,1] + 0.1140*img1[:,:,2]
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)


    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def extract_segmentation_map(feature_model, image, boxes, target_map='map1'):
    """
    Извлекает карту корреляции (похожести) для сегментации с нужного слоя (например, map1 или map2).
    
    Возвращает:
    Тензор формы [Batch_size, M, H, W], где M - количество примеров (boxes).
    """
    N, M = image.shape[0], boxes.shape[2]
    
    # 1. Получаем все карты признаков от модели
    Image_features = feature_model(image)
    
    print("Доступные слои в модели:", Image_features.keys())

    # Определяем фактор сжатия для пересчета координат рамок
    # В оригинальном коде map1 и map2 имели масштаб 4.0
    scale_dict = {'map1': 4.0, 'map2': 4.0, 'map3': 8.0, 'map4': 16.0} 
    Scaling = scale_dict.get(target_map, 4.0)
    
    batch_correlation_maps = []
    
    # 2. Обрабатываем каждое изображение в батче
    for ix in range(N):
        curr_boxes = boxes[ix][0]
        # Достаем только нужный нам слой (например, map1)
        image_features = Image_features[target_map][ix].unsqueeze(0)
        
        # 3. Масштабируем координаты рамок под размер Feature Map
        boxes_scaled = curr_boxes / Scaling
        boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
        boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5]) + 1 
        
        feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
        
        # Защита от выхода за границы картинки
        boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
        boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
        boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)            
        
        box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
        box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]            
        max_h, max_w = math.ceil(max(box_hs)), math.ceil(max(box_ws))            
        
        # 4. Вырезаем признаки для каждого примера
        examples_features = []
        for j in range(M):
            y1, x1 = int(boxes_scaled[j, 1]), int(boxes_scaled[j, 2])  
            y2, x2 = int(boxes_scaled[j, 3]), int(boxes_scaled[j, 4]) 
            
            feat = image_features[:, :, y1:y2, x1:x2]
            
            # Приводим все примеры к единому размеру (самого большого примера)
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear', align_corners=False)
            
            examples_features.append(feat)
            
        # Объединяем примеры в один тензор
        examples_features = torch.cat(examples_features, dim=0)
        
        # 5. Свертка (поиск объектов на картинке)
        h, w = examples_features.shape[2], examples_features.shape[3]
        padding = (int(w/2), int((w-1)/2), int(h/2), int((h-1)/2))
        
        # Получаем карту корреляции
        correlation_map = F.conv2d(
            F.pad(image_features, padding),
            examples_features
        )
        
        batch_correlation_maps.append(correlation_map)
        
    # Возвращаем склеенный результат для всего батча
    return torch.cat(batch_correlation_maps, dim=0)


def show_segmentation_result(image_tensor, correlation_map, window_name="Segmentation Map"):
    """
    Преобразует тензор корреляции в визуальную маску.
    image_tensor: исходное изображение [1, 3, H, W]
    correlation_map: результат функции extract_segmentation_map [1, M, h, w]
    """
    # 1. Усредняем по всем примерам (dim=1) и убираем лишние размерности
    # Результат: матрица [h, w]
    mask = torch.mean(correlation_map, dim=1).squeeze().cpu().detach().numpy()
    
    # 2. Нормализация (чтобы значения были от 0 до 1)
    mask_min = mask.min()
    mask_max = mask.max()
    if mask_max > mask_min:
        mask = (mask - mask_min) / (mask_max - mask_min)
    
    # 3. Приводим к размеру исходного изображения (если нужно)
    # image_tensor имеет форму [1, 3, H, W]
    orig_h, orig_w = image_tensor.shape[-2], image_tensor.shape[-1]
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # 4. Превращаем в формат 0-255 для OpenCV
    mask_visual = (mask_resized * 255).astype(np.uint8)
    
    # 5. Опционально: применяем цветовую карту (Heatmap), чтобы было нагляднее
    heatmap = cv2.applyColorMap(mask_visual, cv2.COLORMAP_JET)
    
    cv2.imshow(window_name, heatmap)
    cv2.waitKey(0) # Ждем нажатия клавиши
    
    return mask_visual


def show_pca_features(feature_map):
    """
    Визуализирует многоканальную карту признаков через метод главных компонент (PCA).
    """
    # 1. Подготовка данных: убираем батч и приводим к виду [C, H*W] -> [H*W, C]
    feat = feature_map[0].detach().cpu().numpy()
    c, h, w = feat.shape
    feat_flat = feat.reshape(c, -1).T 
    
    # 2. PCA: сжимаем все каналы (например, 256) в 3 главных компонента (RGB)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feat_flat) 
    
    # 3. Нормализация в диапазон [0, 1] для корректного отображения цветов
    pca_min = pca_result.min(axis=0)
    pca_max = pca_result.max(axis=0)
    pca_result = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
    
    # Решейпим обратно в картинку [H, W, 3] и переводим в 0-255
    pca_img = (pca_result.reshape(h, w, 3) * 255).astype(np.uint8)
    
    # 4. Увеличение для наглядности (так как map2 в 4 раза меньше оригинала)
    pca_img_resized = cv2.resize(pca_img, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    
    # Конвертируем RGB в BGR для корректной работы OpenCV
    pca_img_resized = cv2.cvtColor(pca_img_resized, cv2.COLOR_RGB2BGR)
    
    return pca_img_resized

def show_top_channels(feature_map, num_channels=6, upscale_factor=4):
    # feature_map: [1, 256, H, W]
    feat = feature_map[0].cpu().detach().numpy()
    
    # Считаем среднюю активацию каждого канала, чтобы найти самые "яркие"
    channel_energy = feat.mean(axis=(1, 2))
    top_indices = np.argsort(channel_energy)[-num_channels:][::-1]

    combined_img = []
    #obel_list = []
    for idx in top_indices:
        ch = feat[idx]
        # Нормализация для отображения
        ch = ((ch - ch.min()) / (ch.max() - ch.min() + 1e-5) * 255).astype(np.uint8)
        #ch_sobel = sobel(ch)
        h, w = ch.shape
        ch_resized = cv2.resize(ch, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_NEAREST)
        #ch_resized_sobel = cv2.resize(ch_sobel, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_NEAREST)
        ch_colored = cv2.applyColorMap(ch_resized, cv2.COLORMAP_VIRIDIS)
        
        #ch_resized_sobel = np.stack([ch_resized_sobel] * 3, axis=-1)

        #ch_resized_sobel = cv2.cvtColor(ch_resized_sobel, cv2.COLOR_GRAY2BGR)
        cv2.putText(ch_colored, f"Ch: {idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        #sobel_list.append(ch_resized_sobel)
        combined_img.append(ch_colored)
        #print(f"Сумма пикселей изображения {idx} = ", np.sum(ch_resized_sobel))

    # Склеиваем в одну плитку
    res = np.hstack(combined_img)
    #res2 = np.hstack(sobel_list)

    return res

def show_max_activation(feature_map, upscale_factor=4):
    """
    Визуализирует карту признаков, выбирая максимальную активацию среди всех каналов.
    Это помогает увидеть четкие границы объектов, которые "заметила" сеть.
    """
    # 1. Находим максимум по всем каналам (dim=1)
    # feature_map имеет форму [1, 256, H, W]
    # Результат будет [1, H, W]
    max_feat, _ = torch.max(feature_map[0], dim=0)
    
    # Переводим в numpy
    max_feat = max_feat.cpu().detach().numpy()
    
    # 2. Нормализация значений в диапазон 0-255
    f_min = max_feat.min()
    f_max = max_feat.max()
    if f_max > f_min:
        max_feat = (max_feat - f_min) / (f_max - f_min)
    
    max_feat_img = (max_feat * 255).astype(np.uint8)
    
    # 3. Увеличение размера (масштабирование под оригинал)
    h, w = max_feat_img.shape
    max_feat_img = cv2.resize(max_feat_img, (w * upscale_factor, h * upscale_factor), 
                               interpolation=cv2.INTER_LINEAR)
    
    # 4. Применение цветовой карты для наглядности (опционально)
    # Можно использовать COLORMAP_BONE для рентгеновского эффекта или JET для теплового
    colored_map = cv2.applyColorMap(max_feat_img, cv2.COLORMAP_MAGMA)
    
    return max_feat_img
