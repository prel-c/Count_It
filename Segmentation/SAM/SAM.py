import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def sam_test(image, h, w):

    checkpoint = "Segmentation/SAM/data/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=16,
        crop_n_layers=1,
        points_per_batch=32
    )


    height, width, _ = image.shape

    MIN_AREA = int(0.01 * w * h)    
    MAX_AREA = int(0.2 * width * height)  

    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
            masks = mask_generator.generate(image)
    else:
        masks = mask_generator.generate(image)

    combined_mask = torch.zeros((height, width), dtype=bool, device=device)

    filtered_count = 0

    for mask_data in masks:
        area = mask_data['area']
        if MIN_AREA <= area <= MAX_AREA:
            m_tensor = torch.from_numpy(mask_data['segmentation']).to(device)
            combined_mask |= m_tensor
            filtered_count += 1

    padding = int((w + h) // 2 * 0.1) 
    if padding < 1: padding = 1
    if padding % 2 == 0: padding += 1
    
    mask_4d = combined_mask.float().unsqueeze(0).unsqueeze(0)
    dilated_mask = F.max_pool2d(mask_4d, kernel_size=padding, stride=1, padding=padding // 2)

    combined_mask = dilated_mask.squeeze(0).squeeze(0) > 0

    image_tensor = torch.from_numpy(image).to(device)


    mean_color = image_tensor.float().mean(dim=(0, 1)).to(torch.uint8)


    clean_image_tensor = torch.where(combined_mask.unsqueeze(-1), image_tensor, 0)

    return clean_image_tensor.cpu().numpy()


if __name__ == "__main__":

    image_path = "D:/mygit/opd/FamNet/data/images_384_VarV2/200.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
        exit()

    box = np.array(cv2.selectROI("Select Area", image, fromCenter=False, showCrosshair=True))
    cv2.destroyAllWindows()
    x, y, w, h = box

    clean_image = sam_test(image, h, w)
    #clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    
    plt.imsave("output.png", clean_image)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("1. Оригинал")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Очищено для нейросети")
    plt.imshow(clean_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite("image.png", clean_image)
