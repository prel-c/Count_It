import os
import shutil
from PIL import Image
from tqdm import tqdm

def process_dataset(img_in, anno_in, img_out, anno_out, target_size=(576, 384)):
    """переименовывает и синхронизует с анотациями файлы"""
    for folder in [img_out, anno_out]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    valid_img_exts = ('.jpg', '.jpeg', '.png')
    
    image_files = [f for f in os.listdir(img_in) if f.lower().endswith(valid_img_exts)]
    image_files.sort()

    print(f"Найдено пар для обработки: {len(image_files)}")

    for i, filename in enumerate(tqdm(image_files), start=1):
        base_name = os.path.splitext(filename)[0]
        new_base_name = str(i)
        
        img_path = os.path.join(img_in, filename)
        with Image.open(img_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            resized_img.save(os.path.join(img_out, f"{new_base_name}.jpg"), "JPEG", quality=95)

        txt_filename = base_name + ".txt"
        src_txt_path = os.path.join(anno_in, txt_filename)
        dst_txt_path = os.path.join(anno_out, f"{new_base_name}.txt")
        
        if os.path.exists(src_txt_path):
            shutil.copy(src_txt_path, dst_txt_path)
        else:
            open(dst_txt_path, 'a').close()
            print(f" Предупреждение: файл {txt_filename} не найден, создан пустой .txt")

IMG_INPUT = 'data/my/Pipes_340/images'
ANNO_INPUT = 'data/my/Pipes_340/labels'

IMG_OUTPUT = 'data/my/Pipes_340/img_new'
ANNO_OUTPUT = 'data/my/Pipes_340/lab_new'

process_dataset(IMG_INPUT, ANNO_INPUT, IMG_OUTPUT, ANNO_OUTPUT)
