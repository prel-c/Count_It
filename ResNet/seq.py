from rembg import remove
from PIL import Image

def remove_background(input_path: str, output_path: str):
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    output_image.save(output_path)

    background = Image.new("RGB", output_image.size, (255, 255, 255))
    background.paste(output_image, mask=output_image.split()[3])  # alpha канал

    background.save(output_path, "JPEG")

# пример использования
remove_background("image.png", "535-out.png")