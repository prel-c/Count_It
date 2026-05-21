from pipeline import *
from dataset import *

if __name__ == "__main__":
    REF_IMAGE = "data/images_384_VarV2/263.jpg"
    SCENE_IMAGE = "data/images_384_VarV2/263.jpg"
    img = cv2.imread(REF_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    point = get_click_point(img)
    print("Выбрана точка:", point)

    print("Извлечение эталона....")
    ref_feat, ref_img, ref_mask, ref_point = get_reference(img, point)

    plt.figure()
    plt.imshow(ref_img)
    plt.imshow(ref_mask, alpha=0.5)
    plt.scatter(ref_point[0], ref_point[1], c="red")
    plt.title("Эталон")
    plt.axis("off")
    plt.show()

    print("Поиск объектов...")

    # Загружаем изображение сцены:
    scene_img = cv2.imread(SCENE_IMAGE)
    scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    results, scene_img_resized, distances, threshold = find_similar_objects(scene_img, ref_feat)

    show_results(scene_img, results)

    if distances:
        plt.figure()
        plt.hist(distances, bins=30)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Порог = {threshold}')
        plt.axvline(0.2)
        plt.title("Распределение расстояний")
        plt.show()