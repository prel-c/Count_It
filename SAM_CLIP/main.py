from SAM_CLIP.pipeline import *

def SAM_CLIPMain(image):

    scene_img = image

    print('Кликните по объекту для поиска')

    point = get_click_point(image)
    print("Выбрана точка:", point)

    print("Извлечение эталона объекта...\n")
    ref_feat, ref_img, ref_mask, ref_point = get_reference(image, point)

    plt.figure()
    plt.imshow(ref_img)
    plt.imshow(ref_mask, alpha=0.5)
    plt.scatter(ref_point[0], ref_point[1], c="red")
    plt.title("Эталон")
    plt.axis("off")
    plt.show()

    print("Выполнение подсчета...\n")

    results, _, _, _ = find_similar_objects(scene_img, ref_feat)

    show_results(scene_img, results)

    """if distances:
        plt.figure()
        plt.hist(distances, bins=30)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Порог = {threshold}')
        plt.axvline(0.2)
        plt.title("Распределение расстояний")
        plt.show()"""

if __name__ == "__main__":
    SAM_CLIPMain(cv2.cvtColor(cv2.imread("FamNet/data/images_384_VarV2/263.jpg"), cv2.COLOR_BGR2RGB))