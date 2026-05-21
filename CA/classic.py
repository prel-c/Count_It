import cv2
import numpy as np

def test(image, template, image1):   
    def LBP(img):

        """
        Функция поиска локального бинарного шаблона (LBP) и перевод его в десятичную систему

        Ввод - матрица ч/б изображения
        Вывод - матрица из LBP чисел

        """

        h, w=img.shape                                                              # Получаем размер изображения
        lbp_image=np.zeros((h, w), dtype=np.uint8)                                  # Создаём матрицу для заполнения LBP числами
        img_new_size=np.pad(img, 1, mode='constant')                                # Увеличиваем матрицу до размеров i+2, j+2

        neighbors=[                                                                 # Создаём матрицы такого же размера как и img, но сдвинутые в ту сторону относительно которой мы сравниваем
            img_new_size[:-2, 1:-1],                                                    # север      
            img_new_size[:-2, 2:],                                                      # северо-восток
            img_new_size[1:-1, 2:],                                                     # восток
            img_new_size[2:, 2:],                                                       # юго-восток
            img_new_size[2:, 1:-1],                                                     # юг
            img_new_size[2:, :-2],                                                      # юго-запад
            img_new_size[1:-1, :-2],                                                    # запад
            img_new_size[:-2, :-2]]                                                     # северо-запад          

        for i, n in enumerate(neighbors):                                           # Векторизованное сравнение
            lbp_image+=(n>img).astype(np.uint8)<<i            

        return lbp_image[1:-1, 1:-1]                                                # Возвращаем матрицу из LBP чисел, обрезанную до размера исходного изображения 

    def LBP_hist(lbp):
        """
        Функция для создания гистограммы из матрицы LBP

        Ввод - LBP матрица
        Вывод - Список размером 256 позиций. 
                В каждой позиции записано количество всех чисел соответсвующих позиции в списке.

        """                                        
        hist=np.bincount(lbp.ravel(), minlength=256)                                # Преобразуем массив в строку и считаем сколько раз повторяется каждый элемент значением от 0 до 255
        return hist/(hist.sum()+1e-10)                                              # Возвращаем L1 нормализованную гистограмму
    
    def Sobel(img):
        """
        Функция применения ядра собеля к ч/б изображению

        Ввод - Матрица, соответсвующая ч/б изображению
        Вывод - Матрица, содержащая в себе магнитуду каждого пикселя

        """ 
        h, w=img.shape
        Gx=np.array([[-1, 0, 1],                                                    # Создаём ядра собеля
                     [-2, 0, 2], 
                     [-1, 0, 1]], 
                     dtype=np.float32)
        
        Gy=np.array([[-1, -2, -1], 
                     [0, 0, 0], 
                     [1, 2, 1]], 
                     dtype=np.float32)
        
        img_new_size=np.pad(img, 1, mode='constant')                                # Увеличиваем матрицу до размеров i+2, j+2
        gx=(Gx[0,0]*img_new_size[0:h, 0:w]+                                             #-1*северо-запад
            Gx[0,2]*img_new_size[0:h, 2:w+2]+                                           # 1*северо-восток
            Gx[1,0]*img_new_size[1:h+1, 0:w]+                                           #-2*запад
            Gx[1,2]*img_new_size[1:h+1, 2:w+2]+                                         # 2*восток
            Gx[2,0]*img_new_size[2:h+2, 0:w]+                                           #-1*юго-запад
            Gx[2,2]*img_new_size[2:h+2, 2:w+2])                                         # 1*юго-восток

        gy=(Gy[0,0]*img_new_size[0:h, 0:w]+                                             #-1*северо-запад
            Gy[0,1]*img_new_size[0:h, 1:w+1]+                                           #-2*север
            Gy[0,2]*img_new_size[0:h, 2:w+2]+                                           #-1*северо-восток
            Gy[2,0]*img_new_size[2:h+2, 0:w]+                                           # 1*юго-запад
            Gy[2,1]*img_new_size[2:h+2, 1:w+1]+                                         #-1*юг
            Gy[2,2]*img_new_size[2:h+2, 2:w+2])                                         # 1*юго-восток      

        magnitude=np.sqrt(gx**2 + gy**2)                                            # Считаем магнитуду
        magnitude=(magnitude/(magnitude.max()+1e-10))*255                           # Нормализуем по максимальному значению
        return magnitude.astype(np.uint8)
    
    def template_search(magnitude_im, magnitude_te):
        """
        Функция реализации шаблонного поиска

        Ввод - Матрица, соответсвующая ч/б изображению
        Вывод - Массив с координатами прямоугольников, 
                соответсвующими потенциальным областям с объектом.

        """                      
        all_rectangles=[]
        h_im, w_im=magnitude_im.shape                                               # Высота и ширина магнитуды изображения
        h_te, w_te=magnitude_te.shape                                               # Высота и ширина магнитуды шаблона
        
        scales=np.linspace(0.5, 1.2, 10)
        
        for scale in scales:                                                        # scales - список масштабов для поиска
            w_te_new=max(10, int(w_te*scale))                                       # Поиск масштабированной ширины, но не менее 10 пикселей (менее 10 пикселей приводит к поломке)
            h_te_new=max(10, int(h_te*scale))                                       # Поиск масштабированной высоты, но не менее 10 пикселей (менее 10 пикселей приводит к поломке)
            
            if w_te_new>w_im or h_te_new>h_im:                                      # Проверка не стал ли шаблон больше изображения
                continue
                
            te_new=cv2.resize(magnitude_te, (w_te_new, h_te_new))                   # Масштабирование шаблона до нового размера
            result=cv2.matchTemplate(magnitude_im, te_new, cv2.TM_CCOEFF_NORMED)    # Шаблонный поиск
            
            ys, xs=np.where(result>=0.3)                                            # Позиции с удовлетворяющим сходством 
            
            for x, y in zip(xs, ys):                                                # Сохраняем все позиции с удовлетворяющим результатом
                all_rectangles.append([x, y, w_te_new, h_te_new])
        
        return all_rectangles
    
    def NMS(rectangles):
        """
        Функция удаления прямоугольников, дублирующих существующий

        Ввод - массив с набором координат прямоугольников
        Вывод - масив с набором координат "уникальных" прямоугольников

        """ 
        if len(rectangles)<=1:
            return rectangles

        rectangles_np=np.array(rectangles, dtype=np.float32)

        x1=rectangles_np[:, 0]                                                      # Получаем левую нижнюю координату по оси x
        y1=rectangles_np[:, 1]                                                      # Получаем левую нижнюю координату по оси y
        x2=rectangles_np[:, 0]+rectangles_np[:, 2]                                  # Получаем правую верхнюю координату по оси x                                     
        y2=rectangles_np[:, 1]+rectangles_np[:, 3]                                  # Получаем правую верхнюю координату по оси y
        
        areas=(x2-x1)*(y2-y1)                                                       # Находим площадь каждого прямоугольника
        idxs=np.argsort(areas)[::-1]                                                # Создаём массив с индексами отсортированных площадей по убыванию
        
        keep=[]                                                                     # Массив для сохранения нужных прямоугольников
        while len(idxs)>0:
            i=idxs[0]                                                               # Сохраняем индекс прямоугольника с наибольшей площадью
            keep.append(rectangles[i])                                              # Сохраняем прямоугольник с наибольшей площадью
            
            if len(idxs)==1:
                break

            xx1=np.maximum(x1[i], x1[idxs[1:]])                                     # Находим координату левого нижнего угла пересечения прямоугольников по оси x
            yy1=np.maximum(y1[i], y1[idxs[1:]])                                     # Находим координату левого нижнего угла пересечения прямоугольников по оси y
            xx2=np.minimum(x2[i], x2[idxs[1:]])                                     # Находим координату правого верхнего угла пересечения прямоугольников по оси x
            yy2=np.minimum(y2[i], y2[idxs[1:]])                                     # Находим координату правого верхнего угла пересечения прямоугольников по оси y
            
            w=np.maximum(0, xx2-xx1)                                                # Вычисляем ширину области пересечения
            h=np.maximum(0, yy2-yy1)                                                # Вычисляем высоту области пересечения
            S=w*h                                                                   # Ищем площадь пересечения
            iou=S/(areas[i]+areas[idxs[1:]]-S)                                      # Считаем долю, которую составляет площадь перекрытия относительно общей площади,\n
                                                                                    # занимаемой двумя прямоугольниками
            
            idxs=idxs[1:][iou<0.1]                                                  # Оставляем слабопересекающиеся прямоугольники                       

        return keep
    
    
    """
    Пайплайн обработки
    """
    image_blur=cv2.GaussianBlur(image, (5, 5), 41)                                  # Размытие по Гауссу для изображения
    template_blur=cv2.GaussianBlur(template, (5, 5), 41)                            # Размытие по Гауссу для шаблона
    
    #image_blur=image.copy()
    #template_blur=template.copy()

    magnitude_im=Sobel(image_blur)                                                  # Применение ядра Собеля к изображению
    magnitude_te=Sobel(template_blur)                                               # Применение ядра Собеля к шаблону
    
    rectangles=template_search(magnitude_im, magnitude_te)                          # Шаблонный поиск
    
    rectangles=NMS(rectangles)                                                      # Подавление немаксимумов
    
    lbp_template=LBP(template_blur)                                                 # Матрица LBP для шаблона
    hist_template=LBP_hist(lbp_template)                                            # Гистограмма LBP для
    
    valid_rectangles=[]                                                             
    for rect in rectangles:                                                         # Перебор регионов
        x, y, w, h=map(int, rect)                                                   
        Reg=image_blur[y:y+h, x:x+w]
        if Reg.size > 0:
            lbp_Reg=LBP(Reg)                                                        # Получение LBP картинки региона
            hist_Reg=LBP_hist(lbp_Reg)                                              # Получение LBP гистограммы региона
            chi2=np.sum((hist_Reg-hist_template)**2/(hist_Reg+hist_template+1e-10)) # Вычисляем хи квадрат
            if chi2<0.3:                                                            
                valid_rectangles.append(rect)
                cv2.rectangle(image1, (rect[0], rect[1]), 
                              (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)   # Рисуем результат
    return len(valid_rectangles), image1                                            # Возвращаем длинну массива удовлетворяющих областей