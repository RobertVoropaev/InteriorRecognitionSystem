import numpy as np
import os
import shutil
import random

from skimage.io import imread, imsave

from scripts_.PathFinder import PathFinder


class SegEncoder:
    def __init__(self,
                 objectnames_path: str = "static/objectnames.txt"):
        """
            Конструктор класса
        ------------------------------------------------------
        Desc:
            Класс декодирования информации из пикселей
        Input:
            * objectnames_path - путь до файла objectnames.txt
        """
        self.index_old_to_new_dict = dict()
        self.pf = PathFinder()

        self.obj_names = []
        with open(objectnames_path, 'r') as f:
            for line in f:
                self.obj_names.append(line.strip())
        self.index_dict_is_load = False

    def get_class_index(self, pixel):
        """
            Индекс класса по пикселю
        ------------------------------------------------------
        Desc:
            R / 10 * 256 + G
        Input:
            * pixel - numpy.array([R, G, B]) - пиксель
        Output:
            Индекс класса от 1 до ...
        """

        R = pixel[0]
        G = pixel[1]
        B = pixel[2]
        return int(R / 10 * 256 + G)

    def get_class(self, pixel):
        """
            Название класса по пикселю
        ------------------------------------------------------
        Desc:
            objectnames[R / 10 * 256 + G - 1]
        Input:
            * pixel - numpy.array([R, G, B]) - пиксель
        Output:
            Имя класса
        """
        return self.obj_names[self.get_class_index(pixel) - 1]

    ################################## Load ######################################

    def load_index_dict(self,
                        index_old_to_new_dict: dict):
        """
            Загружает словарь перекодирования классов
        ------------------------------------------------------
        Input:
            * index_old_to_new_dict - словарь {старый индекс -> новый}
        """
        self.index_old_to_new_dict = index_old_to_new_dict
        self.index_dict_is_load = True

    def get_encoded_seg(self, seg):
        """
            Перекодируут маску в новую по правилам загруженного словаря
        ------------------------------------------------------
        Input:
            * seg - маска изображения с пикселями от 0 до 255
        Output:
            Перекодированная одноканальная маска
        """
        if not self.index_dict_is_load:
            raise Exception("Index dict don't load. Use load_index().")

        R = seg[:, :, 0]
        G = seg[:, :, 1]
        B = seg[:, :, 2]

        new_seg = (R / 10 * 256 + G).astype(np.uint16)
        new_seg[new_seg != 0] -= 1

        old_list = sorted(self.index_old_to_new_dict.keys())
        for old in old_list:
            new_seg[new_seg == old] = self.index_old_to_new_dict[old]

        return new_seg

    def encode_dataset(self,
                       dir_src: str = 'data/ADE20K_filtred/images/train/',
                       dir_dst: str = 'data/ADE20K_encoded/',
                       update_current_dir: bool = False,
                       progress_step: int = 0):
        """
            Перекодировка всего датасета по загруженным правилам
        ------------------------------------------------------
        Desc:
            Создаёт новый датасет, разделяя на подпапки:
                train/img/
                train/mask/
                val/img/
                val/mask/
        Input:
            * dir_src - папка исходного датасета, в подпапках которого лежат файлы img и seg
            * dir_dst - папка, в которой будет создан новый датасет
            * update_current_dir - флаг: переписывать ли папку назначения, если она уже существует
            * progress_step <int> - шаг, с которым будет выводиться сообщения о прогрессе
                Done: <num> - обработано <num> фотографий
        """

        if not self.index_dict_is_load:
            raise Exception("Index dict don't load. Use load_index().")

        img_dst_train = dir_dst + 'train/img/'
        mask_dst_train = dir_dst + 'train/mask/'

        img_dst_val = dir_dst + 'val/img/'
        mask_dst_val = dir_dst + 'val/mask/'

        if (os.path.isdir(img_dst_train) and os.path.isdir(mask_dst_train) and
                os.path.isdir(img_dst_val) and os.path.isdir(mask_dst_val) and not update_current_dir):
            raise Exception("Dst dir don't empty. Use update_current_dir=True.")

        if os.path.isdir(dir_dst):
            shutil.rmtree(dir_dst)
        os.makedirs(img_dst_train)
        os.makedirs(mask_dst_train)
        os.makedirs(img_dst_val)
        os.makedirs(mask_dst_val)

        if progress_step != 0:
            progress_counter = 0
            dataset_size = self.pf.get_dataset_size(dir_src)
            print("Size: " + str(dataset_size))

        for file, path in self.pf.data_gen(dir_src, return_path=True):
            name, description, train_or_val, parts_num = self.pf.get_format(file)

            img_dst = img_dst_train
            mask_dst = mask_dst_train
            if train_or_val == 'val':
                img_dst = img_dst_val
                mask_dst = mask_dst_val

            if description == 'img':
                img_path = path + '/' + file
                shutil.copyfile(img_path, img_dst + name + '.jpg')

            elif description == 'seg':
                seg_path = path + '/' + file
                seg = imread(seg_path)
                new_seg = self.get_encoded_seg(seg)
                imsave(mask_dst + name + '.png', new_seg, check_contrast=False)

                if progress_step != 0:
                    progress_counter += 1
                    if progress_counter % progress_step == 0:
                        print("Done: " + str(progress_counter))

    def get_bounding_box(self, seg, class_arr: list):
        """
            Возвращает словарь из границ квадрата для каждого элемента из class_arr на маске
        ------------------------------------------------------
        Desc:
            Проходит по маске и собирает все bounding box'ы для классов из class_arr
        Input:
            * seg - np.array() маска
            * class_arr - список имён классов
        Output:
            Словарь: class_name => (min_h, max_h, min_w, max_w), где индекс в массиве соответсвует индексу в class_arr
        """

        MAX, MIN = 10000, -1

        hw_dict = dict()
        for class_name in class_arr:
            hw_dict[class_name] = [MAX, MIN, MAX, MIN]

        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                pixel = seg[i, j]
                current_class = self.get_class(pixel)

                if current_class not in hw_dict:
                    continue

                if hw_dict[current_class][0] == MAX:
                    hw_dict[current_class][0] = i
                hw_dict[current_class][1] = i

                if j < hw_dict[current_class][2]:
                    hw_dict[current_class][2] = j
                if j > hw_dict[current_class][3]:
                    hw_dict[current_class][3] = j
        return hw_dict

    def get_hw_dict(self,
                    dir_path: str = 'data/ADE20K_filtred/images/train/',
                    class_list: list = list(),
                    max_num: int = 5,
                    skip_probably: int = 0.5,
                    progress_bar: bool = False):
        """
            Возвращает словарь из указанного числа разных bounding_box'ов объектов списка классов
        ------------------------------------------------------
        Desc:
            Рекурсивно обходит папку и сохраняет bounding_box и путь до файла, если встречает объект в маске
        Input:
            * dir_path - папка для обхода
            * class_list - список названий (оригинальных) объектов классов
                    Если передаётся пустой, то функция будет собирать все объекты
            * max_num - максимальное число уже найденных объектов класса, при котором новая маска будет рассматриваться
                        (если данного объекта уже слишком много, то маска рассматриваться не будет,
                        если там нет другого объекта из списка, которого не хватает
            * skip_probably - вероятность того, что данный файл датасета будет пропущен
                        (нужно, так как иногда встречаются подряд много фотографий в высоком качестве,
                        которые очень долго обрабатываются)
            * progress_bar - вывод количества уже рассмотренных файлов и статистики по найденным предметам
        Output:
            Словарь: class_name => dict(path, min_h, max_h, min_w, max_w),
                где class_name в массиве соответсвует элементу в class_list
        """



        hw_all = dict()
        progress_counter = 0
        for file, path in self.pf.data_gen(dir_path, return_path=True):
            if random.random() > skip_probably:
                continue

            path += '/'
            name, description, train_or_val, parts_num = self.pf.get_format(file)

            if description == "img":
                progress_counter += 1

                img = imread(path + self.pf.get_img(file))
                seg = imread(path + self.pf.get_seg(file))
                list_obj = self.pf.get_class_list_from_desc(path + self.pf.get_text(file))

                if class_list:
                    is_contain = False
                    for class_name in class_list:
                        if class_name in list_obj:
                            if class_name not in hw_all or len(hw_all[class_name]) < max_num:
                                is_contain = True
                                break
                    if not is_contain:
                        continue

                hw_d = self.get_bounding_box(seg, list_obj)

                if class_list:
                    class_arr = class_list
                else:
                    class_arr = list_obj

                for class_name in class_arr:
                    if class_name not in list_obj:
                        continue

                    box = (min_h, max_h, min_w, max_w) = hw_d[class_name]
                    d = {"path": path + self.pf.get_img(file),
                         "min_h": box[0], "max_h": box[1],
                         "min_w": box[2], "max_w": box[3]}

                    if class_name in hw_all:
                        hw_all[class_name].append(d)
                    else:
                        hw_all[class_name] = [d]

                if progress_bar:
                    print("Done: " + str(progress_counter))

                if class_list:
                    class_arr = class_list
                else:
                    class_arr = hw_all.keys()

                is_all_more_max = True
                for class_name in class_arr:
                    if class_name in hw_all:
                        num = len(hw_all[class_name])
                    else:
                        num = 0

                    if num < max_num:
                        is_all_more_max = False

                    if progress_bar:
                        print(class_name + ": " + str(num), end="; ")

                if progress_bar:
                    print()

                if is_all_more_max:
                    break

        return hw_all
