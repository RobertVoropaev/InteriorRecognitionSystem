import numpy as np
import os
import shutil
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
            * dir_src - папка с файлами img и seg
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
