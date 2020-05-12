import os
import pandas as pd
import numpy as np
from skimage.io import imread


class PathFinder:
    def get_format(self, file_name):
        """
            Функция преобразования имени файлов
        ------------------------------------------------------
        Desc:
            Раскладывает имя файла на части
        Input:
            * file_name - имя файла
        Output:
            * имя(номер) объекта
            * файловый формат: img, text, seg, parts
            * приналдлежность к валидации: train, val
            * номер части: (число) для файлов parts, 0 для остальных
        """

        name_split = file_name.split("_")
        file_format = name_split[0:-1] + name_split[-1].split(".")
        ade, train_or_val, name = file_format[:3]
        extension = file_format[-1]

        description = 0
        parts_num = 0
        if extension == 'jpg':
            description = 'img'
        elif extension == 'txt':
            description = 'text'
        elif extension == 'png':
            description = file_format[3]
            if file_format[4].isdigit():
                parts_num = int(file_format[4])

        return [name, description, train_or_val, parts_num]

    def data_gen(self, dir_path, return_path=False):
        """
            Генератор объектов
        ------------------------------------------------------
        Desc:
            Делает рекурсивный обход папки и возвращает имена всех файлов
        Input:
            * dir_path - имя главной папки обхода
            * return_path - возвращать ли путь до папки с файлом
        Output:
            * генератор, yield имя файла, (путь до папки файла)
        """
        for path, dirs, files in os.walk(dir_path):
            if files:
                for file in files:
                    if return_path:
                        yield file, path
                    else:
                        yield file

    def get_df_description(self, desc_path, only_part_level=-1):
        """
            Возвращает DataFrame из описания
        ------------------------------------------------------
        Desc:
            Преобразует файл формата text в dataframe таблицу
        Input:
            * desc_path - имя файла с описанием маски
            * only_part_level <level> -
                -1, чтобы выводить все уровни,
                <level>, только указанный
        Output:
            * DataFrame указанного файла
        """

        description = []
        with open(desc_path) as f:
            for line in f:
                (instance_n, part_level, occluded,
                 class_name, original_name, attributes_list) = line.rstrip().split(" # ")

                if only_part_level != -1:
                    if int(part_level) != only_part_level:
                        continue

                description.append([int(instance_n), int(part_level), occluded == "1",
                                    class_name, original_name, attributes_list.strip("\"")])

        return pd.DataFrame(description).rename(columns={0: "instance_n", 1: "part_level", 2: "occluded",
                                                         3: "class_name", 4: "original_name", 5: "attributes"})

    def get_dataset_size(self, dir_path):
        """
            Количество уникальных объектов датасета в папке
        ------------------------------------------------------
        Desc:
            Рекурсивно обходит указанную папку и считает количество файлов Img
            Файлы seg, parts, text не учитываются
        Input:
            * dir_path - имя главной папки обхода
        Output:
            * количество уникальных объектов
        """

        dataset_size = 0
        for file, path in self.data_gen(dir_path, return_path=True):
            name, description, train_or_val, parts_num = self.get_format(file)
            if description == 'img':
                dataset_size += 1
        return dataset_size

    def get_clear_name(self, name):
        """
            "ADE_train_00000218.jpg", "ADE_train_00000218_seg.png",
            "ADE_train_00000218_atr.txt", "ADE_train_00000218_parts_1.png"
            => "ADE_train_00000218"
        """

        return "_".join(name.split(".")[0].split("_")[:3])

    def get_img(self, name):
        """
            "ADE_train_00000218_atr.txt" => "ADE_train_00000218.jpg"
        """
        return self.get_clear_name(name) + ".jpg"

    def get_seg(self, name):
        """
            "ADE_train_00000218_atr.txt" => "ADE_train_00000218_seg.png"
        """
        return self.get_clear_name(name) + "_seg.png"

    def get_part(self, name, part_num):
        """
            "ADE_train_00000218_atr.txt" => "ADE_train_00000218_parts_{part_num}.png"
        """
        return self.get_clear_name(name) + "_parts_" + str(part_num) + ".png"

    def get_text(self, name):
        """
            "ADE_train_00000218_parts_1.txt" => "ADE_train_00000218_art.txt"
        """
        return self.get_clear_name(name) + "_atr.txt"

    def get_shape_arr(self, dir_path, progress_step=0):
        """
            Функция возвращает массив размеров всех изображений в папке
        ------------------------------------------------------
        Desc:
            Рекурсивно обходит указанную папку и для всех файлов img сохнаряет размер
            Файлы seg, parts, text не учитываются
        Input:
            * dir_path - имя главной папки обхода
            * progress_step <int> - шаг, с которым будет выводиться сообщения о прогрессе
                Done: <num> - обработано <num> фотографий
        Output:
            * np.array() с объектами (h, w) для каждой фотографии
        """
        shape_arr = []
        progress_counter = 0
        for file, path in self.data_gen(dir_path, return_path=True):
            name, description, train_or_val, parts_num = self.get_format(file)

            if description == 'img':
                img = imread(path + '/' + file)
                shape_arr.append(img.shape[:2])

                if progress_step != 0:
                    progress_counter += 1
                    if progress_counter % progress_step == 0:
                        print("Done: " + str(progress_counter))
        return np.array(shape_arr)

    def get_full_df_description(self, dir_path, only_part_level=-1, progress_step=0):
        """
            Возвращает DataFrame из всех объектов описания
        ------------------------------------------------------
        Desc:
            Рекурсивно обходит указанную папку и преобразует добавляет файл text в dataframe таблицу
        Input:
            * dir_path - имя главной папки обхода
            * progress_step <int> - шаг, с которым будет выводиться сообщения о прогрессе
                Done: <num> - обработано <num> фотографий
            * only_part_level <level> -
                -1, чтобы выводить все уровни,
                <level>, только указанный
        Output:
            * DataFrame всех объектов в папке
            columns=["instance_n", "part_level", "occluded", "class_name", "original_name", "attributes"]
        """
        df = pd.DataFrame(columns=["instance_n", "part_level", "occluded",
                                   "class_name", "original_name", "attributes"])

        if progress_step != 0:
            progress_counter = 0
            dataset_size = self.get_dataset_size(dir_path)
            print("Size: " + str(dataset_size))

        for file, path in self.data_gen(dir_path, return_path=True):
            name, description, train_or_val, parts_num = self.get_format(file)
            if description == 'text':
                df = df.append(self.get_df_description(path + '/' + file,
                                                       only_part_level=only_part_level),
                               ignore_index=True, sort=True)

                if progress_step != 0:
                    progress_counter += 1
                    if progress_counter % progress_step == 0:
                        print("Done: " + str(progress_counter))

        return df
