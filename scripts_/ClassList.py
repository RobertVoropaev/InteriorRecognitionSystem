from scripts_.PathFinder import PathFinder
from scripts_.SegEncoder import SegEncoder


class ClassList:
    def __init__(self,
                 dir_path: str = "dir_path",
                 min_obj: int = 50,
                 progress_step: int = 0,
                 objectnames_path: str = "static/objectnames.txt",
                 load_class_encode: bool = False,
                 class_encode_path: str = "static/class_encode.txt"
                 ):
        """
            Констуктор создания нового списка классов
        ------------------------------------------------------
        Desc:
            Рекурсивно обходит указанную папку и преобразует добавляет файл text в dataframe таблицу
            Таблицу сортирует по количеству нахождения элеметов в датасете
            В таблицу попадают только элементы, чьё число вхождений превышает порог
        Input:
            * dir_path - имя главной папки обхода
            * min_obj - минимальное количество вхождений объекта в датасет
            * progress_step <int> - шаг, с которым будет выводиться сообщения о прогрессе
                Done: <num> - обработано <num> фотографий
            * objectnames_path - путь до файла objectnames.txt
            * load_class_encode - флаг загрузки готового списка (если True, то параметры выше не учитываются)
            * class_encode_path - путь до файла результата ClassList.save_class_encode()
        """

        self.se = SegEncoder()

        self.class_encode = []
        self.class_list = []

        if load_class_encode:
            self.load_class_encode(class_encode_path, objectnames_path)
        else:
            df = PathFinder().get_full_df_description(dir_path, only_part_level=0,
                                                      progress_step=progress_step)

            indexes = list(df.class_name.value_counts() > min_obj)

            self.class_list = ['-'] + list(df.class_name.value_counts()[indexes].index)
            self.class_list = list(map(lambda x: [x], self.class_list))

            self.obj_names = []
            with open(objectnames_path, 'r') as f:
                for line in f:
                    self.obj_names.append(line.strip())

            self.class_encode_is_load = False

    def remove_class(self, class_name):
        """
            Удаляет указанный класс из списка
        ------------------------------------------------------
        Desc:
            Если указанный класс был подклассом большого класса,
            то удаляется только указанный класс
        Input:
            * class_name - имя класса
        Output:
            * True - класс удалён
            * False - класс, не удалён (напр. его нет)
        """

        for i in range(len(self.class_list)):
            if class_name in self.class_list[i]:
                self.class_list[i].remove(class_name)

                if [] in self.class_list:
                    self.class_list.remove([])

                return True

        return False

    def remove_classes(self, class_list: list):
        """
            Удаляет массив классов
        ------------------------------------------------------
        Desc:
            В цикле вызывает remove_class
        Input:
            * class_list - список удаляемых классов
        Output:
            Массив классов, которые не были удалены из-за ошибки
        """

        exc_list = []  # классы, которые не были удалены
        for class_name in class_list:
            if not self.remove_class(class_name):
                exc_list.append(class_name)
        return exc_list

    def find_i(self, class_name):
        """
            Ищет класс в списке классов
        ------------------------------------------------------
        Desc:
            Возвращает номер мегакласса, в котором находится указанный класс
        Input:
            * class_name - имя класса
        Output:
            * i - индекс мегакласса, в котором находится указанный класс
            * -1 - такого класса нет
        """
        for i in range(len(self.class_list)):
            if class_name in self.class_list[i]:
                return i

        return -1

    def join(self, class_name_from, class_name_to):
        """
            Объединяет два мегакласса
        ------------------------------------------------------
        Input:
            * class_name_from - мегакласс, объекты которого будут перемещены
            * class_name_to - магакласс, в который будут перемещены объекты
        Output:
            * True - классы объеденены
            * False - классы не объеденены (одного из мегаклассов нет)
        """
        i_from = self.find_i(class_name_from)
        i_to = self.find_i(class_name_to)

        if i_from == -1 or i_to == -1 or i_from == i_to:
            return False

        for i in self.class_list[i_from]:
            self.class_list[i_to].append(i)

        self.class_list.remove(self.class_list[i_from])
        return True

    def join_list(self, join_list):
        """
            Объединяет мегаклассы по списку
        ------------------------------------------------------
        Input:
            * join_list - список элеметов (class_name_from, class_name_to)
        Output:
            Массив элементов, которые не были объеденены из-за ошибки
        """
        exc_list = []
        for class_from, class_to in join_list:
            if not self.join(class_from, class_to):
                exc_list.append([class_from, class_to])
        return exc_list


    def size(self):
        """
            Количество мегаклассов
        """
        return len(self.class_list)

    def save_class_list(self, filepath):
        """
            Сохраняет мегаклассы в файл
        ------------------------------------------------------
        Input:
            * filepath - путь до файла, в который будёт сохранятся список классов
        """
        with open(filepath, 'w') as f:
            for line in self.class_list:
                f.write(str(line) + '\n')

    ##################################################################################################

    def save_class_encode(self, filepath):
        """
            Сохраняет правила кодирования классов в файл
        ------------------------------------------------------
        Input:
            * filepath - путь до файла, в который будут сохрянятся правила кодирования
        """
        with open(filepath, 'w') as f:
            for i, class_arr in zip(range(len(self.class_list)), self.class_list):
                index_arr = []
                for class_name in class_arr:
                    index_arr.append(self.obj_names.index(class_name))
                index_arr = list(map(str, index_arr))
                f.write(str(i) + ';' + "|".join(index_arr) + ';' + "|".join(class_arr) + '\n')

    def load_class_encode(self,
                          class_encode_path: str = "static/class_encode.txt",
                          objectnames_path: str = "static/objectnames.txt"):
        """
            Загружает класс из списка
        ------------------------------------------------------
        Desc:
            Загружает данные списка классов из файла
        Input:
            * class_encode_path - путь до файла результата ClassList.save_class_encode()
            * objectnames_path - путь до файла objectnames.txt
        """

        with open(class_encode_path) as f:
            for line in f:
                line = line.strip()
                new_index, old_index_arr, class_name_arr = line.split(";")

                new_index = int(new_index)
                old_index_arr = list(map(int, old_index_arr.split("|")))
                class_name_arr = class_name_arr.split("|")
                self.class_encode.append([new_index, old_index_arr, class_name_arr])
                self.class_list.append(class_name_arr)

        self.class_encode_is_load = True

    def get_old_to_new_dict(self):
        """
            Возвращает словарь перевода из старого индекса в новый
        ------------------------------------------------------
        Output:
            Словарь старый индекс -> новый
        """

        if not self.class_encode_is_load:
            raise Exception("Class_encode don't load. Use load_class_encode().")

        index_old_to_new_dict = dict()

        for i in range(len(self.se.obj_names)):
            index_old_to_new_dict[i] = 0

        for line in self.class_encode:
            new_index = line[0]
            old_index_arr = line[1]
            for old_index in old_index_arr:
                index_old_to_new_dict[old_index] = new_index

        return index_old_to_new_dict
