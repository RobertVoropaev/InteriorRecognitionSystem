from scripts_.PathFinder import PathFinder

class ClassList:
    def __init__(self, dir_path, min_obj=25, progress_step=0,
                 objectnames_path="static/objectnames.txt"):
        """
            Констуктор класса
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
        """

        df = PathFinder().get_full_df_description(dir_path, only_part_level=0,
                                                  progress_step=progress_step)

        indexes = list(df.class_name.value_counts() > min_obj)

        self.class_list = ['-'] + list(df.class_name.value_counts()[indexes].index)
        self.class_list = list(map(lambda x: [x], self.class_list))

        self.obj_names = []
        with open(objectnames_path, 'r') as f:
            for line in f:
                self.obj_names.append(line.strip())


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
