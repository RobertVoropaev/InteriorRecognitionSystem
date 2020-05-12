class SegEncoder:
    def __init__(self, objectnames_path="static/objectnames.txt"):
        """
            Конструктор класса
        ------------------------------------------------------
        Desc:
            Класс декодирования информации из пикселей
        Input:
            * objectnames_path - путь до файла objectnames.txt
        """

        self.obj_names = []
        with open(objectnames_path, 'r') as f:
            for line in f:
                self.obj_names.append(line.strip())

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
