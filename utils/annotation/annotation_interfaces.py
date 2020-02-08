import copy
from typing import List, Dict, Tuple

"""
Object Detection Annotations Interface
[
    {
        "filepath": (str),
        "image_width" : (int),
        "image_height": (int),
        "objects":
                    [
                        {
                            "class" : (str),
                            "xmin" : (float),
                            "ymin" : (float),
                            "xmax" : (float),
                            "ymax" : (float),
                            "theta": (float) radian
                        },
                        ...
                    ]
    }
    ...
]
"""


class _BaseAnnoComponents:

    def __init__(self, info, required_keys=[]):
        self.REQUIRED_KEYS = required_keys

        self._validate_keys(info)

    def _validate_keys(self, info):
        info_keys = info.keys()
        validate_elements = [KEY in info_keys for KEY in self.REQUIRED_KEYS]
        if len(validate_elements) != sum(validate_elements):
            indices = [i for i, is_valid in enumerate(validate_elements) if is_valid is False]
            missing_keys = [self.REQUIRED_KEYS[idx] for idx in indices]
            raise RuntimeError("Must contain all required key, "
                               "missing key are `{}`".format(",".join(missing_keys)))

    def dump(self):
        raise NotImplementedError


class DetectionObject(_BaseAnnoComponents):

    def __init__(self, object_info: Dict) -> None:
        """
        Interface of Detection object property
        Args:
            object_info : object information as follow
                          {
                            "class" : (str), # class name
                            "xmin" : (float), # x coordinates of left top point
                            "ymin" : (float), # y coordinates of left top point
                            "xmax" : (float), # x coordinates of right bottom point
                            "ymax" : (float), # y coordinates of right bottom point
                            "theta": (float) #radian
                          }
        """
        self.REQUIRED_KEYS = ["class", "xmin", "ymin", "xmax", "ymax", "theta"]
        super(DetectionObject, self).__init__(object_info, self.REQUIRED_KEYS)

        self.CLASS: str = str(object_info["class"])
        self.XMIN: float = float(object_info["xmin"])
        self.YMIN: float = float(object_info["ymin"])
        self.XMAX: float = float(object_info["xmax"])
        self.YMAX: float = float(object_info["ymax"])
        self.THETA: float = float(object_info["theta"])

    def dump(self) -> None:
        """
        print function for DetectionObject
        """
        print("\t\tclass:\t{}".format(self.CLASS))
        print("\t\txmin:\t{}".format(self.XMIN))
        print("\t\tymin:\t{}".format(self.YMIN))
        print("\t\txmax:\t{}".format(self.XMAX))
        print("\t\tymax:\t{}".format(self.YMAX))
        print("\t\ttheta:\t{}\n".format(self.THETA))


class DetectionFile(_BaseAnnoComponents):

    def __init__(self, files_info: Dict) -> None:
        """
        Interface of Detection Image property
        Args:
            files_info (Dict) : files information as contain dict as follow
                             {
                                "filepath": (str),
                                "image_width" : (int),
                                "image_height": (int),
                                "objects":
                                [
                                    {"class" : "",
                                     "xmin" : (float),
                                     "ymin" : (float),
                                     "xmax" : (float),
                                     "ymax" : (float),
                                     "theta": (float) radian}
                                     ...
                                ]
                            }
        """
        self.REQUIRED_KEYS = ["filepath", "image_width", "image_height", "objects"]
        super(DetectionFile, self).__init__(files_info, self.REQUIRED_KEYS)

        self.FILEPATH: str = str(files_info["filepath"])
        self.IMAGE_WIDTH: int = int(files_info["image_width"])
        self.IMAGE_HEIGHT: int = int(files_info["image_height"])
        self.NUMBER_OF_OBJECTS: int = len(files_info["objects"])
        self.OBJECTS: List = self._parse_objects(files_info["objects"])

    @staticmethod
    def _parse_objects(objects: List) -> List:
        """
        parse object information from objects parameter
        """

        return [DetectionObject(obj) for obj in objects]

    def dump(self) -> None:
        """
        print function for DetectionFile
        """
        print("\tfilepath:\t{}".format(self.FILEPATH))
        print("\timage width:\t{}".format(self.IMAGE_WIDTH))
        print("\timage height:\t{}".format(self.IMAGE_HEIGHT))
        print("\tnumber of objects:\t{}".format(self.NUMBER_OF_OBJECTS))
        [OBJ.dump() for OBJ in self.OBJECTS]


class DetectionAnnotations(_BaseAnnoComponents):

    def __init__(self, anno_info: List):
        """
        Interface of Detection annotation property
        Args:
            anno_info: detection annotation info as follow
                       [{
                            "filepath": (str),
                            "image_width" : (int),
                            "image_height": (int),
                            "objects":
                            [
                                {"class" : "",
                                 "xmin" : (float),
                                 "ymin" : (float),
                                 "xmax" : (float),
                                 "ymax" : (float),
                                 "theta": (float) radian}
                                 ...
                            ]
                        }
                        ...
                        ]
        """

        self.NUMBER_OF_FILES = len(anno_info)
        self.FILES = self._parse_files(anno_info)
        self.FILES, self.NUMBER_OF_FILES = self._error_correction(self.FILES)

    def dump(self) -> None:
        """
        print function for DetectionAnnotations
        """
        print("\tnumber of files:\t{}".format(self.NUMBER_OF_FILES))
        [FILE.dump() for FILE in self.FILES]

    def _error_correction(self, files: List[DetectionFile]) -> Tuple[List[DetectionFile], int]:
        """
        check filepath overlap or not.
        merge DetectionFile objects when detected filepath overlap
        Args:
            files (List[DetectionFile]) : Object contain detection object information
        Returns:
            (Tuple[List[DetectionFile], int])
        """
        filepath_list = self._collect_filepath(files)
        unique_value = self._find_unique(filepath_list)

        if len(unique_value) == len(filepath_list):
            return files, len(files)

        for element in unique_value:
            overlap_indexs = self._find_index(filepath_list, element)

            if len(overlap_indexs) < 2:
                continue

            for idx in overlap_indexs:
                # ignore first index. cause others objs copy into first index
                # so, it should be preserved
                if idx == overlap_indexs[0]:
                    continue

                # copy into first index obj
                for obj in files[idx].OBJECTS:
                    copied_obj = copy.deepcopy(obj)
                    files[overlap_indexs[0]].OBJECTS.append(copied_obj)

                files[overlap_indexs[0]].NUMBER_OF_OBJECTS = len(files[overlap_indexs[0]].OBJECTS)

            for idx in overlap_indexs:
                # preserve first index obj
                if idx == overlap_indexs[0]:
                    continue

                # Do not use `del` keyword
                # we have fixed `overlap indexs`
                # `overlap indexs` will change when we delete file in list
                # because delete elements in list. it will be rearranged
                files[idx] = None

        files = [file for file in files if file is not None]
        return files, len(files)

    @staticmethod
    def _find_index(container: List, value: str) -> List:
        return [i for i, x in enumerate(container) if x == value]

    @staticmethod
    def _parse_files(anno_info: List):
        """
        parse annotation information from anno_info parameter
        """
        return [DetectionFile(anno) for anno in anno_info]

    @staticmethod
    def _collect_filepath(files: List[DetectionFile]) -> List:
        return [file.FILEPATH for file in files]

    @staticmethod
    def _find_unique(filepath_list: List) -> List:
        return list(set(filepath_list))


if __name__ == "__main__":
    # normal case
    case1 = [
        {
            "filepath": "a",
            "image_width": 0,
            "image_height": 0,
            "objects": [
                {
                    "class": "a",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "theta": 0
                },

                {
                    "class": "a",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "theta": 0
                },

                {
                    "class": "b",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "theta": 0
                }
            ]
        },

        {
            "filepath": "b",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "b",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    }
                ]
        },
        {
            "filepath": "a",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "c",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    }
                ]
        },
        {
            "filepath": "a",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    }
                ]
        },
        {
            "filepath": "a",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    },
                    {
                        "class": "c",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    }
                ]
        },
        {
            "filepath": "c",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0,
                        "theta": 0
                    }
                ]
        }
    ]

    # abnormal case2. missing filepath
    case2 = [
        {
            "image_width": 0,
            "image_height": 0,
            "objects": [
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                },
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                },
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                }
            ]
        },
        {
            "filepath": "",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        }
    ]

    # abnormal case3 missing `class`
    case3 = [
        {
            "image_width": 0,
            "image_height": 0,
            "objects": [
                {
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                },
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0

                },
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                }
            ]
        },
        {
            "filepath": "",
            "image_width": 0,
            "image_height": 0,
            "objects": [
                {
                    "class": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                }
            ]
        }
    ]

    # abnormal case4 missing whole annotations
    case4 = []

    try:
        # parsing logic test
        # Correct Error logic test
        annotations = DetectionAnnotations(case1)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        # key `filepath` missing in DetectionFile object
        annotations = DetectionAnnotations(case2)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        # key `class` missing in DetectionObject
        annotations = DetectionAnnotations(case3)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        # empty list
        annotations = DetectionAnnotations(case4)
        annotations.dump()
    except Exception as e:
        print(e)
