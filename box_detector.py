from typing import Tuple, List, Callable

import cv2
import os
import itertools


def as_rect(a):
    x1, y1, w, h = a
    x2, y2 = x1 + w, y1 + h
    return x1, y1, x2, y2


def is_intersect(a, b):
    x1, y1, x2, y2 = as_rect(a)
    x3, y3, x4, y4 = as_rect(b)
    return x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3


def detect_bounding_box(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    cv2.threshold(img, 200, 255, cv2.THRESH_BINARY, img)

    area = img.shape[0] * img.shape[1]
    min_area, max_area = area / 75, area / 55
    min_square, max_square = 0.9, 1.1

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_area < w * h < max_area and min_square < w / h < max_square:
            boxes.append([x, y, w, h])

    while True:
        for a, b in itertools.permutations(boxes, 2):
            if a != b and is_intersect(a, b):
                if a[2] * a[3] < b[2] * b[3]:
                    boxes.remove(a)
                else:
                    boxes.remove(b)
                break
        else:
            break

    return boxes


class BoxDetector:

    def __init__(self):
        self.inited = False
        self.descriptors = []
        self.path_to_icons = r'./res/img/priconne/unit/'
        self._path_to_icons = os.path.expanduser(self.path_to_icons)

        self.classify_thresh = 10
        self.star_position_xs = [
            26, 50, 74, 98, 122
        ]
        self.star_position_ys = [
            170, 170, 170, 170, 170
        ]
        self.icon_normal_width = 197
        self.icon_normal_height = 197

        self.f_cid = lambda s: int(s[10:14])
        self.f_star = lambda s: int(s[14])

        # self._detector = cv2.BRISK(thresh=10, octaves=1)

        # self._extractor = cv2.DescriptorExtractor_create("BRISK")
        self._extractor = cv2.xfeatures2d.SIFT_create()
        # self._extractor = cv2.xfeatures2d.SURF_create()

        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        # index_params = dict(algorithm=0, trees=5)
        # search_params = dict(checks=50)
        # self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def available(self) -> bool:
        """Test if detector is available

        :return: True if available
        """
        return self.inited

    def set_config(self, *,
                   path_to_icon: str = ...,
                   f_cid: Callable[[str], int] = ...,
                   f_star: Callable[[str], int] = ...,
                   ):
        """Set config of Classifier

        :param path_to_icon: path containing character icons
        :param f_cid: function convert filename to cid
        :param f_star: function convert filename to star
        :return: None
        """
        if path_to_icon is not ...:
            self.path_to_icons = path_to_icon
        if f_cid is not ...:
            self.f_cid = f_cid
        if f_star is not ...:
            self.f_star = f_star

    def init(self):
        self.inited = False
        self.descriptors = []
        self._path_to_icons = os.path.expanduser(self.path_to_icons)

        files = os.listdir(self.path_to_icons)
        for file in files:
            if file.endswith('.png'):
                cid = self.f_cid(file)
                star = self.f_star(file)

                img = cv2.imread(os.path.join(self.path_to_icons, file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, des = self._extractor.detectAndCompute(gray, None)
                self.descriptors.append(((cid, star), des))

        self.inited = True

    def _classify(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, query_des = self._extractor.detectAndCompute(gray, None)

        dict_matches = {}
        for (cid, star), des in self.descriptors:
            matches = self._matcher.match(query_des, des)
            dict_matches[(cid, star)] = matches

        thresh_distance = img.shape[0]
        retries = 0
        result = (1000, 3)
        # print(f'----------')
        while retries < 6:
            # print(f'start from dist = {thresh_distance}')
            max_matches = 0
            for key, matches in dict_matches.items():
                t = len(list(filter(lambda m: m.distance < thresh_distance, matches)))
                if t > max_matches:
                    max_matches = t
                    result = key

            if max_matches < 0.75 * self.classify_thresh:
                thresh_distance *= 1.2
            elif max_matches > 1.25 * self.classify_thresh:
                thresh_distance *= 0.8
            else:
                break
            retries += 1

        return result

    def detect(self, img) -> List[Tuple[int, int]]:
        """Detect box characters

        :param img: Image of box, in OpenCV form
        :return: List[Tuple[cid, star]]
        """
        if not self.available():
            raise ValueError("Detector not initialized! Call init() first")

        bounding_boxes = detect_bounding_box(img)
        box = []
        for b in bounding_boxes:
            x, y, w, h = b
            chara_im = img[y:y + h, x:x + w]
            chara_id, icon_star = self._classify(chara_im)

            if icon_star == 6:
                stars = 6
            else:
                stars = 0
                for (x, y) in zip(self.star_position_xs, self.star_position_ys):
                    x = int(x * w / self.icon_normal_width)
                    y = int(y * h / self.icon_normal_height)
                    if chara_im[y, x, 0] < 140:
                        stars += 1

            chara_info = (chara_id, stars)
            box.append(chara_info)

        return box
