from typing import Tuple, List, Callable, Any

import cv2
from PIL import Image
import numpy as np
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

    BRISK = 0
    SIFT = 1

    def __init__(self):
        self.inited = False
        self.descriptors = []
        self.path_to_icons = r'./res/img/priconne/unit/'
        self._path_to_icons = os.path.expanduser(self.path_to_icons)

        self.default_result = (1000, 3)

        self.classify_thresh = 15
        self.classify_distance_k = 3.0
        self.classify_calc_times = 10

        self.star_position_xs = [
            0.125, 0.25, 0.375, 0.5, 0.625
        ]
        self.star_position_ys = [0.86] * 5
        self.icon_norm_size = (64, 64)

        self.f_cid = lambda s: int(s[10:14])
        self.f_star = lambda s: int(s[14])

        self._extractor = cv2.BRISK_create(thresh=10, octaves=2)
        # self._extractor = cv2.xfeatures2d.SIFT_create()
        # self._extractor = cv2.ORB_create()
        # self._extractor = cv2.xfeatures2d.SURF_create()

        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        # self._matcher = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))

    def available(self) -> bool:
        """Test if detector is available

        :return: True if available
        """
        return self.inited

    def set_config(self, *,
                   path_to_icon: str = ...,
                   f_cid: Callable[[str], int] = ...,
                   f_star: Callable[[str], int] = ...,
                   default_result = ...,
                   extractor: Any = ...,
                   classify_thresh: int = ...,
                   classify_distance_k: float = ...,
                   classify_calc_times: int = ...,
                   icon_norm_size: Tuple[int, int] = ...
                   ):
        """Set config of classifier

        :param path_to_icon: path containing character icons
        :param f_cid: function convert filename to cid
        :param f_star: function convert filename to star
        :param default_result: return if not found
        :param extractor: BRISK or SIFT or cv2.FeatureDescriptor
        :param classify_thresh: number of matched points
        :param classify_distance_k: initialized distance thresh param
        :param classify_calc_times: hierarchical calc times
        :param icon_norm_size: normalized size of unit icons
        :return:
        """
        if path_to_icon is not ...:
            self.path_to_icons = path_to_icon
        if f_cid is not ...:
            self.f_cid = f_cid
        if f_star is not ...:
            self.f_star = f_star
        if default_result is not ...:
            self.default_result = default_result
        if extractor == self.BRISK:
            self._extractor = cv2.BRISK_create(thresh=10, octaves=1)
        elif extractor == self.SIFT:
            self._extractor = cv2.xfeatures2d.SIFT_create()
        elif extractor is not ...:
            self._extractor = extractor
        if classify_thresh is not ...:
            self.classify_thresh = classify_thresh
        if classify_distance_k is not ...:
            self.classify_distance_k = classify_distance_k
        if classify_calc_times is not ...:
            self.classify_calc_times = classify_calc_times
        if icon_norm_size is not ...:
            self.icon_norm_size = icon_norm_size

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
                img = cv2.resize(img, self.icon_norm_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, des = self._extractor.detectAndCompute(gray, None)

                self.descriptors.append(((cid, star), des))

        self.inited = True

    def _classify(self, img):
        img = cv2.resize(img, self.icon_norm_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, query_des = self._extractor.detectAndCompute(gray, None)

        dict_matches = {}
        for (cid, star), des in self.descriptors:
            matches = self._matcher.match(query_des, des)
            dict_matches[(cid, star)] = matches

        thresh_distance = self.icon_norm_size[0] * self.classify_distance_k
        retries = 0
        result = self.default_result

        while retries < self.classify_calc_times:
            max_matches = 0
            for key, matches in dict_matches.items():
                t = len(list(filter(lambda m: m.distance < thresh_distance, matches)))
                if t > max_matches:
                    max_matches = t
                    result = key

            if max_matches < 0.77 * self.classify_thresh:
                thresh_distance *= 1.16
            elif max_matches > 1.3 * self.classify_thresh:
                thresh_distance *= 0.86
            else:
                break
            retries += 1
        return result

    def detect(self, img: Any) -> List[Tuple[int, int]]:
        """Detect box characters

        :param img: Image of box, can be cv2 / PIL / bytes form
        :return: List[Tuple[cid, star]]
        """
        if not self.available():
            raise ValueError("Detector not initialized! Call init() first")

        if isinstance(img, bytes):
            img = cv2.imdecode(np.asarray(bytearray(img), dtype=np.uint8), cv2.IMREAD_COLOR).copy()
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif not isinstance(img, np.ndarray):
            raise TypeError("Unsupported type of img!")

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
                    x = int(x * w)
                    y = int(y * h)
                    if chara_im[y, x, 0] < 140:
                        stars += 1

            chara_info = (chara_id, stars)
            box.append(chara_info)

        return box
