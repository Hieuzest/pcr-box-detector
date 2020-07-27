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

        self.sift_n_features = 200
        self.classify_thresh = 15
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

    def available(self) -> bool:
        """Test if detector is available

        :return: True if available
        """
        return self.inited

    def set_config(self, *,
                   path_to_icon: str = ...,
                   f_cid: Callable[[str], int] = ...,
                   f_star: Callable[[str], int] = ...,
                   sift_n_features: int = ...):
        """Set config of Classifier

        :param path_to_icon: path containing character icons
        :param f_cid: function convert filename to cid
        :param f_star: function convert filename to star
        :param sift_n_features: number of sift features
        :return: None
        """
        if path_to_icon is not ...:
            self.path_to_icons = path_to_icon
        if f_cid is not ...:
            self.f_cid = f_cid
        if f_star is not ...:
            self.f_star = f_star
        if sift_n_features is not ...:
            self.sift_n_features = sift_n_features

    def init(self):
        self.inited = False
        self.descriptors = []
        self.images = []
        self._path_to_icons = os.path.expanduser(self.path_to_icons)

        sift = cv2.xfeatures2d.SIFT_create(self.sift_n_features)

        files = os.listdir(self.path_to_icons)
        for file in files:
            if file.endswith('.png'):
                cid = self.f_cid(file)
                star = self.f_star(file)

                img = cv2.imread(os.path.join(self.path_to_icons, file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, des = sift.detectAndCompute(gray, None)

                self.descriptors.append(((cid, star), des))

        self.inited = True

    def _classify(self, img):
        if not self.descriptors:
            raise ValueError("Classifier not initialized! Call init() first")

        sift = cv2.xfeatures2d.SIFT_create(self.sift_n_features)
        query_kp, query_ds = sift.detectAndCompute(img, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        all_matches = {}
        for (cid, star), des in self.descriptors:
            matches = flann.knnMatch(query_ds, des, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            all_matches[(cid, star)] = len(good)

        max_matches = None
        result = None
        for (cid, star), matches in all_matches.items():
            if max_matches is None or matches > max_matches:
                max_matches = matches
                result = (cid, star)

        return result if max_matches > self.classify_thresh else None

    def detect(self, img) -> List[Tuple[int, int]]:
        """Detect box characters

        :param img: Image of box, in OpenCV form
        :return: List[Tuple[cid, star]]
        """
        bounding_boxes = detect_bounding_box(img)
        box = []
        for b in bounding_boxes:
            x, y, w, h = b
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
