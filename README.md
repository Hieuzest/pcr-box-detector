# PCR Box Detector

For Princess Connect Re:DIVE

Auto detect characters and stars in box screenshot

However, the detector should be executed in ThreadPoolExecutor, due to its 0.3s cost

This repo contains a working example plugin for [HoshinoBot](https://github.com/Ice-Cirno/HoshinoBot)

#### Dependencies
```shell script
pip install opencv-contrib-python
```

##### If using SIFT:

SIFT is not patented now in OpenCV 3.4.10, 4.3.0 or above.

If no distribution found, try updating your pip using `pip install -U pip`

#### Usage

BoxDetector::detect(img) -> List[Tuple[CharaId, Star]]

-   accept image in OpenCv / PIL / bytes forms

#### Example
```python
import cv2
from box_detector import BoxDetector

box_detector = BoxDetector()
box_detector.set_config(path_to_icon=f'./unit')
box_detector.init()

img = cv2.imread("test.jpg")

print(box_detector.detect(img))
```

#### Build into HoshinoBot
A recommended workflow is:
```shell script
.../hoshino/modules/priconne/$ git submodule add -b master https://github.com/Hieuzest/pcr-box-detector box_detector
```
This should be enough to go.
