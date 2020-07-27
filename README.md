# PCR Box Detector

For Princess Connect Re:DIVE

Auto detect characters and stars in box screenshot

**This method is SLOW so plz never call it in main thread**


#### Example
```python
import asyncio
import aiohttp
import cv2
import numpy as np

from box_detector import BoxDetector

_detector = BoxDetector()

# By default, the path is suitable for Hoshino Bot
_detector.set_config(path_to_icon=f"PATH_TO_UNINTS")

# Generally this should not be executed in main thread
# Took about 1.5 s
_detector.init()

...

# Must executed in worker thread
def detect_box(img):
    # Took about 6.5 s
    res = _detector.detect(img)

    # Store character data

    # Use synchronous method to send reply
    # Or create task in main event loop
    # Depends on framework

async def _handler(...):
    ...

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.content.read()
    
            file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
            asyncio.get_event_loop().run_in_executor(None, detect_box, img)

```