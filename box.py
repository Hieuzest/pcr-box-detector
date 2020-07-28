import asyncio
import aiohttp
import cv2
import numpy as np

from itertools import starmap

from hoshino import Service

from .box_detector import BoxDetector
from .. import chara

is_hoshino_v2 = hasattr(Service, 'get_bundles')

if is_hoshino_v2:
    sv = Service('pcr-box', bundle='pcrbox', help_='''
[box] <图片> [<图片> ...]
'''.strip())
    fromid = chara.fromid
else:
    sv = Service('pcr-box')
    fromid = chara.Chara.fromid

_detector = BoxDetector()
_detector.init()

# @sv.on_prefix('box')
async def _handler(bot, event):
    for ms in event.message:
        if ms.type == 'image':
            sv.logger.info(f'Starting detecting box from image ...')

            try:
                url = ms.data['url']
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        content = await response.content.read()
                        file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR).copy()

                        res = await asyncio.get_event_loop().run_in_executor(None, _detector.detect, img)
                        res = starmap(lambda cid, star: f'{star}★{fromid(cid, star=star).name}', res)
                        await bot.send(event, "图中有：" + "  ".join(res), at_sender=True)

            except Exception as e:
                sv.logger.error('Error occurred when fetching image: ')
                sv.logger.exception(e)

if is_hoshino_v2:
    sv.on_prefix('box')(_handler)
else:
    sv.on_command('box')(lambda session: _handler(session.bot, session.event))
