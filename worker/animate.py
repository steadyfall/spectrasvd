from PIL import Image, ImageFont, ImageDraw
import numpy as np, cv2, imageio.v2 as iio
from io import BytesIO

import typing as typ
from collections.abc import Callable

from .compressor import (
    minLength,
    AbstractImageCompressor,
)

def mp4Creator_imageio(
        compressor: Callable[[typ.Any], None] | AbstractImageCompressor, 
        imagePath: str, 
        duration: int
    ) -> bool:
    try:
        if compressor(1) is None:
            return False
    except AssertionError:
        return False
    else:
        if duration < 5 or (not isinstance(imagePath, str)): 
            return False
    videoName = ''.join(imagePath.split('.')[:-1])
    videoExt = ".mp4"
    backgroundMode = "L" if compressor(1).ndim == 2 else "RGB"
    maxRank = minLength(imagePath)

    # for writing on images
    fontSize = int((20 * maxRank) / 256)
    chosenFont = ImageFont.truetype(r'trebuc.ttf', fontSize)
    originalSize = Image.open(imagePath).size
    newSize = tuple([(9 * max(originalSize)) // 8 for _ in range(2)])
    newSize = tuple([l + 16 - (l % 16) for l in newSize])
    box = (
        (newSize[0] - originalSize[0]) // 2, 
        4 * (newSize[1] - originalSize[1]) // 5
    )

    # creating mp4 file
    with iio.get_writer(
        videoName + "-anim" + backgroundMode + videoExt, 
        mode='I', 
        fps=maxRank // duration
    ) as writer:
        for n in range(1, maxRank + 1):
            newIm = Image.new(backgroundMode, newSize, "White")
            with BytesIO() as output:
                with Image.fromarray(compressor(n), backgroundMode) as img:
                    newIm.paste(img, box)
                draw = ImageDraw.Draw(newIm)
                draw.text(
                    (newSize[0] // 2, (4 * box[1]) // 5), 
                    f"n = {n}", 
                    font = chosenFont,
                    fill = "Black",
                    anchor = "ms"
                )
                newIm.save(output, 'PNG')
                data = output.getvalue()
            image = iio.imread(data)
            writer.append_data(image)
    return True

def mp4Creator_cv2(
        compressor: Callable[[typ.Any], None] | AbstractImageCompressor, 
        imagePath: str, duration: int
    ) -> bool:
    try:
        if compressor(1) is None:
            return False
    except AssertionError:
        return False
    else:
        if duration < 5 or (not isinstance(imagePath, str)): 
            return False
    videoName = ''.join(imagePath.split('.')[:-1])
    videoExt = ".mp4"
    backgroundMode = "L" if compressor(1).ndim == 2 else "RGB"
    maxRank = minLength(imagePath)

    # for writing on images
    fontSize = int((20 * maxRank) / 256)
    chosenFont = ImageFont.truetype(r'trebuc.ttf', fontSize)
    originalSize = Image.open(imagePath).size
    newSize = tuple([(9 * max(originalSize)) // 8 for _ in range(2)])
    newSize = tuple([l + 16 - (l % 16) for l in newSize])
    box = (
        (newSize[0] - originalSize[0]) // 2, 
        4 * (newSize[1] - originalSize[1]) // 5
    )

    # creating mp4 file
    output = videoName + "-anim" + backgroundMode + videoExt
    framesPerSec = maxRank / duration
    with Image.new(backgroundMode, newSize, "White") as newIm:
        with Image.fromarray(compressor(1), backgroundMode) as img:
            newIm.paste(img, box)
        draw = ImageDraw.Draw(newIm)
        draw.text(
            (newSize[0] // 2, (4 * box[1]) // 5), 
            f"n = 1", 
            font = chosenFont,
            fill = "Black",
            anchor = "ms"
        )
        frame = cv2.cvtColor(
            np.array(newIm), 
            cv2.COLOR_RGB2BGR if backgroundMode == "RGB" else cv2.COLOR_GRAY2BGR
        )
        h, w, c = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'h265')
    # cv2.imshow('video',frame)
    out = cv2.VideoWriter(output, fourcc, framesPerSec, (w, h))

    for n in range(1, maxRank + 1):
        with Image.new(backgroundMode, newSize, "White") as newIm:
            with Image.fromarray(compressor(n), backgroundMode) as img:
                newIm.paste(img, box)
            draw = ImageDraw.Draw(newIm)
            draw.text(
                (newSize[0] // 2, (4 * box[1]) // 5), 
                f"n = {n}", 
                font = chosenFont,
                fill = "Black",
                anchor = "ms"
            )
            frame = cv2.cvtColor(
                np.array(newIm), 
                cv2.COLOR_RGB2BGR if backgroundMode == "RGB" else cv2.COLOR_GRAY2BGR
            )
            out.write(frame)
            # cv2.imshow('video',frame)
            """ if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                break """
    out.release()
    # cv2.destroyAllWindows()
    return True

