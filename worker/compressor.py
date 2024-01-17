from PIL import Image
import numpy as np, os
import requests
from io import BytesIO

import typing as typ
from collections.abc import Callable

AbstractImageCompressor = Callable[[Image.Image], np.ndarray]
ImageCompressor = Callable[[str], Callable[[typ.Any], None] | AbstractImageCompressor]

def minLength(imageObj: str | Image.Image) -> int | None:
    '''
    Wrapper for minimum of width and height for better understanding
    and less repeated code.
    '''
    if isinstance(imageObj, str):
        if imageObj.startswith("https://"):
            imageObj = Image.open(BytesIO(requests.get(imageObj).content))
        elif os.path.isfile(imageObj):
            imageObj = Image.open(imageObj).convert('L') if os.path.isfile(imageObj) else None
        else:
            return None
    elif not isinstance(imageObj, Image.Image):
        return None
    width, height = imageObj.size
    maxRank = min((width, height))
    return maxRank

def singleChannelImageCompressor(singleChannelImg: Image.Image) -> np.ndarray:
    '''
    Performs compression by SVD on an single channel image, while ensuring that
    image is single channel.
    '''
    if not (
        isinstance(singleChannelImg, Image.Image) 
        and singleChannelImg.mode == 'L'
    ):
        return lambda _: None
    maxRank = minLength(singleChannelImg)
    arr = np.asarray(singleChannelImg)
    U, S, Vh = np.linalg.svd(arr, full_matrices=False)
    rank = S.shape[0]
    assert rank <= maxRank

    def rank_k_approx(rank_k: int) -> np.ndarray:
        '''
        Rank-k approximation of a given matrix (in this case, a single-channel 
        image represented as a matrix).

        (1 <= rank_k <= rank <= maxRank)
        '''
        assert rank_k >= 1
        assert rank_k <= rank
        lowRankApprox = U[:, :rank_k] @ np.diag(S[:rank_k]) @ Vh[:rank_k, :]
        if rank_k == rank:
            lowRankApprox = U @ np.diag(S) @ Vh
        lowRankApprox = lowRankApprox.astype(np.uint8)
        return lowRankApprox
    return rank_k_approx

def multiChannelImageCompressor(multiChannelImg: Image.Image) -> np.ndarray:
    '''
    Performs compression by SVD on an multi channel image, while ensuring that
    image is multi channel.
    '''
    if not (
        isinstance(multiChannelImg, Image.Image) 
    ):
        return lambda _: None
    maxRank = minLength(multiChannelImg)
    channels = tuple(
        singleChannelImageCompressor(
            multiChannelImg.getchannel(chnl)
        )
        for chnl in multiChannelImg.getbands()
    )
    def rank_k_approx(rank_k: int) -> np.ndarray:
        '''
        Rank-k approximation of multiple matrices combined to give a
        `np.ndarray` object with depth >= 1, created using `np.dstack()`.

        NOTE: The `np.ndarray` object returned is ready to make images using
        the `Image.fromarray()` method.
        '''
        assert rank_k <= maxRank
        compressedChannels = tuple(
            chnl(rank_k) 
            for chnl in channels
        )
        try:
            lowRankApprox = np.dstack(compressedChannels)
        except Exception as e:
            return False
        else:
            return lowRankApprox
    return rank_k_approx

def singularValues(imageObj: Image.Image) -> np.array:
    '''
    Returns all the singular values returned from doing 
    SVD on the matrix given by the image.
    '''
    if not (
        isinstance(imageObj, Image.Image) 
    ):
        return np.ndarray([])
    channelSingularVal = tuple(
        np.linalg.svd(np.asarray(imageObj.getchannel(chnl)))[1]
        for chnl in imageObj.getbands()
    )
    return np.array(channelSingularVal)

def grayScaleImageCompression(
        imagePath: str
    ) -> Callable[[typ.Any], None] | AbstractImageCompressor:
    '''
    Returns an AbstractImageCompressor object for 
    compressing images in grayscale using SVD.
    '''
    if imagePath.startswith("https://"):
        im = Image.open(BytesIO(requests.get(imagePath).content)).convert('L')
    elif os.path.isfile(imagePath):
        im = Image.open(imagePath).convert('L')
    else:
        return lambda _: None
    return singleChannelImageCompressor(im)

def RGBImageCompression(
        imagePath: str
    ) -> Callable[[typ.Any], None] | AbstractImageCompressor:
    '''
    Returns an AbstractImageCompressor object for 
    compressing images in grayscale using SVD.
    '''
    if imagePath.startswith("https://"):
        im = Image.open(BytesIO(requests.get(imagePath).content))
    elif os.path.isfile(imagePath):
        im = Image.open(imagePath)
    else:
        return lambda _: None
    return multiChannelImageCompressor(im)

