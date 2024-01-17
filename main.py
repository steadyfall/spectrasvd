from worker.compressor import (
    minLength,
    RGBImageCompression,
    grayScaleImageCompression,
)
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import os, io, requests
from types import NoneType
from functools import reduce

def noTIFFfiles(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.endswith(".tiff"):
                yield f

def main():  
    st.set_page_config(page_title="SpectraSVD", layout="wide")
    st.header("SpectraSVD")
    chosenImage = "https://raw.githubusercontent.com/steadyfall/svd-compression/main/img/flower.JPEG"

    col1, col2 = st.columns([4,6])
    placeholderForImage = col1.empty()

    with col2.container():
        st.write(
            "Choose from any one of the given images."
        )
        imageCarouselComponent = components.declare_component(
            "image-carousel-component", 
            path="image-carousel/frontend/public"
        )
        imgList = list(
            map(
                lambda fi: f"https://raw.githubusercontent.com/steadyfall/svd-compression/main/img/{fi}",
                noTIFFfiles("img")
            )
        )
        selectedImageUrl = imageCarouselComponent(imageUrls=imgList, height=300)
        if selectedImageUrl is not None:
            chosenImage = str(selectedImageUrl)

    compressor = RGBImageCompression(chosenImage)
    
    variable, compressionInfo = st.columns(2)

    gray = variable.toggle('Grayscale')
    if gray:
        compressor = grayScaleImageCompression(chosenImage)
    singVal = variable.slider(
        "k-rank approximation", 
        1,
        minLength(chosenImage), 
        minLength(chosenImage),
    )
    render = compressor(singVal)
    placeholderForImage.image(render if not isinstance(render, NoneType) else chosenImage)

    size = Image.open(io.BytesIO(requests.get(chosenImage).content)).size
    pixels = reduce(lambda acc, val: acc * val, size)
    compressedSize = singVal * (sum(size) + 1)
    compressionInfo.write(f"- Number of pixels: **{pixels}**")
    compressionInfo.write(f"- COMPRESSED SIZE (approximately proportional to): **{compressedSize}**")
    compressionInfo.write(f"- Compression Ratio = {pixels}/{compressedSize} = **{round(pixels/compressedSize, 3)}**")

    st.divider()

    

if __name__ == "__main__":
    main()