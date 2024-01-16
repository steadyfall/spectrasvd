from worker.compressor import (
    minLength,
    RGBImageCompression,
    grayScaleImageCompression,
)
import streamlit as st
import streamlit.components.v1 as components
import os

def noTIFFfiles(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.endswith(".tiff"):
                yield f

def main():  
    st.set_page_config(page_title="SVD Image Compression", layout="wide")
    st.header("Image Compression using Singular Value Decomposition (SVD)")
    chosenImage = "img/flower.jpg"

    col1, col2 = st.columns([6,4])

    with col1.container():
        st.write(
            "Choose from any one of the given images:"
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
    
    st.divider()

    placeholder = col2.empty()
    gray = st.toggle('Grayscale')
    if gray:
        compressor = grayScaleImageCompression(chosenImage)
    singVal = st.slider(
        "k-rank approximation", 
        1,
        minLength(chosenImage), 
        minLength(chosenImage),
    )
    placeholder.image(compressor(singVal))
    

if __name__ == "__main__":
    main()