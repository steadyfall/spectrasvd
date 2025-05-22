# SpectraSVD
Image compression using SVD (Singular Value Decomposition) in Python using NumPy, Pillow and Matplotlib.

## Demo
You can demo the project by either going to the mentioned link OR running it locally on a container:
```bash
docker run -p 8501:8501 ghcr.io/steadyfall/spectrasvd:v1.0.0
python -m webbrowser https://localhost:8501
```

## Libraries used:
- `streamlit`
- `Pillow`
- `numpy`
- `opencv-python`
- `imageio`
- `matplotlib`

---

Credit for [sample images](./img):
- [USC-SIPI Image Database (Volume 3: Miscellaneous)](https://sipi.usc.edu/database/database.php?volume=misc)
- [EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)
