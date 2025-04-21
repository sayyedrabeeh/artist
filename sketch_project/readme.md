# 🖼️ Image to Sketch Web App

This is a Django-based web application that allows users to upload an image and convert it into **six different sketch styles** using OpenCV. It's a fun and creative tool for transforming images into pencil-drawn art and sketchy effects.

---

## ✨ Features

- Upload any JPG image.
- Convert it into 6 different sketch styles:
  1. **Edge Sketch**
  2. **Enhanced Edge Sketch**
  3. **Simple Pencil Sketch**
  4. **Advanced Pencil Sketch**
  5. **Color Pencil Sketch**
  6. **Advanced Color Pencil Sketch**
- View and download all versions.

---

## 🖼️ Sample Output

For each uploaded image, the following are generated:

| Style                     | Filename Suffix               |
|--------------------------|-------------------------------|
| Edge Sketch              | `_edges.jpg`                  |
| Enhanced Edge Sketch     | `_edges1.jpg`                 |
| Simple Pencil Sketch     | `_pencil.jpg`                 |
| Advanced Pencil Sketch   | `_pencil1.jpg`                |
| Color Pencil Sketch      | `_pencil_colored.jpg`         |
| Advanced Color Pencil Sketch | `_pencil_colored1.jpg`     |

---

## 🛠️ How It Works

- The user uploads an image using a Django form.
- The uploaded image is saved and passed through a series of OpenCV transformations.
- The output images are saved with suffixes and displayed back to the user on the frontend.

Each function uses OpenCV techniques like:
- **Canny edge detection**
- **Gaussian blur**
- **Bilateral filtering**
- **Image blending and masking**
- **Texture and grain simulation**
- **Adaptive thresholding**

---

## 🧾 Requirements

Create a `requirements.txt` file in your project directory with the following:

```txt
Django>=3.2
opencv-python
numpy
